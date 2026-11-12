"""KafkaBus verification, at three levels of fidelity (see docs/VERIFICATION.md):
  1. the REAL librdkafka client with no broker: configuration validation and failure behaviour;
  2. contract doubles built with the real confluent_kafka types, for every error path;
  3. a real-broker test, skipped unless SINTETICO_TEST_KAFKA=host:port is set (CI provides one).
"""
import os, time, uuid
import pytest
ck = pytest.importorskip("confluent_kafka")
from sintetico.distributed import (RESULTS_TOPIC, TASKS_TOPIC, AgentWorker, BusError, DistributedAIGateway, KafkaBus, TaskEnvelope)
from sintetico.orchestration import TaskStatus

DEAD = "127.0.0.1:1"                                     # nothing listens here


# ============================================================ 1. real librdkafka, no broker
def test_real_clients_accept_the_configuration_kafkabus_builds():
    bus = KafkaBus(DEAD, flush_timeout=0.5)               # builds a real Producer with idempotence + acks=all (librdkafka validates the combination)
    c = bus.consumer([TASKS_TOPIC], "g1", from_beginning=False)
    c.close(); bus.close()


def test_invalid_client_configuration_is_reported_as_a_bus_error():
    with pytest.raises(BusError, match="producer configuration"):
        KafkaBus(DEAD, client_config={"no.such.property": "x"})                             # real librdkafka rejects unknown properties
    with pytest.raises(BusError, match="Invalid value"):
        KafkaBus(DEAD, client_config={"session.timeout.ms": "not-a-number"})                # ... and invalid values
    def bad_consumer(cfg): raise ck.KafkaException(ck.KafkaError(ck.KafkaError._INVALID_ARG))
    bus = KafkaBus(DEAD, consumer_factory=bad_consumer)
    with pytest.raises(BusError, match="consumer configuration"):
        bus.consumer(["t"], "g")


def test_publishing_to_an_unreachable_broker_fails_fast_with_a_bus_error_instead_of_hanging():
    bus = KafkaBus(DEAD, flush_timeout=0.6)
    t0 = time.monotonic()
    with pytest.raises(BusError):
        bus.publish(TASKS_TOPIC, b"x", key="k")
    assert time.monotonic() - t0 < 5


def test_creating_topics_on_an_unreachable_broker_is_a_bus_error():
    with pytest.raises(BusError):
        KafkaBus(DEAD, flush_timeout=0.6).ensure_topics(["a"])


def test_polling_an_unreachable_broker_never_returns_a_message_and_never_hangs():
    c = KafkaBus(DEAD, flush_timeout=0.5).consumer([TASKS_TOPIC], "g-dead")
    t0 = time.monotonic()
    try:
        assert c.poll(0.5) is None
    except BusError:
        pass                                              # librdkafka may surface "all brokers down" as an error event
    assert time.monotonic() - t0 < 5 and c.wait_ready(0.3) is False
    c.close()


# ================================================================ 2. contract doubles with real types
class Msg:
    def __init__(self, topic="t", key=b"k", value=b"v", offset=7, partition=2, err=None): self._a = (topic, key, value, offset, partition, err)
    def topic(self): return self._a[0]
    def key(self): return self._a[1]
    def value(self): return self._a[2]
    def offset(self): return self._a[3]
    def partition(self): return self._a[4]
    def error(self): return self._a[5]


class Client:
    def __init__(self, cfg=None, queue=(), assignment=True):
        self.cfg, self.queue, self.calls, self._assigned = cfg, list(queue), [], assignment
        self.fail_commit = False
    def subscribe(self, topics): self.calls.append(("subscribe", topics))
    def poll(self, timeout=None): return self.queue.pop(0) if self.queue else None
    def commit(self, **kw):
        self.calls.append(("commit", kw))
        if self.fail_commit: raise ck.KafkaException(ck.KafkaError(ck.KafkaError._FAIL))
    def seek(self, tp): self.calls.append(("seek", tp))
    def assignment(self): return [ck.TopicPartition("t", 0)] if self._assigned else []
    def close(self): self.calls.append(("close",))


class Producer:
    def __init__(self, cfg=None, deliver_error=None, remaining=0, raises=None):
        self.cfg, self.sent, self.deliver_error, self.remaining, self.raises = cfg, [], deliver_error, remaining, raises
    def produce(self, topic, value=None, key=None, on_delivery=None):
        if self.raises: raise self.raises
        self.sent.append((topic, value, key)); on_delivery(self.deliver_error, None)
    def flush(self, timeout): return self.remaining


class _Admin:
    def create_topics(self, new): return {}                                                  # nothing to wait for


def kbus(client=None, producer=None, **kw):
    made = {}
    def cf(cfg): made["consumer_cfg"] = cfg; return client or Client(cfg)
    def pf(cfg): made["producer_cfg"] = cfg; return producer or Producer(cfg)
    return KafkaBus("broker:9092", producer_factory=pf, consumer_factory=cf, admin_factory=lambda cfg: _Admin(), **kw), made


def test_producer_is_idempotent_with_acks_all_and_consumer_commits_manually():
    bus, made = kbus(client=Client())
    bus.consumer(["a", "b"], "grp", from_beginning=True); bus.consumer(["a"], "grp2", from_beginning=False)
    assert made["producer_cfg"] == {"bootstrap.servers": "broker:9092", "enable.idempotence": True, "acks": "all"}
    assert made["consumer_cfg"] == {"bootstrap.servers": "broker:9092", "group.id": "grp2", "enable.auto.commit": False, "auto.offset.reset": "latest"}
    bus2, made2 = kbus(client=Client(), client_config={"security.protocol": "SASL_SSL"}); bus2.consumer(["a"], "g", from_beginning=True)
    assert made2["consumer_cfg"]["auto.offset.reset"] == "earliest" and made2["consumer_cfg"]["security.protocol"] == "SASL_SSL"


def test_poll_converts_messages_and_ignores_end_of_partition_but_raises_real_errors():
    eof = ck.KafkaError(ck.KafkaError._PARTITION_EOF)
    broker_down = ck.KafkaError(ck.KafkaError._ALL_BROKERS_DOWN)
    client = Client(queue=[Msg(key=b"order-1", value=b"payload"), Msg(key=None, offset=8), Msg(err=eof), Msg(err=broker_down)])
    c = kbus(client=client)[0].consumer(["t"], "g")
    m = c.poll(0.1)
    assert (m.topic, m.key, m.value, m.offset, m.partition) == ("t", "order-1", b"payload", 7, 2) and m.raw is not None
    assert c.poll(0.1).key is None and c.poll(0.1) is None
    with pytest.raises(BusError, match="consumer error"): c.poll(0.1)
    assert c.poll(0.1) is None                                                           # empty queue


def test_commit_is_synchronous_on_the_original_message_and_failures_become_bus_errors():
    client = Client(queue=[Msg()]); c = kbus(client=client)[0].consumer(["t"], "g"); m = c.poll(0.1)
    c.commit(m)
    assert client.calls[-1][0] == "commit" and client.calls[-1][1]["asynchronous"] is False and client.calls[-1][1]["message"] is m.raw
    client.fail_commit = True
    with pytest.raises(BusError, match="commit failed"): c.commit(m)


def test_nack_rewinds_to_exactly_the_failed_offset():
    client = Client(queue=[Msg(topic="agent.tasks", offset=41, partition=3)]); c = kbus(client=client)[0].consumer(["agent.tasks"], "g")
    c.nack(c.poll(0.1))
    kind, tp = client.calls[-1]
    assert kind == "seek" and tp == ck.TopicPartition("agent.tasks", 3, 41)


def test_wait_ready_does_not_lose_a_message_received_while_waiting():
    class Late(Client):
        def __init__(self): super().__init__(queue=[Msg(value=b"early")]); self.polls = 0
        def assignment(self): return [ck.TopicPartition("t", 0)] if self.polls >= 1 else []
        def poll(self, timeout=None): self.polls += 1; return super().poll(timeout)
    c = kbus(client=Late())[0].consumer(["t"], "g")
    assert c.wait_ready(2.0) is True and c.poll(0.1).value == b"early"                   # buffered, not dropped
    assert kbus(client=Client(assignment=False))[0].consumer(["t"], "g").wait_ready(0.25) is False


def test_publish_success_and_every_failure_mode():
    prod = Producer(); bus = kbus(producer=prod)[0]
    bus.publish("topic", b"data", key="k"); assert prod.sent == [("topic", b"data", "k")]
    with pytest.raises(BusError, match="delivery to topic failed"):
        kbus(producer=Producer(deliver_error=ck.KafkaError(ck.KafkaError._MSG_TIMED_OUT)))[0].publish("topic", b"x")
    with pytest.raises(BusError, match="1 message"):
        kbus(producer=Producer(remaining=1))[0].publish("topic", b"x")
    with pytest.raises(BusError, match="produce to topic failed"):
        kbus(producer=Producer(raises=BufferError("queue full")))[0].publish("topic", b"x")


def test_ensure_topics_tolerates_existing_topics_but_not_other_errors():
    from concurrent.futures import Future
    def fut(exc=None):
        f = Future(); f.set_exception(exc) if exc else f.set_result(None); return f
    class Admin:
        def __init__(self, results): self.results, self.created = results, None
        def create_topics(self, new): self.created = new; return {n.topic: self.results[i] for i, n in enumerate(new)}
    ok = Admin([fut(), fut(ck.KafkaException(ck.KafkaError(ck.KafkaError.TOPIC_ALREADY_EXISTS)))])
    bus = KafkaBus("b:1", producer_factory=lambda c: Producer(), consumer_factory=lambda c: Client(), admin_factory=lambda c: ok, partitions=3, replication_factor=2)
    bus.ensure_topics(["a", "b"])
    assert [(n.topic, n.num_partitions, n.replication_factor) for n in ok.created] == [("a", 3, 2), ("b", 3, 2)]
    bad = Admin([fut(ck.KafkaException(ck.KafkaError(ck.KafkaError.TOPIC_AUTHORIZATION_FAILED)))])
    bus2 = KafkaBus("b:1", producer_factory=lambda c: Producer(), consumer_factory=lambda c: Client(), admin_factory=lambda c: bad)
    with pytest.raises(BusError, match="could not create topic a"): bus2.ensure_topics(["a"])


def test_a_worker_over_kafkabus_commits_only_after_publishing_the_result_and_rewinds_on_failure():
    order, task = [], TaskEnvelope("t1", "echo", {"n": 1}, "gw")
    client = Client(queue=[Msg(topic=TASKS_TOPIC, value=task.to_bytes(), offset=5, partition=0)])
    class Prod(Producer):
        def produce(self, topic, value=None, key=None, on_delivery=None): order.append(("produce", topic)); super().produce(topic, value, key, on_delivery)
    class Cl(Client):
        def commit(self, **kw): order.append(("commit",)); super().commit(**kw)
    client.__class__ = Cl
    bus, _ = kbus(client=client, producer=Prod())
    w = AgentWorker(bus, lambda t: "done", sleep=lambda s: None)
    assert w.run_once(0.1) == "completed" and order == [("produce", RESULTS_TOPIC), ("commit",)]
    flaky = Client(queue=[Msg(topic=TASKS_TOPIC, value=task.to_bytes(), offset=9)])
    w2 = AgentWorker(kbus(client=flaky, producer=Producer(remaining=1))[0], lambda t: "done", sleep=lambda s: None)
    with pytest.raises(BusError): w2.run_once(0.1)
    assert [c[0] for c in flaky.calls if c[0] in ("commit", "seek")] == ["seek"] and flaky.calls[-1][1] == ck.TopicPartition(TASKS_TOPIC, 2, 9)     # not committed; rewound


# ===================================================================== 3. real broker (opt-in)
REAL = os.environ.get("SINTETICO_TEST_KAFKA")


@pytest.mark.skipif(not REAL, reason="set SINTETICO_TEST_KAFKA=host:port to run against a real Kafka broker")
class TestRealBroker:
    @pytest.fixture()
    def names(self):
        s = uuid.uuid4().hex[:8]
        return {"tasks_topic": f"tasks-{s}", "results_topic": f"results-{s}", "dlq_topic": f"dlq-{s}", "group": f"workers-{s}"}

    def _worker(self, bus, n, handler, **kw):
        return AgentWorker(bus, handler, group_id=n["group"], tasks_topic=n["tasks_topic"], results_topic=n["results_topic"], dlq_topic=n["dlq_topic"],
                           sleep=lambda s: None, **kw)

    def test_round_trip_through_a_real_broker(self, names):
        bus = KafkaBus(REAL)
        w = self._worker(bus, names, lambda t: {"echo": t.payload}); th = w.start(poll_timeout=0.2)
        with DistributedAIGateway(bus, tasks_topic=names["tasks_topic"], results_topic=names["results_topic"]) as gw:
            ids = [gw.submit_task("echo", {"i": i}) for i in range(5)]
            tasks = [gw.wait_for_result(i, 60) for i in ids]
        w.stop(); th.join(10)
        assert all(t.status is TaskStatus.COMPLETED for t in tasks)

    def test_poison_message_and_failing_handler_reach_the_dlq(self, names):
        bus = KafkaBus(REAL)
        def handler(t): raise RuntimeError("boom")
        w = self._worker(bus, names, handler, max_attempts=1); th = w.start(poll_timeout=0.2)
        bus.ensure_topics([names["tasks_topic"], names["dlq_topic"]]); bus.publish(names["tasks_topic"], b"garbage")
        probe = bus.consumer([names["dlq_topic"]], f"probe-{names['group']}"); seen = []
        deadline = time.time() + 60
        while len(seen) < 1 and time.time() < deadline:
            m = probe.poll(0.5)
            if m: seen.append(m)
        w.stop(); th.join(10); probe.close()
        assert seen and b"undecodable" in seen[0].value

    def test_uncommitted_work_is_redelivered_after_a_consumer_leaves(self, names):
        bus = KafkaBus(REAL); bus.ensure_topics([names["tasks_topic"]])
        task = TaskEnvelope("t-redeliver", "echo", {}, "gw")
        bus.publish(names["tasks_topic"], task.to_bytes(), key=task.task_id)
        first = bus.consumer([names["tasks_topic"]], names["group"])
        m = None; deadline = time.time() + 60
        while m is None and time.time() < deadline: m = first.poll(0.5)
        assert m is not None; first.close()                                                   # leaves WITHOUT committing
        second = bus.consumer([names["tasks_topic"]], names["group"]); again = None; deadline = time.time() + 60
        while again is None and time.time() < deadline: again = second.poll(0.5)
        second.close()
        assert again is not None and again.value == m.value and again.offset == m.offset
