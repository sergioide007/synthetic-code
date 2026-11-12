"""distributed.py behaviours that mutation testing showed were unpinned."""
import json, threading, time
from sintetico.distributed import (RESULTS_TOPIC, TASKS_TOPIC, AgentWorker, DistributedAIGateway, InMemoryBus, ResultEnvelope,
                                   TaskEnvelope, KafkaBus)


def put(bus, e): bus.publish(TASKS_TOPIC, e.to_bytes(), key=e.task_id)


def test_worker_defaults_are_the_documented_ones():
    w = AgentWorker(InMemoryBus(), lambda t: None)
    assert (w.max_attempts, w.task_timeout, w.backoff_base, w.backoff_max, w._dedupe_size) == (3, 60.0, 0.5, 30.0, 10_000)
    assert (w.tasks_topic, w.results_topic, w.dlq_topic) == ("agent.tasks", "agent.results", "agent.tasks.dlq") and w.worker_id.startswith("worker-")
    assert len(w.worker_id) == len("worker-") + 8 and not w._stop.is_set()
    g = DistributedAIGateway(InMemoryBus())
    assert (g.max_pending, g.max_retained, g.default_timeout, g.ready_timeout) == (1000, 10_000, None, 15.0) and g.gateway_id.startswith("gw-")
    assert w.worker_id != AgentWorker(InMemoryBus(), lambda t: None).worker_id


def test_background_threads_are_daemons_so_they_never_block_interpreter_exit():
    bus = InMemoryBus(); w = AgentWorker(bus, lambda t: None); th = w.start(poll_timeout=0.02)
    assert th.daemon and th.name == w.worker_id
    gw = DistributedAIGateway(bus); gw.start()
    assert gw._thread.daemon and gw._thread.name == gw.gateway_id
    gw.close(); w.stop(); th.join(3)


def test_a_new_gateway_starts_at_the_end_of_the_results_topic_and_never_replays_history():
    bus = InMemoryBus()
    for i in range(100):
        bus.publish(RESULTS_TOPIC, ResultEnvelope(f"old-{i}", "completed", "someone-else", "x").to_bytes())
    gw = DistributedAIGateway(bus, gateway_id="fresh"); gw.start()
    assert bus.lag("gateway-fresh", RESULTS_TOPIC) == 0                                       # nothing from before it existed
    gw.close()


def test_expired_tasks_are_counted_in_the_worker_metrics():
    bus = InMemoryBus(); w = AgentWorker(bus, lambda t: "r", clock=lambda: 100.0, sleep=lambda s: None)
    put(bus, TaskEnvelope("late", "x", {}, "gw", 1, 0.0, 99.0)); put(bus, TaskEnvelope("ok", "x", {}, "gw", 1, 0.0, None))
    assert w.run_once(0.05) == "timeout" and w.run_once(0.05) == "completed"
    assert dict(w.metrics) == {"timeout": 1, "completed": 1}


def test_worker_metrics_account_for_every_outcome():
    bus = InMemoryBus(); n = {"c": 0}
    def handler(t):
        n["c"] += 1
        if t.task_id == "flaky" and n["c"] < 2: raise __import__("sintetico.distributed", fromlist=["x"]).RetryableError("x")
        if t.task_id == "bug": raise ValueError("x")
        return "r"
    w = AgentWorker(bus, handler, sleep=lambda s: None)
    for tid in ("flaky", "bug", "fine", "fine"): put(bus, TaskEnvelope(tid, "x", {}, "gw"))
    bus.publish(TASKS_TOPIC, b"junk")
    while w.run_once(0.02): pass
    assert dict(w.metrics) == {"retried": 1, "failed": 1, "dlq": 2, "completed": 2, "duplicates": 1, "poison": 1}


def test_results_are_keyed_by_the_gateway_so_a_partitioned_topic_keeps_one_callers_replies_together():
    bus = InMemoryBus(); w = AgentWorker(bus, lambda t: "r", sleep=lambda s: None)
    put(bus, TaskEnvelope("t1", "x", {}, "gw-A")); put(bus, TaskEnvelope("t2", "x", {}, "gw-B")); w.run_once(0.05); w.run_once(0.05)
    assert [m.key for m in bus.log(RESULTS_TOPIC)] == ["gw-A", "gw-B"]
    assert [m.key for m in bus.log(TASKS_TOPIC)] == ["t1", "t2"]                              # tasks are keyed by id: ordering per task


def test_kafkabus_defaults_and_requires_either_the_library_or_all_three_factories():
    bus = KafkaBus("h:1", producer_factory=lambda c: type("P", (), {"flush": lambda s, t: 0})(), consumer_factory=lambda c: None, admin_factory=lambda c: None)
    assert (bus.flush_timeout, bus.partitions, bus.replication) == (10.0, 1, 1) and bus.base == {"bootstrap.servers": "h:1"}


def test_a_stuck_handler_never_blocks_the_worker_itself():
    bus, release = InMemoryBus(), threading.Event()
    def handler(t):
        if t.task_id == "stuck": release.wait(10)
        return "ok"
    w = AgentWorker(bus, handler, task_timeout=0.1, max_attempts=1, sleep=lambda s: None)
    put(bus, TaskEnvelope("stuck", "x", {}, "gw")); put(bus, TaskEnvelope("next", "x", {}, "gw"))
    t0 = time.monotonic(); first = w.run_once(0.05); took = time.monotonic() - t0
    assert first == "timeout" and took < 1.0                                                   # gave up after ~0.1s; did NOT wait for the 10s handler
    t0 = time.monotonic(); assert w.run_once(0.05) == "completed" and time.monotonic() - t0 < 1.0     # and a fresh thread serves the next task immediately
    release.set(); w.close()


def test_message_and_envelope_value_semantics():
    from sintetico.distributed import BusMessage
    a, b = BusMessage("t", "k", b"v", 1, raw=object()), BusMessage("t", "k", b"v", 1, raw=object())
    assert a == b and "raw" not in repr(a) and (a.partition, a.raw is not None) == (0, True)    # the client's own object never affects equality or logs
    e = TaskEnvelope("id", "type", {}, "gw")
    assert (e.attempt, e.created_at, e.deadline) == (1, 0.0, None)


def test_run_forever_defaults():
    import inspect
    sig = inspect.signature(AgentWorker.run_forever).parameters
    assert (sig["poll_timeout"].default, sig["error_backoff"].default) == (0.5, 1.0)


def test_dedupe_memory_is_bounded_to_exactly_its_size():
    bus = InMemoryBus(); w = AgentWorker(bus, lambda t: "r", dedupe_size=2, sleep=lambda s: None)
    for i in range(3): put(bus, TaskEnvelope(f"t{i}", "x", {}, "gw"))
    while w.run_once(0.02): pass
    assert len(w._done) == 2 and list(w._done) == ["t1", "t2"]
    for tid in ("t1", "t2", "t0"): put(bus, TaskEnvelope(tid, "x", {}, "gw"))
    outcomes = []
    while (r := w.run_once(0.02)): outcomes.append(r)
    assert outcomes == ["duplicate", "duplicate", "completed"]                                  # the two newest are remembered, the oldest was forgotten


def test_a_missing_created_at_defaults_to_zero():
    raw = json.dumps({"task_id": "a", "task_type": "x", "payload": {}, "reply_to": "g"}).encode()
    e = TaskEnvelope.from_bytes(raw)
    assert (e.created_at, e.attempt, e.deadline) == (0.0, 1, None)


def test_bus_errors_are_counted_once_per_failure():
    bus, failing = InMemoryBus(), [True]
    bus.fail_publish = lambda topic: topic == RESULTS_TOPIC and failing[0]
    w = AgentWorker(bus, lambda t: "r", sleep=lambda s: None); put(bus, TaskEnvelope("t", "x", {}, "gw"))
    th = w.start(poll_timeout=0.02, error_backoff=0.6)
    time.sleep(0.25)                                                                            # first failure happened; the loop is in its 0.6s backoff
    assert w.metrics["bus_errors"] == 1
    failing[0] = False; w.stop(); th.join(3)


def test_submitted_envelopes_carry_attempt_one_and_exact_timestamps_and_deadlines():
    now = [1000.0]; bus = InMemoryBus()
    with DistributedAIGateway(bus, clock=lambda: now[0], default_timeout=30.0) as gw:
        gw.submit_task("a", {}, timeout=5.0); gw.submit_task("b", {}); gw.submit_task("c", {}, timeout=0)
    a, b, c = [TaskEnvelope.from_bytes(m.value) for m in bus.log(TASKS_TOPIC)]
    assert (a.attempt, a.created_at, a.deadline) == (1, 1000.0, 1005.0)
    assert b.deadline == 1030.0                                                                  # default_timeout applies when none is given
    assert c.deadline is None and a.reply_to == gw.gateway_id and a.task_type == "a"             # an explicit 0 means "no deadline"
