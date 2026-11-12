import base64, json, threading, time
from collections import Counter
import pytest
from sintetico.distributed import (DLQ_TOPIC, RESULTS_TOPIC, TASKS_TOPIC, AgentWorker, BusError, DistributedAIGateway, InMemoryBus,
                                   ResultEnvelope, RetryableError, TaskEnvelope)
from sintetico.orchestration import TaskStatus


class SimulatedCrash(BaseException):
    """Like a SIGKILL: not an Exception, so nothing in the worker may swallow it."""


def env(task_id="t1", attempt=1, reply_to="gw", deadline=None, payload=None, task_type="echo"):
    return TaskEnvelope(task_id, task_type, payload if payload is not None else {"n": 1}, reply_to, attempt, 0.0, deadline)


def put(bus, e, topic=TASKS_TOPIC): bus.publish(topic, e.to_bytes(), key=e.task_id)
def results(bus): return [ResultEnvelope.from_bytes(m.value) for m in bus.log(RESULTS_TOPIC)]
def dlq(bus): return [json.loads(m.value) for m in bus.log(DLQ_TOPIC)]
def drain(w, n=50):
    out = []
    for _ in range(n):
        r = w.run_once(0.02)
        if r is None: break
        out.append(r)
    return out


def worker(bus, handler=lambda t: {"ok": t.payload}, **kw):
    kw.setdefault("sleep", lambda s: None)
    return AgentWorker(bus, handler, **kw)


# ============================================================================================== the bus itself
def test_log_order_offsets_keys_and_independent_groups():
    bus = InMemoryBus()
    for i in range(3): bus.publish("t", f"m{i}".encode(), key=f"k{i}")
    a, b = bus.consumer(["t"], "ga"), bus.consumer(["t"], "gb")
    got_a = [a.poll(0.01) for _ in range(4)]
    assert [m.value for m in got_a[:3]] == [b"m0", b"m1", b"m2"] and [m.offset for m in got_a[:3]] == [0, 1, 2] and got_a[3] is None
    assert got_a[1].key == "k1" and [b.poll(0.01).value for _ in range(3)] == [b"m0", b"m1", b"m2"]        # each group sees everything
    late = bus.consumer(["t"], "gc", from_beginning=False)
    assert late.poll(0.01) is None
    bus.publish("t", b"m3"); assert late.poll(0.5).value == b"m3"


def test_uncommitted_messages_return_to_the_group_when_a_consumer_dies_committed_ones_do_not():
    bus = InMemoryBus()
    for i in range(4): bus.publish("t", f"m{i}".encode())
    a = bus.consumer(["t"], "g")
    m0, _m1, _m2 = a.poll(0.01), a.poll(0.01), a.poll(0.01)
    a.commit(m0)
    assert bus.lag("g", "t") == 3                                       # m1, m2 in flight + m3 undelivered
    a.close()                                                            # "crash": m1, m2 were never committed
    b = bus.consumer(["t"], "g")
    assert [b.poll(0.01).value for _ in range(3)] == [b"m1", b"m2", b"m3"] and b.poll(0.01) is None


def test_competing_consumers_never_get_the_same_message_and_nack_redelivers():
    bus = InMemoryBus()
    for i in range(6): bus.publish("t", str(i).encode())
    a, b = bus.consumer(["t"], "g"), bus.consumer(["t"], "g")
    seen = [m.value for m in (a.poll(0.01), b.poll(0.01), a.poll(0.01), b.poll(0.01), a.poll(0.01), b.poll(0.01))]
    assert sorted(seen) == [str(i).encode() for i in range(6)]
    c = bus.consumer(["t2"], "g2"); bus.publish("t2", b"x")
    m = c.poll(0.01); c.nack(m); again = c.poll(0.01)
    assert again.offset == m.offset and bus.lag("g2", "t2") == 1
    c.commit(again); assert bus.lag("g2", "t2") == 0
    c.commit(again)                                                      # committing twice is harmless


def test_poll_blocks_until_a_message_arrives_and_times_out_otherwise():
    bus = InMemoryBus(); c = bus.consumer(["t"], "g")
    t0 = time.monotonic(); assert c.poll(0.1) is None and 0.08 <= time.monotonic() - t0 < 0.5
    threading.Timer(0.1, lambda: bus.publish("t", b"late")).start()
    t0 = time.monotonic(); assert c.poll(2.0).value == b"late" and time.monotonic() - t0 < 1.0
    c.close(); assert c.poll(0.01) is None                               # a closed consumer returns nothing


def test_injected_publish_failures_raise_and_leave_no_trace():
    bus = InMemoryBus(); bus.fail_publish = lambda topic: topic == "bad"
    with pytest.raises(BusError): bus.publish("bad", b"x")
    assert bus.log("bad") == []; bus.publish("good", b"x"); assert len(bus.log("good")) == 1


# ================================================================================================== envelopes
def test_envelopes_round_trip_and_validate_strictly():
    e = env("abc", attempt=2, deadline=12.5, payload={"q": "héllo ñ"})
    assert TaskEnvelope.from_bytes(e.to_bytes()) == e
    r = ResultEnvelope("abc", "completed", "gw", {"a": 1}, None, "w1", 2, 9.0)
    assert ResultEnvelope.from_bytes(r.to_bytes()) == r
    good = json.loads(e.to_bytes())
    def mutate(**kw): return json.dumps({**good, **kw}).encode()
    for bad in (b"not json", b"\xff\xfe", b"[1, 2]", b'"x"', json.dumps({k: v for k, v in good.items() if k != "task_id"}).encode(),
                json.dumps({k: v for k, v in good.items() if k != "reply_to"}).encode(), mutate(task_id=""), mutate(task_id=5), mutate(payload=[1]),
                mutate(attempt=0), mutate(attempt="2"), mutate(attempt=True), mutate(deadline="soon"), mutate(task_type=None)):
        with pytest.raises(ValueError): TaskEnvelope.from_bytes(bad)
    with pytest.raises(ValueError): env(payload={"x": object()}).to_bytes()
    for bad in (b"{}", b'{"status": "weird", "task_id": "a", "reply_to": "g"}', b"nope"):
        with pytest.raises(ValueError): ResultEnvelope.from_bytes(bad)
    assert TaskEnvelope.from_bytes(json.dumps({k: v for k, v in good.items() if k not in ("attempt", "created_at", "deadline")}).encode()).attempt == 1


# ================================================================================================== the worker
def test_happy_path_publishes_the_result_then_commits():
    bus = InMemoryBus(); w = worker(bus, worker_id="w1")
    put(bus, env("t1", payload={"q": 1}))
    assert w.run_once(0.05) == "completed"
    (r,) = results(bus)
    assert (r.task_id, r.status, r.result, r.error, r.worker_id, r.attempts, r.reply_to) == ("t1", "completed", {"ok": {"q": 1}}, None, "w1", 1, "gw") and r.finished_at > 0
    assert bus.lag("agent-workers", TASKS_TOPIC) == 0 and dlq(bus) == [] and w.metrics["completed"] == 1
    assert bus.log(RESULTS_TOPIC)[0].key == "gw" and w.run_once(0.01) is None


def test_non_retryable_failure_becomes_a_failed_result_and_a_dead_letter():
    bus, calls = InMemoryBus(), []
    def boom(t): calls.append(1); raise ValueError("bad input")
    w = worker(bus, boom); put(bus, env("t1"))
    assert w.run_once(0.05) == "failed" and len(calls) == 1
    (r,), (d,) = results(bus), dlq(bus)
    assert r.status == "failed" and r.error == "ValueError: bad input" and r.attempts == 1
    assert d["reason"] == "failed" and d["error"] == "ValueError: bad input" and d["task"]["task_id"] == "t1" and d["source"]["topic"] == TASKS_TOPIC
    assert bus.lag("agent-workers", TASKS_TOPIC) == 0


def test_retryable_errors_back_off_exponentially_then_give_up_into_the_dlq():
    bus, calls, sleeps = InMemoryBus(), [], []
    def flaky(t): calls.append(t.attempt); raise RetryableError("503")
    w = worker(bus, flaky, max_attempts=3, backoff_base=0.5, sleep=sleeps.append); put(bus, env("t1"))
    assert drain(w) == ["retry", "retry", "failed"] and calls == [1, 2, 3] and sleeps == [0.5, 1.0]
    (r,), (d,) = results(bus), dlq(bus)
    assert r.status == "failed" and r.attempts == 3 and "RetryableError: 503" == r.error and d["reason"] == "failed"
    assert [TaskEnvelope.from_bytes(m.value).attempt for m in bus.log(TASKS_TOPIC)] == [1, 2, 3] and bus.lag("agent-workers", TASKS_TOPIC) == 0


def test_retry_that_eventually_succeeds_leaves_no_dead_letter_and_backoff_is_capped():
    bus = InMemoryBus(); n = {"c": 0}
    def second_time(t):
        n["c"] += 1
        if n["c"] < 2: raise RetryableError("once")
        return "fine"
    w = worker(bus, second_time); put(bus, env("t1"))
    assert drain(w) == ["retry", "completed"] and results(bus)[0].attempts == 2 and results(bus)[0].result == "fine" and dlq(bus) == []
    bus2, sleeps = InMemoryBus(), []
    w2 = worker(bus2, lambda t: (_ for _ in ()).throw(RetryableError("x")), max_attempts=6, backoff_base=1.0, backoff_max=3.0, sleep=sleeps.append)
    put(bus2, env("t2")); drain(w2)
    assert sleeps == [1.0, 2.0, 3.0, 3.0, 3.0]


def test_a_stuck_handler_times_out_is_retried_and_does_not_block_the_next_task():
    bus, release = InMemoryBus(), threading.Event()
    def handler(t):
        if t.task_id == "slow": release.wait(5); return "late"
        return "quick"
    w = worker(bus, handler, task_timeout=0.1, max_attempts=2); put(bus, env("slow")); put(bus, env("fast"))
    outcomes = drain(w)
    release.set()
    assert outcomes == ["retry", "completed", "timeout"]                      # slow(1) times out -> requeued behind fast -> fast done -> slow(2) gives up
    by = {r.task_id: r for r in results(bus)}
    assert by["fast"].status == "completed" and by["fast"].result == "quick" and by["slow"].status == "timeout" and by["slow"].attempts == 2
    assert "exceeded 0.1s" in by["slow"].error and dlq(bus)[0]["reason"] == "timeout"


def test_unserialisable_results_are_failures_not_crashes():
    bus = InMemoryBus(); w = worker(bus, lambda t: object()); put(bus, env("t1"))
    assert w.run_once(0.05) == "failed" and "not JSON-serialisable" in results(bus)[0].error


def test_poison_messages_go_to_the_dlq_with_the_original_bytes_and_do_not_stop_the_worker():
    bus = InMemoryBus(); w = worker(bus)
    bus.publish(TASKS_TOPIC, b"\x00garbage"); bus.publish(TASKS_TOPIC, json.dumps({"task_id": "x"}).encode()); put(bus, env("good"))
    assert drain(w) == ["dlq", "dlq", "completed"]
    d = dlq(bus)
    assert [x["reason"] for x in d] == ["undecodable"] * 2 and base64.b64decode(d[0]["raw_b64"]) == b"\x00garbage" and d[0]["task"] is None
    assert d[0]["source"] == {"topic": TASKS_TOPIC, "offset": 0, "partition": 0} and "missing" in d[1]["error"]
    assert results(bus)[0].task_id == "good" and bus.lag("agent-workers", TASKS_TOPIC) == 0 and w.metrics["poison"] == 2


def test_duplicate_deliveries_run_the_handler_once():
    bus, calls = InMemoryBus(), []
    w = worker(bus, lambda t: calls.append(t.task_id) or "r"); put(bus, env("same")); put(bus, env("same")); put(bus, env("other"))
    assert drain(w) == ["completed", "duplicate", "completed"] and calls == ["same", "other"] and len(results(bus)) == 2 and w.metrics["duplicates"] == 1
    small = worker(InMemoryBus(), dedupe_size=2)
    for i in range(3): put(small.bus, env(f"t{i}"))
    drain(small); put(small.bus, env("t0")); assert drain(small) == ["completed"]          # t0 was forgotten after 3 newer tasks (bounded memory)


def test_tasks_whose_deadline_has_passed_are_not_executed():
    bus, calls, now = InMemoryBus(), [], [100.0]
    w = worker(bus, lambda t: calls.append(1) or "r", clock=lambda: now[0])
    put(bus, env("late", deadline=99.9)); put(bus, env("edge", deadline=100.0)); put(bus, env("open", deadline=None))
    assert drain(w) == ["timeout", "completed", "completed"] and len(calls) == 2
    r = {x.task_id: x for x in results(bus)}
    assert r["late"].status == "timeout" and "before the task started" in r["late"].error and dlq(bus) == []


def test_crash_before_publishing_the_result_means_the_task_runs_again_elsewhere():
    bus, calls = InMemoryBus(), []
    def crash(stage):
        if stage == "after_handler": raise SimulatedCrash()
    w1 = worker(bus, lambda t: calls.append("w1") or "r", fault_hook=crash); put(bus, env("t1"))
    with pytest.raises(SimulatedCrash): w1.run_once(0.05)
    assert results(bus) == [] and bus.lag("agent-workers", TASKS_TOPIC) == 1
    w1.close()                                                                            # the process is gone: its in-flight work is released
    w2 = worker(bus, lambda t: calls.append("w2") or "r")
    assert drain(w2) == ["completed"] and calls == ["w1", "w2"] and len(results(bus)) == 1       # at-least-once: executed twice, answered once


def test_crash_after_publishing_but_before_commit_yields_a_duplicate_result_the_gateway_ignores():
    bus = InMemoryBus()
    def crash(stage):
        if stage == "after_publish": raise SimulatedCrash()
    with DistributedAIGateway(bus) as gw:
        w1 = worker(bus, lambda t: "answer-1", fault_hook=crash)
        tid = gw.submit_task("echo", {"q": 1})
        with pytest.raises(SimulatedCrash): w1.run_once(1.0)
        w1.close()
        w2 = worker(bus, lambda t: "answer-2")
        assert drain(w2) == ["completed"]                                                 # redelivered, processed again
        assert len(results(bus)) == 2 and {r.task_id for r in results(bus)} == {tid}
        task = gw.wait_for_result(tid, 2)
        time.sleep(0.3)
        assert task.status is TaskStatus.COMPLETED and gw.get_task(tid).result == "answer-1"        # first result wins; the duplicate is ignored


def test_failed_result_publication_is_not_lost():
    bus, calls, failing = InMemoryBus(), [], [True]
    bus.fail_publish = lambda topic: topic == RESULTS_TOPIC and failing[0]
    w = worker(bus, lambda t: calls.append(1) or "r"); put(bus, env("t1"))
    with pytest.raises(BusError): w.run_once(0.05)
    assert bus.lag("agent-workers", TASKS_TOPIC) == 1 and results(bus) == []
    failing[0] = False
    assert w.run_once(0.05) == "completed" and len(calls) == 2 and len(results(bus)) == 1 and bus.lag("agent-workers", TASKS_TOPIC) == 0
    bus2, fail_dlq = InMemoryBus(), [True]
    bus2.fail_publish = lambda topic: topic == DLQ_TOPIC and fail_dlq[0]
    w2 = worker(bus2); bus2.publish(TASKS_TOPIC, b"garbage")
    with pytest.raises(BusError): w2.run_once(0.05)
    fail_dlq[0] = False; assert w2.run_once(0.05) == "dlq" and len(dlq(bus2)) == 1


def test_run_forever_survives_bus_errors_and_stops_gracefully():
    bus, failing = InMemoryBus(), [True]
    bus.fail_publish = lambda topic: topic == RESULTS_TOPIC and failing[0]
    w = worker(bus); put(bus, env("t1"))
    th = w.start(poll_timeout=0.02, error_backoff=0.02)
    time.sleep(0.2); assert w.metrics["bus_errors"] >= 1 and th.is_alive()
    failing[0] = False; deadline = time.time() + 3
    while not results(bus) and time.time() < deadline: time.sleep(0.02)
    w.stop(); th.join(3)
    assert not th.is_alive() and len(results(bus)) == 1 and bus.lag("agent-workers", TASKS_TOPIC) == 0


def test_three_competing_workers_process_every_task_exactly_once():
    bus, seen, lock = InMemoryBus(), [], threading.Lock()
    def handler(t):
        with lock: seen.append(t.task_id)
        time.sleep(0.002); return t.task_id
    ws = [worker(bus, handler) for _ in range(3)]
    for i in range(60): put(bus, env(f"t{i}"))
    ths = [w.start(poll_timeout=0.02) for w in ws]
    deadline = time.time() + 10
    while len(results(bus)) < 60 and time.time() < deadline: time.sleep(0.02)
    [w.stop() for w in ws]; [t.join(3) for t in ths]
    assert sorted(seen) == sorted(f"t{i}" for i in range(60)) and Counter(r.task_id for r in results(bus)).most_common(1)[0][1] == 1
    assert bus.lag("agent-workers", TASKS_TOPIC) == 0 and sum(w.metrics["completed"] for w in ws) == 60


def test_worker_parameter_validation():
    for bad in (dict(max_attempts=0), dict(task_timeout=0)):
        with pytest.raises(ValueError): worker(InMemoryBus(), **bad)
