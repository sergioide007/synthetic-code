import json, threading, time
import pytest
pytest.importorskip("pydantic")
from sintetico.distributed import (RESULTS_TOPIC, TASKS_TOPIC, AgentWorker, BusError, DistributedAIGateway, GatewayOverloaded, InMemoryBus,
                                   ResultEnvelope, TaskTimeout, react_handler)
from sintetico.orchestration import TaskStatus
from sintetico.providers import LLMProvider, LLMResponse
from sintetico.reasoning_resilient import ReActAgentResilient


def start_worker(bus, handler, **kw):
    kw.setdefault("sleep", lambda s: None)
    w = AgentWorker(bus, handler, **kw)
    return w, w.start(poll_timeout=0.02)


def stop(*pairs):
    for w, th in pairs: w.stop()
    for w, th in pairs: th.join(3)


def test_submit_returns_immediately_and_the_result_arrives_through_the_bus():
    bus = InMemoryBus()
    w = start_worker(bus, lambda t: {"echo": t.payload, "type": t.task_type})
    with DistributedAIGateway(bus) as gw:
        t0 = time.monotonic()
        tid = gw.submit_task("summarise", {"doc": 7})
        assert time.monotonic() - t0 < 0.5 and isinstance(tid, str) and len(tid) == 32
        task = gw.wait_for_result(tid, 5)
        assert task.status is TaskStatus.COMPLETED and json.loads(task.result) == {"echo": {"doc": 7}, "type": "summarise"} and task.error is None and task.payload == {"doc": 7}
    stop(w)


def test_failed_and_timed_out_tasks_surface_their_status_and_error():
    bus = InMemoryBus()
    def handler(t):
        if t.payload.get("kind") == "bug": raise KeyError("missing")
        time.sleep(0.5)
    w = start_worker(bus, handler, task_timeout=0.05, max_attempts=1)
    with DistributedAIGateway(bus) as gw:
        bug, slow = gw.submit_task("x", {"kind": "bug"}), gw.submit_task("x", {"kind": "slow"})
        a, b = gw.wait_for_result(bug, 5), gw.wait_for_result(slow, 5)
        assert a.status is TaskStatus.FAILED and "KeyError" in a.error and a.result is None
        assert b.status is TaskStatus.TIMEOUT and "exceeded 0.05s" in b.error
    stop(w)


def test_waiting_gives_up_without_cancelling_and_a_late_result_still_lands():
    bus = InMemoryBus()
    with DistributedAIGateway(bus) as gw:
        tid = gw.submit_task("x", {})
        with pytest.raises(TaskTimeout): gw.wait_for_result(tid, 0.05)
        assert gw.get_task(tid).status is TaskStatus.PENDING and gw.pending() == 1
        w = start_worker(bus, lambda t: "late but fine")
        assert gw.wait_for_result(tid, 5).result == "late but fine" and gw.pending() == 0
        stop(w)
        with pytest.raises(KeyError): gw.wait_for_result("nope", 0.01)
        with pytest.raises(KeyError): gw.get_task("nope")


def test_misuse_is_rejected_before_anything_is_sent():
    bus = InMemoryBus(); gw = DistributedAIGateway(bus)
    with pytest.raises(RuntimeError, match="start"): gw.submit_task("x", {})
    with gw:
        with pytest.raises(ValueError): gw.submit_task("x", {"bad": object()})
        assert gw.pending() == 0 and bus.log(TASKS_TOPIC) == []
        bus.fail_publish = lambda topic: topic == TASKS_TOPIC
        with pytest.raises(BusError): gw.submit_task("x", {})
        assert gw.pending() == 0                                                         # the half-registered task was rolled back


def test_backpressure_limits_tasks_in_flight():
    bus = InMemoryBus()
    with DistributedAIGateway(bus, max_pending=2) as gw:
        gw.submit_task("x", {}); gw.submit_task("x", {})
        with pytest.raises(GatewayOverloaded): gw.submit_task("x", {})
        w = start_worker(bus, lambda t: "ok")
        deadline = time.time() + 5
        while gw.pending() and time.time() < deadline: time.sleep(0.02)
        assert gw.submit_task("x", {})                                                    # capacity is back once results arrive
        stop(w)


def test_foreign_malformed_and_duplicate_results_are_ignored():
    bus = InMemoryBus()
    with DistributedAIGateway(bus, gateway_id="me") as gw:
        tid = gw.submit_task("x", {})
        bus.publish(RESULTS_TOPIC, ResultEnvelope(tid, "completed", "someone-else", "WRONG").to_bytes())      # another gateway's reply_to
        bus.publish(RESULTS_TOPIC, b"not even json")
        bus.publish(RESULTS_TOPIC, ResultEnvelope("unknown-task", "completed", "me", "x").to_bytes())
        time.sleep(0.2); assert gw.get_task(tid).status is TaskStatus.PENDING
        bus.publish(RESULTS_TOPIC, ResultEnvelope(tid, "completed", "me", "first").to_bytes())
        bus.publish(RESULTS_TOPIC, ResultEnvelope(tid, "failed", "me", None, "second").to_bytes())
        assert gw.wait_for_result(tid, 2).result == "first"
        time.sleep(0.2); t = gw.get_task(tid); assert (t.status, t.result, t.error) == (TaskStatus.COMPLETED, "first", None)


def test_two_gateways_sharing_the_workers_each_get_only_their_own_results():
    bus = InMemoryBus()
    w = start_worker(bus, lambda t: t.payload["who"])
    with DistributedAIGateway(bus, gateway_id="A") as a, DistributedAIGateway(bus, gateway_id="B") as b:
        ids_a = [a.submit_task("x", {"who": "A"}) for _ in range(5)]; ids_b = [b.submit_task("x", {"who": "B"}) for _ in range(5)]
        assert [a.wait_for_result(i, 5).result for i in ids_a] == ["A"] * 5 and [b.wait_for_result(i, 5).result for i in ids_b] == ["B"] * 5
        with pytest.raises(KeyError): a.get_task(ids_b[0])
    stop(w)


def test_finished_tasks_are_trimmed_but_pending_ones_never_are():
    bus = InMemoryBus()
    w = start_worker(bus, lambda t: "ok")
    with DistributedAIGateway(bus, max_retained=5) as gw:
        for _ in range(20):
            gw.submit_task("x", {})
        deadline = time.time() + 10
        while time.time() < deadline:
            with gw._lock:
                if all(t.status is TaskStatus.COMPLETED for t in gw._tasks.values()) and len(gw._tasks) <= 5: break
            time.sleep(0.02)
        assert len(gw._tasks) == 5 and len(gw._events) == 5
    stop(w)


def test_a_deadline_makes_workers_skip_work_nobody_is_waiting_for():
    bus, calls = InMemoryBus(), []
    with DistributedAIGateway(bus) as gw:
        tid = gw.submit_task("x", {}, timeout=0.05)
        time.sleep(0.3)                                                                    # the caller's time budget has passed; no worker was alive
        w = start_worker(bus, lambda t: calls.append(1))
        task = gw.wait_for_result(tid, 5)
        assert task.status is TaskStatus.TIMEOUT and "before the task started" in task.error and calls == []
        stop(w)


# ------------------------------------------------------------------------------ the book's scenario: ReAct on a worker
class Scripted(LLMProvider):
    def __init__(self, *answers): self.answers, self.n, self.lock = list(answers), 0, threading.Lock()
    def complete(self, request):
        with self.lock:
            self.n += 1; a = self.answers.pop(0)
        return LLMResponse(a if isinstance(a, str) else json.dumps(a), {"input_tokens": 3, "output_tokens": 2}, "s")
    @property
    def cost_per_million_tokens(self): return 0.0


def step(thought, action=None, **inp): return {"thought": thought, "action": action, "action_input": inp}


def test_a_react_agent_runs_inside_a_worker_end_to_end():
    prov = Scripted(step("look it up", "lookup", key="k"), step("done", None, result="the value is 42"))
    bus = InMemoryBus()
    w = start_worker(bus, react_handler(lambda: ReActAgentResilient(prov, {"lookup": lambda key: "42"})))
    with DistributedAIGateway(bus) as gw:
        task = gw.wait_for_result(gw.submit_task("react", {"query": "what is k?"}), 5)
        out = json.loads(task.result)
        assert task.status is TaskStatus.COMPLETED and out["answer"] == "the value is 42" and out["steps"] == 2 and out["usage"] == {"input_tokens": 6, "output_tokens": 4}
    stop(w)


def test_react_handler_retries_model_flakes_but_not_loops_or_bad_payloads():
    prov = Scripted("junk", "junk", "junk", step("recovered", None, result="ok"))          # first task: 3 invalid answers = invalid_output; second attempt succeeds
    bus = InMemoryBus()
    w = start_worker(bus, react_handler(lambda: ReActAgentResilient(prov, {}, max_attempts=3)), max_attempts=3)
    with DistributedAIGateway(bus) as gw:
        ok = gw.wait_for_result(gw.submit_task("react", {"query": "q"}), 5)
        assert ok.status is TaskStatus.COMPLETED and json.loads(ok.result)["answer"] == "ok" and w[0].metrics["retried"] == 1
        bad = gw.wait_for_result(gw.submit_task("react", {"nothing": 1}), 5)
        assert bad.status is TaskStatus.FAILED and "payload.query" in bad.error and w[0].metrics["retried"] == 1          # a bad payload is not retried
    stop(w)
    loop_prov = Scripted(*[step(f"t{i}" * 4 + "x" * i, "same", a=1) for i in range(5)])
    loop_prov.answers = [step(w_, "same", a=1) for w_ in ("alpha one", "beta two", "gamma three", "delta four")]
    bus2 = InMemoryBus(); w2 = start_worker(bus2, react_handler(lambda: ReActAgentResilient(loop_prov, {"same": lambda a: "const"})))
    with DistributedAIGateway(bus2) as gw2:
        t = gw2.wait_for_result(gw2.submit_task("react", {"query": "q"}), 5)
        assert t.status is TaskStatus.FAILED and "repeated 3 times" in t.error and w2[0].metrics["retried"] == 0         # a detected loop is not retried
    stop(w2)
