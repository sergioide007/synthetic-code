"""Agents as workers: submit a task, get a task_id back immediately, collect the result later.

In-memory by default (no broker needed). Against a real Kafka:  python examples/11_distributed_agents.py --kafka localhost:9092
    (start one with: docker compose -f docker/kafka-compose.yml up -d     -- needs: pip install "codigo-sintetico[kafka]")
"""
import json
import sys

from sintetico import AgentWorker, DistributedAIGateway, InMemoryBus, KafkaBus, RetryableError
from sintetico.orchestration import TaskStatus

bus = KafkaBus(sys.argv[sys.argv.index("--kafka") + 1]) if "--kafka" in sys.argv else InMemoryBus()
attempts = {"flaky": 0}


def handler(task):
    kind = task.payload["kind"]
    if kind == "ok":
        return {"summary": task.payload["text"][:20].upper()}
    if kind == "flaky":                                           # fails twice with a transient error, then works
        attempts["flaky"] += 1
        if attempts["flaky"] < 3:
            raise RetryableError("upstream 503")
        return {"summary": "recovered"}
    raise ValueError("this input can never work")                 # a bug: retrying would not help


workers = [AgentWorker(bus, handler, max_attempts=3, backoff_base=0.01) for _ in range(2)]
threads = [w.start(poll_timeout=0.1) for w in workers]
with DistributedAIGateway(bus) as gw:
    ids = {k: gw.submit_task("demo", {"kind": k, "text": "a long document about quarterly revenue"}) for k in ("ok", "flaky", "bad")}
    print("submitted (returned immediately):", {k: v[:8] for k, v in ids.items()})
    done = {k: gw.wait_for_result(v, 60) for k, v in ids.items()}
for k, t in done.items():
    print(f"  {k:6} -> {t.status.name:10} result={t.result}  error={t.error}")
for w in workers:
    w.stop()
for t in threads:
    t.join(5)
print("worker metrics:", {w.worker_id: dict(w.metrics) for w in workers})
assert done["ok"].status is TaskStatus.COMPLETED and done["flaky"].status is TaskStatus.COMPLETED and json.loads(done["flaky"].result) == {"summary": "recovered"}
assert done["bad"].status is TaskStatus.FAILED and "never work" in done["bad"].error
print("OK")
