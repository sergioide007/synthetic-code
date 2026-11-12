"""Event-driven agent execution over a message bus (book Chapter 9, "asynchronous architecture").

    caller --submit_task--> [agent.tasks] --> AgentWorker --> handler (e.g. a ReAct agent) --> [agent.results] --> gateway.wait_for_result
                                                  |--> [agent.tasks.dlq]  (poison messages, exhausted retries)

* :class:`MessageBus` is a tiny interface with two implementations: :class:`InMemoryBus` (tests, demos, single process)
  and :class:`KafkaBus` (``confluent-kafka``; idempotent producer, manual commits).
* :class:`AgentWorker` gives **at-least-once** processing: the offset is committed only AFTER the result has been
  published, so a crash re-delivers the task. Handlers should therefore be idempotent; the worker also de-duplicates by
  ``task_id`` within its own lifetime and the gateway ignores duplicate results. It is NOT exactly-once.
* :class:`DistributedAIGateway` returns a ``task_id`` immediately and lets callers wait for the result.

Known limits (documented, tested where possible): a Python thread cannot be killed, so a handler that exceeds
``task_timeout`` is abandoned, not stopped; retries back off by sleeping inside the worker (keep ``backoff_max`` well below
Kafka's ``max.poll.interval.ms``); the KafkaBus was verified against an API-faithful simulator, see ``docs/VERIFICATION.md``.
"""

from __future__ import annotations

import base64
import concurrent.futures as cf
import itertools
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Protocol, Tuple

from .orchestration import AgentTask, TaskStatus

__all__ = ["TASKS_TOPIC", "RESULTS_TOPIC", "DLQ_TOPIC", "BusError", "RetryableError", "TaskTimeout", "GatewayOverloaded", "BusMessage",
           "TaskEnvelope", "ResultEnvelope", "MessageBus", "BusConsumer", "InMemoryBus", "KafkaBus", "AgentWorker",
           "DistributedAIGateway", "react_handler"]

log = logging.getLogger(__name__)
TASKS_TOPIC, RESULTS_TOPIC, DLQ_TOPIC = "agent.tasks", "agent.results", "agent.tasks.dlq"
COMPLETED, FAILED, TIMEOUT = "completed", "failed", "timeout"


class BusError(RuntimeError):
    """The bus could not publish/consume (broker down, delivery failed...)."""


class RetryableError(Exception):
    """Raise from a handler to ask for another attempt (transient failure)."""


class TaskTimeout(TimeoutError):
    """The caller stopped waiting. The task may still complete later."""


class GatewayOverloaded(RuntimeError):
    pass


@dataclass(frozen=True)
class BusMessage:
    topic: str
    key: Optional[str]
    value: bytes
    offset: int
    partition: int = 0
    raw: Any = field(default=None, repr=False, compare=False)     # the broker client's own message object (for commits)


# ------------------------------------------------------------------------------------------------ envelopes
def _require(d: Dict[str, Any], name: str, types: tuple, allow_none: bool = False) -> Any:
    v = d.get(name)
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, types):
        raise ValueError(f"field '{name}' missing or of the wrong type")
    return v


@dataclass
class TaskEnvelope:
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    reply_to: str
    attempt: int = 1
    created_at: float = 0.0
    deadline: Optional[float] = None       # epoch seconds; workers skip tasks whose caller already gave up

    def to_bytes(self) -> bytes:
        try:
            return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")
        except TypeError as exc:
            raise ValueError(f"payload is not JSON-serialisable: {exc}") from exc

    @classmethod
    def from_bytes(cls, data: bytes) -> "TaskEnvelope":
        try:
            d = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        if not isinstance(d, dict):
            raise ValueError("envelope must be a JSON object")
        task_id = _require(d, "task_id", (str,))
        if not task_id:
            raise ValueError("task_id must not be empty")
        attempt = _require(d, "attempt", (int,)) if "attempt" in d else 1
        if attempt < 1:
            raise ValueError("attempt must be >= 1")
        return cls(task_id, _require(d, "task_type", (str,)), _require(d, "payload", (dict,)), _require(d, "reply_to", (str,)), attempt,
                   float(_require(d, "created_at", (int, float))) if "created_at" in d else 0.0, _require(d, "deadline", (int, float), allow_none=True))


@dataclass
class ResultEnvelope:
    task_id: str
    status: str                             # completed | failed | timeout
    reply_to: str
    result: Any = None
    error: Optional[str] = None
    worker_id: str = ""
    attempts: int = 1
    finished_at: float = 0.0

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":"), default=str).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ResultEnvelope":
        try:
            d = json.loads(data)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"not valid JSON: {exc}") from exc
        if not isinstance(d, dict) or _require(d, "status", (str,)) not in (COMPLETED, FAILED, TIMEOUT):
            raise ValueError("invalid result envelope")
        return cls(_require(d, "task_id", (str,)), d["status"], _require(d, "reply_to", (str,)), d.get("result"), d.get("error"),
                   str(d.get("worker_id", "")), int(d.get("attempts", 1)), float(d.get("finished_at", 0.0)))


# ------------------------------------------------------------------------------------------------- the bus
class BusConsumer(Protocol):
    def poll(self, timeout: float) -> Optional[BusMessage]: ...
    def commit(self, message: BusMessage) -> None: ...
    def nack(self, message: BusMessage) -> None: ...          # make the message deliverable again to this consumer
    def wait_ready(self, timeout: float) -> bool: ...         # partitions assigned / consuming
    def close(self) -> None: ...


class MessageBus(Protocol):
    def publish(self, topic: str, value: bytes, key: Optional[str] = None) -> None: ...   # durable when it returns, else raises BusError
    def consumer(self, topics: List[str], group_id: str, *, from_beginning: bool = True) -> BusConsumer: ...
    def ensure_topics(self, topics: List[str]) -> None: ...
    def close(self) -> None: ...


class _GroupState:
    def __init__(self, start: int) -> None:
        self.next_offset = start
        self.redeliver: Deque[int] = deque()
        self.inflight: Dict[int, int] = {}                  # offset -> consumer id


class InMemoryBus:
    """Single-process bus with Kafka-like consumer groups: competing consumers inside a group, independent groups,
    manual commits, and re-delivery of un-committed messages when a consumer closes or crashes. One partition."""

    def __init__(self) -> None:
        self._logs: Dict[str, List[BusMessage]] = defaultdict(list)
        self._groups: Dict[Tuple[str, str], _GroupState] = {}
        self._cond = threading.Condition()
        self._ids = itertools.count(1)
        self.fail_publish: Optional[Callable[[str], bool]] = None   # fault injection for tests

    def ensure_topics(self, topics: List[str]) -> None:
        with self._cond:
            for t in topics:
                self._logs[t]

    def publish(self, topic: str, value: bytes, key: Optional[str] = None) -> None:
        if self.fail_publish and self.fail_publish(topic):
            raise BusError(f"injected publish failure on {topic}")
        with self._cond:
            log = self._logs[topic]
            log.append(BusMessage(topic, key, bytes(value), len(log)))
            self._cond.notify_all()

    def consumer(self, topics: List[str], group_id: str, *, from_beginning: bool = True) -> "_InMemoryConsumer":
        with self._cond:
            for t in topics:
                self._groups.setdefault((group_id, t), _GroupState(0 if from_beginning else len(self._logs[t])))
        return _InMemoryConsumer(self, list(topics), group_id, next(self._ids))

    def log(self, topic: str) -> List[BusMessage]:
        with self._cond:
            return list(self._logs[topic])

    def lag(self, group_id: str, topic: str) -> int:
        """Messages not yet committed by the group (undelivered + in flight + waiting for re-delivery)."""
        with self._cond:
            st = self._groups[(group_id, topic)]
            return (len(self._logs[topic]) - st.next_offset) + len(st.inflight) + len(st.redeliver)

    def close(self) -> None:
        with self._cond:
            self._cond.notify_all()


class _InMemoryConsumer:
    def __init__(self, bus: InMemoryBus, topics: List[str], group: str, cid: int) -> None:
        self.bus, self.topics, self.group, self.cid, self._closed = bus, topics, group, cid, False

    def poll(self, timeout: float) -> Optional[BusMessage]:
        deadline = time.monotonic() + timeout
        bus = self.bus
        while True:
            with bus._cond:
                if self._closed:
                    return None
                for t in self.topics:
                    st, log = bus._groups[(self.group, t)], bus._logs[t]
                    if st.redeliver:
                        off = st.redeliver.popleft()
                    elif st.next_offset < len(log):
                        off, st.next_offset = st.next_offset, st.next_offset + 1
                    else:
                        continue
                    st.inflight[off] = self.cid
                    return log[off]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                bus._cond.wait(min(remaining, 0.05))

    def commit(self, message: BusMessage) -> None:
        with self.bus._cond:
            self.bus._groups[(self.group, message.topic)].inflight.pop(message.offset, None)

    def nack(self, message: BusMessage) -> None:
        with self.bus._cond:
            st = self.bus._groups[(self.group, message.topic)]
            if st.inflight.pop(message.offset, None) is not None:
                st.redeliver.appendleft(message.offset)
                self.bus._cond.notify_all()

    def wait_ready(self, timeout: float) -> bool:
        return True

    def close(self) -> None:
        with self.bus._cond:
            self._closed = True
            for t in self.topics:
                st = self.bus._groups[(self.group, t)]
                mine = sorted(o for o, c in st.inflight.items() if c == self.cid)
                for o in mine:
                    del st.inflight[o]
                for o in reversed(mine):                      # un-committed work goes back to the group, oldest first
                    st.redeliver.appendleft(o)
            self.bus._cond.notify_all()


class KafkaBus:
    """Kafka through ``confluent-kafka``: idempotent producer with ``acks=all``, consumers with manual commits.

    Client classes are injectable (``producer_factory`` etc.) so the logic is testable without a broker. It has been run
    against ``mockafka`` (an API-faithful in-memory simulator), NOT against a real broker in this repository's own test run;
    ``tests/integration`` contains a real-broker test that runs when ``SINTETICO_TEST_KAFKA`` is set (the CI job provides one)."""

    def __init__(self, bootstrap_servers: str, *, client_config: Optional[Dict[str, Any]] = None, flush_timeout: float = 10.0,
                 producer_factory: Optional[Callable[[Dict[str, Any]], Any]] = None, consumer_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
                 admin_factory: Optional[Callable[[Dict[str, Any]], Any]] = None, partitions: int = 1, replication_factor: int = 1) -> None:
        self.base = {"bootstrap.servers": bootstrap_servers, **(client_config or {})}
        self.flush_timeout, self.partitions, self.replication = flush_timeout, partitions, replication_factor
        ck: Any
        try:
            import confluent_kafka as ck
        except ImportError:
            ck = None
        if ck is None and not (producer_factory and consumer_factory and admin_factory):
            raise ImportError('KafkaBus needs confluent-kafka: pip install "codigo-sintetico[kafka]"')
        self._ck = ck
        self._producer_factory = producer_factory or (lambda cfg: ck.Producer(cfg))
        self._consumer_factory = consumer_factory or (lambda cfg: ck.Consumer(cfg))
        self._admin_factory = admin_factory or (lambda cfg: ck.admin.AdminClient(cfg))
        try:
            self._producer = self._producer_factory({**self.base, "enable.idempotence": True, "acks": "all"})
        except Exception as exc:  # noqa: BLE001 - librdkafka rejects bad configuration with KafkaException
            raise BusError(f"invalid Kafka producer configuration: {exc}") from exc
        self._lock = threading.Lock()

    def ensure_topics(self, topics: List[str]) -> None:
        new_topic: Any
        try:
            from confluent_kafka.admin import NewTopic as new_topic
        except ImportError:  # pragma: no cover
            new_topic = None
        try:
            admin = self._admin_factory(self.base)
            new = [new_topic(t, num_partitions=self.partitions, replication_factor=self.replication) if new_topic else t for t in topics]
            futures = admin.create_topics(new)
        except Exception as exc:  # noqa: BLE001
            raise BusError(f"could not create topics {topics}: {exc}") from exc
        for topic, fut in (futures.items() if isinstance(futures, dict) else []):
            try:
                fut.result(timeout=self.flush_timeout)
            except Exception as exc:  # noqa: BLE001 - "already exists" is fine, anything else is not
                if "already exists" not in str(exc).lower() and "TOPIC_ALREADY_EXISTS" not in str(exc):
                    raise BusError(f"could not create topic {topic}: {exc}") from exc

    def publish(self, topic: str, value: bytes, key: Optional[str] = None) -> None:
        errors: List[Any] = []
        def on_delivery(err: Any, _msg: Any) -> None:
            if err is not None:
                errors.append(err)
        with self._lock:
            try:
                self._producer.produce(topic, value=value, key=key, on_delivery=on_delivery)
                remaining = self._producer.flush(self.flush_timeout)
            except Exception as exc:  # noqa: BLE001
                raise BusError(f"produce to {topic} failed: {exc}") from exc
        if errors:
            raise BusError(f"delivery to {topic} failed: {errors[0]}")
        if remaining:
            raise BusError(f"{remaining} message(s) not delivered to {topic} within {self.flush_timeout}s")

    def consumer(self, topics: List[str], group_id: str, *, from_beginning: bool = True) -> "_KafkaConsumer":
        cfg = {**self.base, "group.id": group_id, "enable.auto.commit": False, "auto.offset.reset": "earliest" if from_beginning else "latest"}
        try:
            return _KafkaConsumer(self._consumer_factory(cfg), topics, self._ck)
        except Exception as exc:  # noqa: BLE001
            raise BusError(f"invalid Kafka consumer configuration or subscription: {exc}") from exc

    def close(self) -> None:
        try:
            self._producer.flush(self.flush_timeout)
        except Exception:  # noqa: BLE001 - closing must not raise, but a lost flush is worth knowing about
            log.warning("KafkaBus.close(): final flush failed", exc_info=True)


class _KafkaConsumer:
    def __init__(self, client: Any, topics: List[str], ck: Any) -> None:
        self._c, self._ck = client, ck
        self._buffer: Deque[BusMessage] = deque()
        client.subscribe(list(topics))

    def _convert(self, m: Any) -> Optional[BusMessage]:
        err = m.error()
        if err is not None:
            code = err.code() if hasattr(err, "code") else None
            eof = getattr(getattr(self._ck, "KafkaError", None), "_PARTITION_EOF", object())
            if code == eof:
                return None
            raise BusError(f"consumer error: {err}")
        key = m.key()
        return BusMessage(m.topic(), key.decode("utf-8") if isinstance(key, bytes) else key, m.value(), m.offset(), m.partition(), raw=m)

    def poll(self, timeout: float) -> Optional[BusMessage]:
        if self._buffer:
            return self._buffer.popleft()
        m = self._c.poll(timeout)
        return None if m is None else self._convert(m)

    def commit(self, message: BusMessage) -> None:
        try:
            self._c.commit(message=message.raw, asynchronous=False)
        except Exception as exc:  # noqa: BLE001
            raise BusError(f"commit failed: {exc}") from exc

    def nack(self, message: BusMessage) -> None:
        if self._ck is not None:
            self._c.seek(self._ck.TopicPartition(message.topic, message.partition, message.offset))
        else:  # pragma: no cover - only with an injected client and no confluent_kafka installed
            self._buffer.appendleft(message)

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            m = self._c.poll(0.1)
            if m is not None:
                converted = self._convert(m)
                if converted is not None:
                    self._buffer.append(converted)
            if self._c.assignment():
                return True
        return False

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:  # noqa: BLE001
            log.warning("Kafka consumer close() failed", exc_info=True)


# --------------------------------------------------------------------------------------------------- worker
class AgentWorker:
    def __init__(self, bus: MessageBus, handler: Callable[[TaskEnvelope], Any], *, worker_id: Optional[str] = None,
                 group_id: str = "agent-workers", tasks_topic: str = TASKS_TOPIC, results_topic: str = RESULTS_TOPIC,
                 dlq_topic: str = DLQ_TOPIC, max_attempts: int = 3, task_timeout: float = 60.0, backoff_base: float = 0.5,
                 backoff_max: float = 30.0, dedupe_size: int = 10_000, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time, fault_hook: Optional[Callable[[str], None]] = None) -> None:
        if max_attempts < 1 or task_timeout <= 0:
            raise ValueError("max_attempts must be >= 1 and task_timeout > 0")
        self.bus, self.handler = bus, handler
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.tasks_topic, self.results_topic, self.dlq_topic = tasks_topic, results_topic, dlq_topic
        self.max_attempts, self.task_timeout, self.backoff_base, self.backoff_max = max_attempts, task_timeout, backoff_base, backoff_max
        self._sleep, self._clock, self._hook = sleep, clock, fault_hook
        self._done: "OrderedDict[str, None]" = OrderedDict()
        self._dedupe_size = dedupe_size
        self._stop = threading.Event()
        self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix=self.worker_id)
        self.metrics: Dict[str, int] = defaultdict(int)
        bus.ensure_topics([tasks_topic, results_topic, dlq_topic])
        self._consumer = bus.consumer([tasks_topic], group_id, from_beginning=True)

    # -- helpers
    def _fault(self, stage: str) -> None:
        if self._hook:
            self._hook(stage)

    def _remember(self, task_id: str) -> None:
        self._done[task_id] = None
        while len(self._done) > self._dedupe_size:
            self._done.popitem(last=False)

    def _result(self, env: TaskEnvelope, status: str, result: Any = None, error: Optional[str] = None) -> bytes:
        return ResultEnvelope(env.task_id, status, env.reply_to, result, error, self.worker_id, env.attempt, self._clock()).to_bytes()

    def _dlq(self, reason: str, env: Optional[TaskEnvelope], msg: BusMessage, error: Optional[str] = None) -> None:
        body = {"reason": reason, "error": error, "source": {"topic": msg.topic, "offset": msg.offset, "partition": msg.partition},
                "task": asdict(env) if env else None, "raw_b64": base64.b64encode(msg.value or b"").decode("ascii"),
                "worker_id": self.worker_id, "at": datetime.now(timezone.utc).isoformat()}
        self.bus.publish(self.dlq_topic, json.dumps(body).encode("utf-8"), key=env.task_id if env else None)
        self.metrics["dlq"] += 1

    def _execute(self, env: TaskEnvelope) -> Tuple[str, Any, Optional[str], bool]:
        """-> (status, result, error, retryable)"""
        future = self._pool.submit(self.handler, env)
        try:
            value = future.result(timeout=self.task_timeout)
            json.dumps(value)                                       # a result nobody can serialise is a failed task, found here
            return COMPLETED, value, None, False
        except cf.TimeoutError:
            self._pool.shutdown(wait=False, cancel_futures=True)    # the stuck thread cannot be killed; stop using it
            self._pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix=self.worker_id)
            return TIMEOUT, None, f"handler exceeded {self.task_timeout}s", True
        except RetryableError as exc:
            return FAILED, None, f"{type(exc).__name__}: {exc}", True
        except TypeError as exc:
            return FAILED, None, f"result is not JSON-serialisable: {exc}", False
        except Exception as exc:  # noqa: BLE001 - handler bugs are task failures, not worker crashes
            return FAILED, None, f"{type(exc).__name__}: {exc}", False

    # -- one message
    def _process(self, msg: BusMessage) -> str:
        try:
            env = TaskEnvelope.from_bytes(msg.value)
        except ValueError as exc:
            self._dlq("undecodable", None, msg, str(exc))
            self._consumer.commit(msg)
            self.metrics["poison"] += 1
            return "dlq"
        if env.task_id in self._done:
            self._consumer.commit(msg)
            self.metrics["duplicates"] += 1
            return "duplicate"
        if env.deadline is not None and self._clock() > env.deadline:
            self.bus.publish(self.results_topic, self._result(env, TIMEOUT, error="deadline passed before the task started"), key=env.reply_to)
            self._remember(env.task_id)
            self._consumer.commit(msg)
            self.metrics[TIMEOUT] += 1
            return TIMEOUT
        status, value, error, retryable = self._execute(env)
        self._fault("after_handler")
        if status != COMPLETED and retryable and env.attempt < self.max_attempts:
            self._sleep(min(self.backoff_max, self.backoff_base * 2 ** (env.attempt - 1)))
            retry = TaskEnvelope(env.task_id, env.task_type, env.payload, env.reply_to, env.attempt + 1, env.created_at, env.deadline)
            self.bus.publish(self.tasks_topic, retry.to_bytes(), key=env.task_id)
            self._consumer.commit(msg)
            self.metrics["retried"] += 1
            return "retry"
        if status != COMPLETED:
            self._dlq("failed" if status == FAILED else "timeout", env, msg, error)
        self.bus.publish(self.results_topic, self._result(env, status, value, error), key=env.reply_to)
        self._fault("after_publish")
        self._remember(env.task_id)
        self._consumer.commit(msg)
        self.metrics[status] += 1
        return status

    def run_once(self, poll_timeout: float = 1.0) -> Optional[str]:
        """Process at most one message. Returns what happened (completed/failed/timeout/retry/dlq/duplicate) or None."""
        msg = self._consumer.poll(poll_timeout)
        if msg is None:
            return None
        try:
            return self._process(msg)
        except BusError:
            self._consumer.nack(msg)                    # not committed, and made deliverable again
            raise

    def run_forever(self, poll_timeout: float = 0.5, error_backoff: float = 1.0) -> None:
        try:
            while not self._stop.is_set():
                try:
                    self.run_once(poll_timeout)
                except BusError:
                    self.metrics["bus_errors"] += 1
                    self._stop.wait(error_backoff)
        finally:
            self.close()

    def start(self, **kwargs: Any) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, kwargs=kwargs, name=self.worker_id, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        self._consumer.close()
        self._pool.shutdown(wait=False, cancel_futures=True)


# -------------------------------------------------------------------------------------------------- gateway
class DistributedAIGateway:
    """Publishes tasks and tracks their results. ``submit_task`` never blocks on the work itself."""

    def __init__(self, bus: MessageBus, *, tasks_topic: str = TASKS_TOPIC, results_topic: str = RESULTS_TOPIC,
                 gateway_id: Optional[str] = None, max_pending: int = 1000, max_retained: int = 10_000,
                 default_timeout: Optional[float] = None, ready_timeout: float = 15.0, clock: Callable[[], float] = time.time) -> None:
        self.bus, self.tasks_topic, self.results_topic = bus, tasks_topic, results_topic
        self.gateway_id = gateway_id or f"gw-{uuid.uuid4().hex[:8]}"
        self.max_pending, self.max_retained, self.default_timeout, self.ready_timeout, self._clock = max_pending, max_retained, default_timeout, ready_timeout, clock
        self._tasks: "OrderedDict[str, AgentTask]" = OrderedDict()
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consumer: Optional[BusConsumer] = None

    def __enter__(self) -> "DistributedAIGateway":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self.bus.ensure_topics([self.tasks_topic, self.results_topic])
        self._consumer = self.bus.consumer([self.results_topic], f"gateway-{self.gateway_id}", from_beginning=False)
        if not self._consumer.wait_ready(self.ready_timeout):
            self._consumer.close()
            raise BusError(f"results consumer not ready after {self.ready_timeout}s")
        self._thread = threading.Thread(target=self._pump, name=self.gateway_id, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._consumer:
            self._consumer.close()

    def _pump(self) -> None:
        consumer = self._consumer
        if consumer is None:
            return
        while not self._stop.is_set():
            try:
                msg = consumer.poll(0.2)
                if msg is None:
                    continue
                try:
                    self._on_result(ResultEnvelope.from_bytes(msg.value))
                except ValueError:
                    pass                                                # not ours / not a result: ignore
            except BusError:
                self._stop.wait(0.5)

    def _on_result(self, res: ResultEnvelope) -> None:
        if res.reply_to != self.gateway_id:
            return                                                       # another gateway's task
        with self._lock:
            task = self._tasks.get(res.task_id)
            if task is None or task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TIMEOUT):
                return                                                   # unknown, or a duplicate result (at-least-once delivery)
            task.status = {COMPLETED: TaskStatus.COMPLETED, FAILED: TaskStatus.FAILED, TIMEOUT: TaskStatus.TIMEOUT}[res.status]
            task.result = res.result if isinstance(res.result, str) or res.result is None else json.dumps(res.result)
            task.error = res.error
            self._events[res.task_id].set()
            self._trim()

    def _trim(self) -> None:
        terminal = [t for t, v in self._tasks.items() if v.status not in (TaskStatus.PENDING, TaskStatus.PROCESSING)]
        for t in terminal[: max(0, len(terminal) - self.max_retained)]:
            self._tasks.pop(t, None); self._events.pop(t, None)

    def pending(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.PROCESSING))

    def submit_task(self, task_type: str, payload: Dict[str, Any], *, timeout: Optional[float] = None) -> str:
        if self._thread is None:
            raise RuntimeError("call start() (or use the gateway as a context manager) before submitting tasks")
        limit = timeout if timeout is not None else self.default_timeout
        env = TaskEnvelope(uuid.uuid4().hex, task_type, payload, self.gateway_id, 1, self._clock(), self._clock() + limit if limit else None)
        data = env.to_bytes()                                            # validates JSON-serialisability before registering anything
        with self._lock:
            if sum(1 for t in self._tasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.PROCESSING)) >= self.max_pending:
                raise GatewayOverloaded(f"{self.max_pending} tasks already pending")
            self._tasks[env.task_id] = AgentTask(env.task_id, TaskStatus.PENDING, payload, datetime.now(timezone.utc).isoformat())
            self._events[env.task_id] = threading.Event()
        try:
            self.bus.publish(self.tasks_topic, data, key=env.task_id)    # registered first: a very fast result must find its task
        except BusError:
            with self._lock:
                self._tasks.pop(env.task_id, None); self._events.pop(env.task_id, None)
            raise
        return env.task_id

    def get_task(self, task_id: str) -> AgentTask:
        with self._lock:
            return self._tasks[task_id]

    def wait_for_result(self, task_id: str, timeout: Optional[float] = None) -> AgentTask:
        with self._lock:
            event = self._events[task_id]
        if not event.wait(timeout):
            raise TaskTimeout(f"no result for task {task_id} after {timeout}s")
        return self.get_task(task_id)


def react_handler(agent_factory: Callable[[], Any]) -> Callable[[TaskEnvelope], Any]:
    """Adapt a ReAct agent (``ReActAgentResilient``) to a worker handler: payload ``{"query": "..."}``."""
    def handle(task: TaskEnvelope) -> Any:
        query = task.payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("payload.query (non-empty str) is required")
        res = agent_factory().run(query)
        if res.status == "success":
            return {"answer": res.answer, "steps": len(res.steps), "usage": res.usage}
        if res.status == "invalid_output":
            raise RetryableError(res.stop_reason)                        # a model flake: worth another attempt
        raise RuntimeError(f"agent {res.status}: {res.stop_reason}")     # loops, budgets: retrying would repeat them
    return handle
