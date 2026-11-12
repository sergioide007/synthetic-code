"""
sintetico.orchestration — routing by complexity, response cache,
asynchronous gateway and budget control in swarms (Chapters 3, 4 and 18).
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .model_registry import resolve_model_id

__all__ = [
    "TaskStatus",
    "AgentTask",
    "AsyncAIGateway",
    "CycleDetector",
    "SwarmTokenOrchestrator",
    "ModelRouter",
    "SemanticCache",
    "RAGConfig",
    "RAGOptimizer",
]


# ═══════════════════════════════════════════════════════════════════
# Asynchronous gateway (decoupled task queue)
# ═══════════════════════════════════════════════════════════════════


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AgentTask:
    task_id: str
    status: TaskStatus
    payload: dict
    created_at: str
    result: Optional[str] = None
    error: Optional[str] = None


class AsyncAIGateway:
    """Asynchronous gateway with an in-memory queue and a thread pool.

    In production, `_process_async` would be replaced by publishing a
    message to Kafka/SQS/PubSub, and `wait_for_result` by a consumer of
    the result (webhook, polling the database, etc.). This implementation
    is a stand-in for that architecture, so it can be demonstrated with
    no external infrastructure. The Kafka-backed version is
    `sintetico.distributed.DistributedAIGateway`.
    """

    def __init__(self, team_id: str, budget: float, max_workers: int = 8):
        self.team_id = team_id
        # Lazy import to avoid a circular dependency with budget.py
        from .budget import TokenBudget

        self.budget = TokenBudget(budget, team_id=team_id)
        self._task_cache: Dict[str, AgentTask] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ai-gateway")

    def submit_task(self, task_type: str, payload: dict) -> str:
        raw = f"{task_type}{datetime.now(timezone.utc).isoformat()}{id(payload)}"
        task_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
        task = AgentTask(task_id, TaskStatus.PENDING, payload, datetime.now(timezone.utc).isoformat())
        with self._lock:
            self._task_cache[task_id] = task
        self._executor.submit(self._process, task_id, task_type, payload)
        return task_id

    def _process(self, task_id: str, task_type: str, payload: dict) -> None:
        with self._lock:
            task = self._task_cache.get(task_id)
            if task:
                task.status = TaskStatus.PROCESSING
        try:
            time.sleep(0.1)  # Simulates the latency of real work
            with self._lock:
                task = self._task_cache.get(task_id)
                if task:
                    task.status = TaskStatus.COMPLETED
                    task.result = f"Resultado de {task_type}"
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                task = self._task_cache.get(task_id)
                if task:
                    task.status = TaskStatus.FAILED
                    task.error = str(exc)

    def wait_for_result(self, task_id: str, timeout: float = 5.0, poll_interval: float = 0.05) -> AgentTask:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                task = self._task_cache.get(task_id)
            if task and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return task
            time.sleep(poll_interval)
        with self._lock:
            task = self._task_cache.get(task_id)
            if task:
                task.status = TaskStatus.TIMEOUT
        return task

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


# ═══════════════════════════════════════════════════════════════════
# Loop detection and budget control in swarms
# ═══════════════════════════════════════════════════════════════════


class CycleDetector:
    """Detects repeated A→B / B→A ping-pong patterns between agents."""

    def __init__(self, repetition_count: int = 3):
        self.repetition_count = repetition_count

    def check_cycle(self, messages: List[Dict]) -> Tuple[str, str]:
        if len(messages) < self.repetition_count * 2:
            return "no_cycle", "Insufficient messages"
        window = messages[-self.repetition_count * 2 :]
        pairs = [(m.get("from"), m.get("to")) for m in window]
        for i in range(0, len(pairs) - 1, 2):
            if pairs[i] == pairs[i + 1][::-1]:
                return "cycle_detected", f"Pattern: {pairs[i]}"
        return "no_cycle", "OK"


class SwarmTokenOrchestrator:
    """Swarm orchestrator with a token budget and cycle detection."""

    def __init__(
        self, session_id: str, total_budget_tokens: int, warning_threshold: float = 0.70, halt_threshold: float = 0.90
    ):
        if total_budget_tokens <= 0:
            raise ValueError("total_budget_tokens debe ser positivo")
        self.session_id = session_id
        self.total_budget = total_budget_tokens
        self.spent_tokens = 0
        self.warning_threshold = warning_threshold
        self.halt_threshold = halt_threshold
        self.cycle_detector = CycleDetector()
        self.message_history: List[Dict] = []

    def add_message(self, from_agent: str, to_agent: str, content: str, tokens_cost: int) -> Dict:
        self.spent_tokens += tokens_cost
        budget_pct = self.spent_tokens / self.total_budget
        self.message_history.append({"from": from_agent, "to": to_agent, "content": content[:50]})

        if budget_pct >= self.halt_threshold:
            return {"decision": "HALT", "reason": f"Budget {budget_pct:.1%}"}
        if budget_pct >= self.warning_threshold:
            return {"decision": "DEGRADE", "reason": f"Warning {budget_pct:.1%}"}

        cycle_status, reason = self.cycle_detector.check_cycle(self.message_history)
        if cycle_status == "cycle_detected":
            return {"decision": "HALT", "reason": f"Cycle: {reason}"}
        return {"decision": "CONTINUE", "reason": "OK"}


# ═══════════════════════════════════════════════════════════════════
# Routing by complexity and response cache
# ═══════════════════════════════════════════════════════════════════


class ModelRouter:
    """Routes a query to the cheapest model able to answer it.

    Returns logical *aliases* ("haiku", "sonnet", "opus"), not concrete
    API ids: resolving the real id (with its version/date suffix) is the
    job of `sintetico.model_registry`, so a change in Anthropic's model
    catalog does not require touching this class.
    """

    RULES: List[Tuple[str, str]] = [
        (r"^(hola|hi|buenas|hey)\b", "haiku"),
        (r"\b(estado|status|d[oó]nde)\b", "haiku"),
        (r"\b(error|debug|fallo|bug)\b", "sonnet"),
        (r"\b(c[oó]digo|code|review|revisar)\b", "sonnet"),
        (r"\b(arquitectura|design|dise[ñn]a|sistema|estrategia|migraci[oó]n)\b", "opus"),
    ]
    DEFAULT_ALIAS = "sonnet"

    def select_model(self, query: str) -> str:
        """Return the chosen model alias ('haiku' | 'sonnet' | 'opus')."""
        lowered = query.lower()
        for pattern, alias in self.RULES:
            if re.search(pattern, lowered):
                return alias
        return self.DEFAULT_ALIAS

    def select_model_id(self, query: str) -> str:
        """Return the real model id to send to the API."""
        return resolve_model_id(self.select_model(query))


class SemanticCache:
    """Thread-safe LRU cache for the answers to repeated queries.

    It is an *exact* cache keyed by the normalized query text, not an
    embedding-based semantic cache: that is the natural next step (see
    `sintetico.semantic_cache`), but it needs an embedding model and a
    similarity index, which are outside the scope of this base component.
    """

    def __init__(self, max_size: int = 10_000):
        if max_size <= 0:
            raise ValueError("max_size debe ser positivo")
        self.max_size = max_size
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.strip().lower().split())

    def get(self, query: str) -> Optional[str]:
        key = self._normalize(query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def set(self, query: str, response: str) -> None:
        key = self._normalize(query)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = response

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ═══════════════════════════════════════════════════════════════════
# RAG optimization
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class RAGConfig:
    chunk_size: int
    chunk_overlap: int
    top_k_final: int


class RAGOptimizer:
    CONFIGURATIONS: Dict[str, RAGConfig] = {
        "code": RAGConfig(300, 50, 3),
        "docs": RAGConfig(500, 75, 3),
        "legal": RAGConfig(800, 100, 5),
    }

    def optimize_for_domain(self, domain: str) -> RAGConfig:
        return self.CONFIGURATIONS.get(domain, self.CONFIGURATIONS["docs"])

    def estimate_context_tokens(self, config: RAGConfig, query_length: int = 50) -> int:
        return 200 + config.top_k_final * config.chunk_size + int(query_length * 1.3)

    def calculate_cost_reduction(
        self, before: RAGConfig, after: RAGConfig, cost_per_million_input: float = 3.0, monthly_queries: int = 10_000
    ) -> Dict:
        tokens_before = self.estimate_context_tokens(before)
        tokens_after = self.estimate_context_tokens(after)
        reduction = (tokens_before - tokens_after) / tokens_before * 100 if tokens_before > 0 else 0
        monthly_savings = (tokens_before - tokens_after) / 1_000_000 * cost_per_million_input * monthly_queries
        return {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction_percent": round(reduction, 1),
            "estimated_monthly_savings_usd": round(monthly_savings, 2),
        }
