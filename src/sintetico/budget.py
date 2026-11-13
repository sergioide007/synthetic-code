"""
sintetico.budget — token budget control (Chapter 3).

Provides a swappable backend interface that keeps track of the spend
accumulated by each team, with an in-memory implementation (for tests and
single-process demos) and a distributed implementation on Redis that
guarantees atomic "check-and-increment" through a Lua script.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)

__all__ = [
    "BudgetBackend",
    "LocalMemoryBudgetBackend",
    "RedisBudgetBackend",
    "RedisUnavailableError",
    "TokenBudget",
]


class RedisUnavailableError(RuntimeError):
    """Raised when a Redis backend is used without a connection."""


class BudgetBackend(ABC):
    """Interface for atomic, distributed storage of the budget."""

    @abstractmethod
    def get_spent(self, key: str) -> float:
        """Return the spend accumulated for `key`."""

    @abstractmethod
    def add_cost(self, key: str, amount: float, limit: float) -> bool:
        """Try to add `amount` to `key` without exceeding `limit`.

        Must be atomic: if two concurrent calls check the limit at the same
        time, only one of them may succeed when together they exceed it.
        """


class LocalMemoryBudgetBackend(BudgetBackend):
    """Local in-memory backend with a lock, atomic within one process.

    Suitable for tests, demos and single-node deployments. Not suitable
    for several processes/instances, because the state is not shared
    between them.
    """

    def __init__(self) -> None:
        self._store: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get_spent(self, key: str) -> float:
        with self._lock:
            return self._store.get(key, 0.0)

    def add_cost(self, key: str, amount: float, limit: float) -> bool:
        with self._lock:
            current = self._store.get(key, 0.0)
            if current + amount > limit:
                return False
            self._store[key] = current + amount
            return True


class RedisBudgetBackend(BudgetBackend):
    """Distributed Redis backend for multi-node budget control.

    Uses a Lua script for the atomic check-and-increment operation, so
    that several processes/instances share one consistent counter with no
    race conditions.

    Note: the key arrives already formatted from `TokenBudget` (for
    example ``"budget:team_id"``), so the default prefix is empty to
    avoid duplicating it.
    """

    _INCR_SCRIPT = """
    local current = tonumber(redis.call('GET', KEYS[1]) or '0')
    local amount = tonumber(ARGV[1])
    local limit = tonumber(ARGV[2])
    if current + amount > limit then
        return 0
    end
    return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
    """

    def __init__(
        self, host: str = "localhost", port: int = 6379, db: int = 0, prefix: str = "", socket_timeout: float = 2.0
    ) -> None:
        try:
            import redis  # local import so the dependency is not mandatory
        except ImportError as exc:
            raise ImportError("Install redis-py: pip install redis") from exc

        self.prefix = prefix
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=socket_timeout,
        )
        try:
            self.redis.ping()
            self._available = True
            self._connection_error: str | None = None
            self._incr_script = self.redis.register_script(self._INCR_SCRIPT)
        except redis.exceptions.RedisError as exc:
            self._available = False
            self._connection_error = str(exc)
            self._incr_script = None
            logger.warning("No se pudo conectar a Redis (%s:%s): %s", host, port, exc)

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    def get_spent(self, key: str) -> float:
        if not self._available:
            raise RedisUnavailableError(self._connection_error or "Redis no disponible")
        value = self.redis.get(self._full_key(key))
        return float(value) if value else 0.0

    def add_cost(self, key: str, amount: float, limit: float) -> bool:
        if not self._available:
            raise RedisUnavailableError(self._connection_error or "Redis no disponible")

        full_key = self._full_key(key)
        try:
            result = self._incr_script(keys=[full_key], args=[amount, limit])
            return result != 0
        except Exception:
            logger.exception("Lua script failed; falling back to WATCH/MULTI")
            return self._add_cost_with_optimistic_lock(full_key, amount, limit)

    def _add_cost_with_optimistic_lock(self, full_key: str, amount: float, limit: float) -> bool:
        """Fallback with optimistic locking (WATCH/MULTI) if the Lua script fails."""
        import redis as redis_module

        with self.redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(full_key)
                    current = float(pipe.get(full_key) or 0.0)
                    if current + amount > limit:
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.incrbyfloat(full_key, amount)
                    pipe.execute()
                    return True
                except redis_module.WatchError:
                    continue


@dataclass
class TokenBudget:
    """Monthly token budget with a swappable backend (local/Redis).

    Progressive alerts (`alerts_triggered`) are kept in local memory per
    instance: in a multi-node deployment each node emits its own alert the
    first time it observes the threshold. That is acceptable for logging,
    but it must not be used as the source of truth for business decisions
    (the backend itself, which is consistent, is there for that).
    """

    monthly_budget: float
    backend: BudgetBackend = field(default_factory=LocalMemoryBudgetBackend)
    team_id: str = "default"
    alerts_triggered: Dict[float, bool] = field(default_factory=dict)
    on_alert: "callable | None" = None

    def __post_init__(self) -> None:
        if self.monthly_budget <= 0:
            raise ValueError("monthly_budget must be positive")

    @property
    def _key(self) -> str:
        return f"budget:{self.team_id}"

    @property
    def spent(self) -> float:
        """Spend accumulated for this team."""
        return self.backend.get_spent(self._key)

    @property
    def remaining(self) -> float:
        return max(self.monthly_budget - self.spent, 0.0)

    def record_cost(self, cost: float) -> bool:
        """Record `cost` if it does not exceed the budget. Return whether it succeeded."""
        if cost < 0:
            raise ValueError("cost cannot be negative")
        if not self.backend.add_cost(self._key, cost, self.monthly_budget):
            return False
        self._check_alerts()
        return True

    def _check_alerts(self) -> None:
        spent = self.backend.get_spent(self._key)
        pct = spent / self.monthly_budget
        for threshold in (0.5, 0.75, 0.9, 1.0):
            if pct >= threshold and not self.alerts_triggered.get(threshold):
                self.alerts_triggered[threshold] = True
                message = f"Budget {int(threshold * 100)}% reached for {self.team_id}"
                logger.warning(message)
                if self.on_alert:
                    self.on_alert(threshold, self.team_id, spent)
