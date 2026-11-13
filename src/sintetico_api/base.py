"""
sintetico_api.base: common interface for trace storage backends.
`TraceStore` (SQLite) and `PostgresTraceStore` implement the same contract, so
the API layer (`deps.get_store`) can pick either one from the environment
without any router or service having to know.

Why two backends:
- **SQLite** (`sintetico_api.store.TraceStore`): zero configuration, one local
  file. Perfect for the book's demo and for development, but not shared
  between processes or instances.
- **Postgres** (`sintetico_api.postgres_store.PostgresTraceStore`): for a real
  deployment with more than one instance behind a load balancer. It uses
  Postgres's native `LISTEN`/`NOTIFY` so the live tail (Server-Sent Events)
  works across instances: an event inserted by instance A shows up in real
  time on a client connected to instance B.

`sintetico.budget.TokenBudget` solves the same local-vs-distributed problem
with `BudgetBackend`/`RedisBudgetBackend`; this is the same idea applied to the
trace store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

__all__ = ["BaseTraceStore"]


class BaseTraceStore(ABC):
    """Contract every trace/run backend must fulfil."""

    # ─── Events ────────────────────────────────────────────────────
    @abstractmethod
    def insert_event(self, entry: Dict[str, Any], run_id: Optional[str] = None, pillar: Optional[str] = None) -> int:
        """Insert a trace event and publish it to live-tail subscribers.
        Return the assigned id."""

    @abstractmethod
    def get_trace(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Every event of a trace, in chronological order."""

    @abstractmethod
    def list_events(
        self,
        run_id: Optional[str] = None,
        pillar: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 200,
        after_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Recent events, optionally filtered."""

    # ─── Runs ─────────────────────────────────────────
    @abstractmethod
    def create_run(self, pillar: str, team_id: str = "demo") -> str: ...

    @abstractmethod
    def finish_run(self, run_id: str, status: str, summary: Optional[Dict[str, Any]] = None) -> None: ...

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def list_runs(self, pillar: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]: ...

    # ─── Metrics ───────────────────────────────────────────────────
    @abstractmethod
    def metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]: ...

    # ─── Live tail (pub-sub) ────────────────────────────────────────
    @abstractmethod
    def subscribe(self) -> "Any":
        """Register a live-tail subscriber; return a `queue.Queue`-like
        object you can `.get()` from."""

    @abstractmethod
    def unsubscribe(self, subscriber: "Any") -> None: ...

    @abstractmethod
    def close(self) -> None: ...
