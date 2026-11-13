"""Shared FastAPI dependencies.

`get_store` exposes a single `BaseTraceStore` instance per process (lazy
singleton). The backend is chosen automatically:

- If `SINTETICO_API_DB_URL` starts with `postgres://` or `postgresql://`,
  `PostgresTraceStore` is used (recommended for more than one service
  instance; see `sintetico_api.postgres_store`).
- Otherwise `TraceStore` (SQLite) is used on the file named in
  `SINTETICO_API_DB_PATH` (a local file by default). That is the right mode
  for the book's demo and for development.

Tests can override this dependency with
`app.dependency_overrides[get_store] = lambda: test_store`.
"""

from __future__ import annotations

import os
import threading

from .base import BaseTraceStore
from .store import TraceStore

_store: "BaseTraceStore | None" = None
_lock = threading.Lock()


def _create_store() -> BaseTraceStore:
    db_url = os.environ.get("SINTETICO_API_DB_URL")
    if db_url and db_url.startswith(("postgres://", "postgresql://")):
        from .postgres_store import PostgresTraceStore

        return PostgresTraceStore(db_url)

    db_path = os.environ.get("SINTETICO_API_DB_PATH", "sintetico_traces.db")
    return TraceStore(db_path)


def get_store() -> BaseTraceStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = _create_store()
    return _store


def reset_store_for_testing() -> None:
    """Tests only: force the next call to `get_store` to create a new
    instance."""
    global _store
    if _store is not None:
        _store.close()
    _store = None
