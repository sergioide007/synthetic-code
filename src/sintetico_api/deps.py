"""Dependencias compartidas de FastAPI.

`get_store` expone una única instancia de `BaseTraceStore` por proceso
(patrón singleton perezoso). El backend se elige automáticamente:

- Si `SINTETICO_API_DB_URL` empieza por `postgres://` o `postgresql://`,
  se usa `PostgresTraceStore` (recomendado para más de una instancia del
  servicio, ver `sintetico_api.postgres_store`).
- En cualquier otro caso, se usa `TraceStore` (SQLite) sobre el fichero
  indicado en `SINTETICO_API_DB_PATH` (por defecto uno local). Es el
  modo correcto para la demo del libro y para desarrollo.

Los tests pueden sobreescribir esta dependencia con
`app.dependency_overrides[get_store] = lambda: store_de_prueba`.
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
    """Sólo para tests: fuerza que la próxima llamada a `get_store` cree
    una instancia nueva."""
    global _store
    if _store is not None:
        _store.close()
    _store = None
