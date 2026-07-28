"""
sintetico_api.postgres_store — Backend de trazas sobre PostgreSQL, para
despliegues con más de una instancia del servicio.

Implementa exactamente el mismo contrato que
`sintetico_api.store.TraceStore` (ver `BaseTraceStore`), así que
`sintetico_api.deps.get_store()` puede devolver uno u otro sin que
routers ni servicios lo noten.

## Live-tail multi-instancia

El pub-sub en memoria de `TraceStore` sólo entrega eventos a los
suscriptores conectados a *ese mismo proceso*. Con varias instancias
detrás de un balanceador, un cliente SSE conectado a la instancia B
nunca vería un evento insertado por la instancia A.

`PostgresTraceStore` resuelve esto con el mecanismo nativo `LISTEN`/
`NOTIFY` de Postgres: cada instancia mantiene una conexión dedicada en
modo `LISTEN sintetico_events`, y `insert_event()` hace
`NOTIFY sintetico_events, '<id del evento>'` tras cada inserción (sólo
el id, nunca el payload completo: `NOTIFY` tiene un límite de 8000 bytes
por mensaje en Postgres, y el id es suficiente para que el hilo oyente
recupere la fila completa con un `SELECT`). Así, un evento insertado en
cualquier instancia llega a los suscriptores locales de todas las demás.

Requiere `psycopg2` (`pip install -e ".[postgres]"`).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import BaseTraceStore

logger = logging.getLogger(__name__)

__all__ = ["PostgresTraceStore"]

_NOTIFY_CHANNEL = "sintetico_events"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sintetico_events (
    id BIGSERIAL PRIMARY KEY,
    correlation_id TEXT NOT NULL,
    parent_correlation_id TEXT,
    session_id TEXT,
    team_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    pillar TEXT,
    event_type TEXT NOT NULL,
    level TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    latency_ms DOUBLE PRECISION,
    cost_usd DOUBLE PRECISION,
    model_used TEXT,
    retry_count INTEGER,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_sintetico_events_correlation ON sintetico_events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_sintetico_events_run ON sintetico_events(run_id);
CREATE INDEX IF NOT EXISTS idx_sintetico_events_timestamp ON sintetico_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_sintetico_events_pillar ON sintetico_events(pillar);

CREATE TABLE IF NOT EXISTS sintetico_runs (
    run_id TEXT PRIMARY KEY,
    pillar TEXT NOT NULL,
    team_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    summary JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_sintetico_runs_started ON sintetico_runs(started_at);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class PostgresTraceStore(BaseTraceStore):
    """Backend de trazas sobre Postgres con live-tail multi-instancia.

    Uso:

        store = PostgresTraceStore("postgresql://user:pass@host:5432/sintetico")
    """

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 10):
        try:
            import psycopg2
            import psycopg2.extras
            import psycopg2.pool
        except ImportError as exc:
            raise ImportError('Instala el driver de Postgres: pip install -e ".[postgres]"') from exc

        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        self.dsn = dsn
        self._pool = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, dsn)
        self._write_lock = threading.Lock()

        with self._cursor(commit=True) as cur:
            cur.execute(_SCHEMA)

        self._subscribers: List["queue.Queue[Dict[str, Any]]"] = []
        self._subscribers_lock = threading.Lock()
        self._stop_listener = threading.Event()
        self._listening_ready = threading.Event()
        self._listener_thread = threading.Thread(target=self._listen_loop, name="sintetico-pg-listener", daemon=True)
        self._listener_thread.start()

        # Sin esta espera hay una condición de carrera real de arranque: si
        # `insert_event()` se llama inmediatamente después de construir el
        # store (p. ej. un health-check que dispara una demo nada más
        # arrancar), el NOTIFY podría emitirse antes de que el hilo oyente
        # haya ejecutado `LISTEN` en Postgres, y ese primer evento nunca
        # llegaría al live-tail. Se espera (con timeout) a que el hilo
        # confirme que ya está escuchando; si no lo confirma a tiempo, se
        # continúa igualmente (el resto de la API sigue funcionando) pero
        # se registra un aviso explícito.
        if not self._listening_ready.wait(timeout=5):
            logger.warning(
                "El listener de Postgres LISTEN/NOTIFY no confirmó estar activo tras 5s; "
                "el live-tail podría perder eventos hasta que se recupere."
            )

    # ─── Helpers de conexión ────────────────────────────────────────
    def _cursor(self, commit: bool = False):
        return _PooledCursor(self._pool, self._extras, commit=commit)

    def _listen_loop(self) -> None:
        """Hilo dedicado: escucha NOTIFY y republica en los suscriptores
        locales de este proceso. Una conexión de LISTEN debe mantenerse
        abierta indefinidamente, por eso no se toma del pool compartido."""
        conn = None
        try:
            conn = self._psycopg2.connect(self.dsn)
            conn.set_isolation_level(self._psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cur:
                cur.execute(f"LISTEN {_NOTIFY_CHANNEL};")
            self._listening_ready.set()
            while not self._stop_listener.is_set():
                import select

                if not select.select([conn], [], [], 2.0)[0]:
                    continue
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    self._handle_notify(notify.payload)
        except Exception:  # noqa: BLE001
            logger.exception("Listener de Postgres LISTEN/NOTIFY terminó con error")
        finally:
            if conn is not None:
                conn.close()

    def _handle_notify(self, event_id_str: str) -> None:
        try:
            event_id = int(event_id_str)
        except ValueError:
            return
        with self._cursor() as cur:
            cur.execute("SELECT * FROM sintetico_events WHERE id = %s", (event_id,))
            row = cur.fetchone()
        if row:
            self._publish(_row_to_event_dict(row))

    # ─── Eventos ────────────────────────────────────────────────────
    def insert_event(self, entry: Dict[str, Any], run_id: Optional[str] = None, pillar: Optional[str] = None) -> int:
        payload = entry.get("payload", {}) or {}
        timestamp = entry.get("timestamp") or datetime.now(timezone.utc).isoformat()

        with self._write_lock, self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO sintetico_events (correlation_id, parent_correlation_id, session_id, "
                "team_id, agent_id, run_id, pillar, event_type, level, timestamp, latency_ms, "
                "cost_usd, model_used, retry_count, payload) VALUES "
                "(%(correlation_id)s, %(parent_correlation_id)s, %(session_id)s, %(team_id)s, "
                "%(agent_id)s, %(run_id)s, %(pillar)s, %(event_type)s, %(level)s, %(timestamp)s, "
                "%(latency_ms)s, %(cost_usd)s, %(model_used)s, %(retry_count)s, %(payload)s) "
                "RETURNING id",
                {
                    "correlation_id": entry.get("correlation_id", ""),
                    "parent_correlation_id": entry.get("parent_correlation_id"),
                    "session_id": entry.get("session_id"),
                    "team_id": entry.get("team_id"),
                    "agent_id": entry.get("agent_id"),
                    "run_id": run_id,
                    "pillar": pillar,
                    "event_type": entry.get("event_type", "unknown"),
                    "level": entry.get("level", "INFO"),
                    "timestamp": timestamp,
                    "latency_ms": entry.get("latency_ms"),
                    "cost_usd": entry.get("cost_usd"),
                    "model_used": entry.get("model_used"),
                    "retry_count": entry.get("retry_count"),
                    "payload": self._extras.Json(payload),
                },
            )
            event_id = cur.fetchone()["id"]
            # Sólo el id: el payload completo podría superar el límite de
            # 8000 bytes de un mensaje NOTIFY. El listener recupera la
            # fila completa con un SELECT (ver `_handle_notify`).
            cur.execute("SELECT pg_notify(%s, %s)", (_NOTIFY_CHANNEL, str(event_id)))

        return event_id

    def get_trace(self, correlation_id: str) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM sintetico_events WHERE correlation_id = %s ORDER BY id ASC",
                (correlation_id,),
            )
            return [_row_to_event_dict(r) for r in cur.fetchall()]

    def list_events(
        self,
        run_id: Optional[str] = None,
        pillar: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 200,
        after_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], {}
        if run_id:
            clauses.append("run_id = %(run_id)s")
            params["run_id"] = run_id
        if pillar:
            clauses.append("pillar = %(pillar)s")
            params["pillar"] = pillar
        if level:
            clauses.append("level = %(level)s")
            params["level"] = level
        if after_id is not None:
            clauses.append("id > %(after_id)s")
            params["after_id"] = after_id
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params["limit"] = limit

        with self._cursor() as cur:
            cur.execute(f"SELECT * FROM sintetico_events {where} ORDER BY id DESC LIMIT %(limit)s", params)
            rows = [_row_to_event_dict(r) for r in cur.fetchall()]
        rows.reverse()
        return rows

    # ─── Ejecuciones (runs) ─────────────────────────────────────────
    def create_run(self, pillar: str, team_id: str = "demo") -> str:
        run_id = new_id(pillar)
        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO sintetico_runs (run_id, pillar, team_id, status, started_at) "
                "VALUES (%s, %s, %s, 'running', %s)",
                (run_id, pillar, team_id, datetime.now(timezone.utc)),
            )
        return run_id

    def finish_run(self, run_id: str, status: str, summary: Optional[Dict[str, Any]] = None) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "UPDATE sintetico_runs SET status = %s, finished_at = %s, summary = %s WHERE run_id = %s",
                (status, datetime.now(timezone.utc), self._extras.Json(summary or {}), run_id),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM sintetico_runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
        return _row_to_run_dict(row) if row else None

    def list_runs(self, pillar: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            if pillar:
                cur.execute(
                    "SELECT * FROM sintetico_runs WHERE pillar = %s ORDER BY started_at DESC LIMIT %s",
                    (pillar, limit),
                )
            else:
                cur.execute("SELECT * FROM sintetico_runs ORDER BY started_at DESC LIMIT %s", (limit,))
            return [_row_to_run_dict(r) for r in cur.fetchall()]

    # ─── Métricas agregadas ─────────────────────────────────────────
    def metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM sintetico_events WHERE timestamp >= NOW() - (%s * INTERVAL '1 minute') "
                "ORDER BY id DESC LIMIT 5000",
                (window_minutes,),
            )
            rows = [_row_to_event_dict(r) for r in cur.fetchall()]
            cur.execute("SELECT status, COUNT(*) AS c FROM sintetico_runs GROUP BY status")
            runs_by_status = {r["status"]: r["c"] for r in cur.fetchall()}

        total_cost, latencies = 0.0, []
        by_level: Dict[str, int] = {}
        by_event_type: Dict[str, int] = {}
        by_pillar: Dict[str, int] = {}
        by_model: Dict[str, float] = {}

        for row in rows:
            if row.get("cost_usd"):
                total_cost += row["cost_usd"]
                if row.get("model_used"):
                    by_model[row["model_used"]] = by_model.get(row["model_used"], 0.0) + row["cost_usd"]
            if row.get("latency_ms") is not None:
                latencies.append(row["latency_ms"])
            by_level[row["level"]] = by_level.get(row["level"], 0) + 1
            by_event_type[row["event_type"]] = by_event_type.get(row["event_type"], 0) + 1
            if row.get("pillar"):
                by_pillar[row["pillar"]] = by_pillar.get(row["pillar"], 0) + 1

        return {
            "window_minutes": window_minutes,
            "total_events": len(rows),
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 2) if latencies else 0.0,
            "events_by_level": by_level,
            "events_by_type": by_event_type,
            "events_by_pillar": by_pillar,
            "cost_by_model": {k: round(v, 6) for k, v in by_model.items()},
            "runs_by_status": runs_by_status,
        }

    # ─── Live tail (pub-sub sobre LISTEN/NOTIFY) ────────────────────
    def subscribe(self) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)
        with self._subscribers_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, subscriber: "queue.Queue[Dict[str, Any]]") -> None:
        with self._subscribers_lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)

    def _publish(self, event: Dict[str, Any]) -> None:
        with self._subscribers_lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def close(self) -> None:
        self._stop_listener.set()
        self._listener_thread.join(timeout=5)
        self._pool.closeall()


class _PooledCursor:
    """Context manager que toma una conexión del pool, expone un cursor
    con resultados como dict, hace commit/rollback según corresponda, y
    siempre devuelve la conexión al pool."""

    def __init__(self, pool, extras_module, commit: bool = False):
        self._pool = pool
        self._extras = extras_module
        self._commit = commit
        self._conn = None
        self._cur = None

    def __enter__(self):
        self._conn = self._pool.getconn()
        self._cur = self._conn.cursor(cursor_factory=self._extras.RealDictCursor)
        return self._cur

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None and self._commit:
                self._conn.commit()
            elif exc_type is not None:
                self._conn.rollback()
        finally:
            self._cur.close()
            self._pool.putconn(self._conn)
        return False


def _row_to_event_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    if isinstance(d.get("timestamp"), datetime):
        d["timestamp"] = d["timestamp"].isoformat()
    if isinstance(d.get("payload"), str):
        try:
            d["payload"] = json.loads(d["payload"])
        except json.JSONDecodeError:
            d["payload"] = {}
    elif d.get("payload") is None:
        d["payload"] = {}
    return d


def _row_to_run_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    for key in ("started_at", "finished_at"):
        if isinstance(d.get(key), datetime):
            d[key] = d[key].isoformat()
    if isinstance(d.get("summary"), str):
        try:
            d["summary"] = json.loads(d["summary"])
        except json.JSONDecodeError:
            d["summary"] = {}
    elif d.get("summary") is None:
        d["summary"] = {}
    return d


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]
