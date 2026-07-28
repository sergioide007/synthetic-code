"""
sintetico_api.store — Persistencia de trazas y ejecuciones para el
dashboard de observabilidad, y un pub-sub en memoria para el live-tail
(estilo "Live Tail" de Datadog / CloudWatch Logs Insights).

Deliberadamente NO depende de FastAPI ni Pydantic: es lógica de dominio
pura sobre `sqlite3` (stdlib), así que se puede probar con `pytest` sin
necesidad de un cliente HTTP ni de tener esos paquetes instalados. La capa
web (routers/schemas) es un adaptador delgado sobre esta clase.
"""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import BaseTraceStore

__all__ = ["TraceStore", "new_id"]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    correlation_id TEXT NOT NULL,
    parent_correlation_id TEXT,
    session_id TEXT,
    team_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    pillar TEXT,
    event_type TEXT NOT NULL,
    level TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    latency_ms REAL,
    cost_usd REAL,
    model_used TEXT,
    retry_count INTEGER,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_pillar ON events(pillar);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    pillar TEXT NOT NULL,
    team_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
"""


class TraceStore(BaseTraceStore):
    """Almacén de trazas y ejecuciones respaldado por SQLite.

    Implementación de referencia de `BaseTraceStore` para un único
    proceso/instancia (demo del libro, desarrollo local). Para un
    despliegue con varias instancias, usa
    `sintetico_api.postgres_store.PostgresTraceStore`, que respeta el
    mismo contrato.

    Uso típico en el proceso de la API (un único `TraceStore` compartido,
    instanciado una vez en el arranque de la aplicación):

        store = TraceStore("traces.db")
        run_id = store.create_run(pillar="orchestration", team_id="demo")
        store.insert_event({...}, run_id=run_id, pillar="orchestration")
        store.finish_run(run_id, status="completed", summary={...})
    """

    def __init__(self, db_path: str = "sintetico_traces.db", max_events: int = 50_000):
        self.db_path = db_path
        self.max_events = max_events
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        self._subscribers: List["queue.Queue[Dict[str, Any]]"] = []

    # ─── Eventos ────────────────────────────────────────────────────
    def insert_event(self, entry: Dict[str, Any], run_id: Optional[str] = None, pillar: Optional[str] = None) -> int:
        """Inserta un evento de traza. `entry` sigue la forma que produce
        `StructuredAgentLogger` (ver `trazabilidad.logger.StructuredLogEntry`),
        pero acepta cualquier dict con esas claves — no depende de esa
        clase para no crear un acoplamiento circular entre paquetes.
        """
        payload = entry.get("payload", {}) or {}
        row = {
            "correlation_id": entry.get("correlation_id", ""),
            "parent_correlation_id": entry.get("parent_correlation_id"),
            "session_id": entry.get("session_id"),
            "team_id": entry.get("team_id"),
            "agent_id": entry.get("agent_id"),
            "run_id": run_id,
            "pillar": pillar,
            "event_type": entry.get("event_type", "unknown"),
            "level": entry.get("level", "INFO"),
            "timestamp": entry.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "latency_ms": entry.get("latency_ms"),
            "cost_usd": entry.get("cost_usd"),
            "model_used": entry.get("model_used"),
            "retry_count": entry.get("retry_count"),
            "payload": json.dumps(payload, default=str, ensure_ascii=False),
        }
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (correlation_id, parent_correlation_id, session_id, team_id, "
                "agent_id, run_id, pillar, event_type, level, timestamp, latency_ms, cost_usd, "
                "model_used, retry_count, payload) VALUES "
                "(:correlation_id, :parent_correlation_id, :session_id, :team_id, :agent_id, "
                ":run_id, :pillar, :event_type, :level, :timestamp, :latency_ms, :cost_usd, "
                ":model_used, :retry_count, :payload)",
                row,
            )
            self._conn.commit()
            event_id = cur.lastrowid
            self._trim_if_needed()

        row["id"] = event_id
        row["payload"] = payload  # publicar el dict, no el JSON serializado
        self._publish(row)
        return event_id

    def _trim_if_needed(self) -> None:
        """Poda eventos antiguos si se supera `max_events` (contención de
        memoria/disco para una demo de larga duración)."""
        cur = self._conn.execute("SELECT COUNT(*) FROM events")
        count = cur.fetchone()[0]
        if count > self.max_events:
            excess = count - self.max_events
            self._conn.execute(
                "DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()

    def get_trace(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Devuelve todos los eventos de una traza, ordenados cronológicamente
        (la "waterfall" que se renderiza en el dashboard)."""
        with self._lock:
            cur = self._conn.execute("SELECT * FROM events WHERE correlation_id = ? ORDER BY id ASC", (correlation_id,))
            return [self._row_to_dict(r) for r in cur.fetchall()]

    def list_events(
        self,
        run_id: Optional[str] = None,
        pillar: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 200,
        after_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if pillar:
            clauses.append("pillar = ?")
            params.append(pillar)
        if level:
            clauses.append("level = ?")
            params.append(level)
        if after_id is not None:
            clauses.append("id > ?")
            params.append(after_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            cur = self._conn.execute(query, params)
            rows = [self._row_to_dict(r) for r in cur.fetchall()]
        rows.reverse()  # devolver en orden cronológico ascendente
        return rows

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["payload"] = json.loads(d["payload"]) if d.get("payload") else {}
        except (TypeError, json.JSONDecodeError):
            d["payload"] = {}
        return d

    # ─── Ejecuciones (runs) ─────────────────────────────────────────
    def create_run(self, pillar: str, team_id: str = "demo") -> str:
        run_id = new_id(pillar)
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (run_id, pillar, team_id, status, started_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, pillar, team_id, "running", datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()
        return run_id

    def finish_run(self, run_id: str, status: str, summary: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, summary = ? WHERE run_id = ?",
                (
                    status,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(summary or {}, default=str, ensure_ascii=False),
                    run_id,
                ),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["summary"] = json.loads(d["summary"]) if d.get("summary") else {}
        return d

    def list_runs(self, pillar: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            if pillar:
                cur = self._conn.execute(
                    "SELECT * FROM runs WHERE pillar = ? ORDER BY started_at DESC LIMIT ?", (pillar, limit)
                )
            else:
                cur = self._conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["summary"] = json.loads(d["summary"]) if d.get("summary") else {}
            result.append(d)
        return result

    # ─── Métricas agregadas ─────────────────────────────────────────
    def metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        cutoff = datetime.now(timezone.utc).timestamp() - window_minutes * 60
        with self._lock:
            cur = self._conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT 5000")
            recent_rows = cur.fetchall()

        total_cost = 0.0
        total_events = 0
        latencies: List[float] = []
        by_level: Dict[str, int] = {}
        by_event_type: Dict[str, int] = {}
        by_pillar: Dict[str, int] = {}
        by_model: Dict[str, float] = {}

        for row in recent_rows:
            try:
                ts = datetime.fromisoformat(row["timestamp"]).timestamp()
            except (ValueError, TypeError):
                ts = cutoff  # si no se puede parsear, no descartar el evento
            if ts < cutoff:
                continue
            total_events += 1
            if row["cost_usd"]:
                total_cost += row["cost_usd"]
                if row["model_used"]:
                    by_model[row["model_used"]] = by_model.get(row["model_used"], 0.0) + row["cost_usd"]
            if row["latency_ms"] is not None:
                latencies.append(row["latency_ms"])
            by_level[row["level"]] = by_level.get(row["level"], 0) + 1
            by_event_type[row["event_type"]] = by_event_type.get(row["event_type"], 0) + 1
            if row["pillar"]:
                by_pillar[row["pillar"]] = by_pillar.get(row["pillar"], 0) + 1

        with self._lock:
            cur = self._conn.execute("SELECT status, COUNT(*) c FROM runs GROUP BY status")
            runs_by_status = {r["status"]: r["c"] for r in cur.fetchall()}

        return {
            "window_minutes": window_minutes,
            "total_events": total_events,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 2) if latencies else 0.0,
            "events_by_level": by_level,
            "events_by_type": by_event_type,
            "events_by_pillar": by_pillar,
            "cost_by_model": {k: round(v, 6) for k, v in by_model.items()},
            "runs_by_status": runs_by_status,
        }

    # ─── Live tail (pub-sub) ────────────────────────────────────────
    def subscribe(self) -> "queue.Queue[Dict[str, Any]]":
        """Registra un nuevo suscriptor para live-tail. Cada evento nuevo
        insertado se publica a todas las colas suscritas."""
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[Dict[str, Any]]") -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, event: Dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                # Suscriptor lento (cliente SSE desconectado o saturado):
                # se descarta el evento para él en vez de bloquear al
                # productor. Es la semántica correcta para un live-tail.
                pass

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]
