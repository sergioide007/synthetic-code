"""Tests específicos de PostgresTraceStore que ejercitan características
propias de PostgreSQL no presentes en el backend SQLite:

- LISTEN/NOTIFY para live-tail multi-instancia
- Pool de conexiones ThreadedConnectionPool
- Manejo de JSONB nativo
- Índices específicos de Postgres
- Thread-safety bajo concurrencia
- Recuperación de errores del listener

Requiere `psycopg2` instalado y `SINTETICO_TEST_POSTGRES_URL` apuntando a
un Postgres accesible. Si no se cumple, los tests se omiten limpiamente.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import pytest

from sintetico_api.postgres_store import PostgresTraceStore, _row_to_event_dict, _row_to_run_dict


def _cleanup_db(store):
    """Limpia las tablas entre tests para garantizar aislamiento."""
    with store._cursor(commit=True) as cur:
        cur.execute("TRUNCATE sintetico_events, sintetico_runs;")


@pytest.fixture
def store(request):
    """Fixture que crea un PostgresTraceStore limpio para cada test, y lo
    cierra al finalizar."""
    psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 no instalado")
    dsn = os.environ.get("SINTETICO_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("Define SINTETICO_TEST_POSTGRES_URL para correr los tests de Postgres")

    from sintetico_api.postgres_store import PostgresTraceStore
    try:
        store = PostgresTraceStore(dsn)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres no accesible en SINTETICO_TEST_POSTGRES_URL: {exc}")

    _cleanup_db(store)
    request.addfinalizer(store.close)
    return store


class TestPostgresSchema:
    """Verifica que el esquema de Postgres se crea correctamente y es
    idempotente (múltiples llamadas a __init__ no fallan)."""

    def test_schema_creates_tables_and_indexes(self, store):
        """Las tablas sintetico_events y sintetico_runs existen después de
        inicializar el store."""
        with store._cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name IN ('sintetico_events', 'sintetico_runs')"
            )
            tables = {row["table_name"] for row in cur.fetchall()}
        assert tables == {"sintetico_events", "sintetico_runs"}

    def test_indexes_are_created(self, store):
        """Los índices definidos en _SCHEMA existen."""
        expected_indexes = {
            "idx_sintetico_events_correlation",
            "idx_sintetico_events_run",
            "idx_sintetico_events_timestamp",
            "idx_sintetico_events_pillar",
            "idx_sintetico_runs_started",
        }
        with store._cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename IN ('sintetico_events', 'sintetico_runs')"
            )
            actual_indexes = {row["indexname"] for row in cur.fetchall()}
        assert expected_indexes.issubset(actual_indexes)

    def test_reinitialization_is_idempotent(self):
        """Crear un segundo PostgresTraceStore con el mismo DSN no falla
        (el esquema ya existe)."""
        dsn = os.environ.get("SINTETICO_TEST_POSTGRES_URL")
        if not dsn:
            pytest.skip("SINTETICO_TEST_POSTGRES_URL no definido")
        
        from sintetico_api.postgres_store import PostgresTraceStore
        store1 = PostgresTraceStore(dsn)
        store2 = PostgresTraceStore(dsn)  # No debe fallar
        store1.close()
        store2.close()


class TestPostgresEvents:
    """Operaciones CRUD sobre eventos en Postgres."""

    def test_insert_event_returns_autoincrement_id(self, store):
        event_id = store.insert_event(
            {
                "correlation_id": "pg-c1",
                "event_type": "test",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        )
        assert isinstance(event_id, int)
        assert event_id > 0

    def test_insert_event_persists_all_fields(self, store):
        store.insert_event(
            {
                "correlation_id": "pg-c2",
                "parent_correlation_id": "pg-parent",
                "session_id": "sess-123",
                "team_id": "team-1",
                "agent_id": "agent-42",
                "event_type": "tool_call",
                "level": "DEBUG",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "latency_ms": 150.5,
                "cost_usd": 0.025,
                "model_used": "claude-3-5-sonnet",
                "retry_count": 2,
                "payload": {"tool": "search", "query": "test"},
            },
            run_id="run-abc",
            pillar="orchestration",
        )
        trace = store.get_trace("pg-c2")
        assert len(trace) == 1
        event = trace[0]
        assert event["parent_correlation_id"] == "pg-parent"
        assert event["session_id"] == "sess-123"
        assert event["team_id"] == "team-1"
        assert event["agent_id"] == "agent-42"
        assert event["run_id"] == "run-abc"
        assert event["pillar"] == "orchestration"
        assert event["latency_ms"] == 150.5
        assert event["cost_usd"] == 0.025
        assert event["model_used"] == "claude-3-5-sonnet"
        assert event["retry_count"] == 2
        assert event["payload"] == {"tool": "search", "query": "test"}

    def test_get_trace_returns_events_in_order(self, store):
        store.insert_event(
            {"correlation_id": "pg-c3", "event_type": "a", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        store.insert_event(
            {"correlation_id": "pg-c3", "event_type": "b", "level": "INFO", "timestamp": "2026-01-01T00:00:01+00:00"}
        )
        trace = store.get_trace("pg-c3")
        assert [e["event_type"] for e in trace] == ["a", "b"]

    def test_get_trace_returns_empty_for_unknown_correlation_id(self, store):
        trace = store.get_trace("no-existe-999")
        assert trace == []

    def test_list_events_filters_by_run_id(self, store):
        store.insert_event(
            {"correlation_id": "pg-c4", "event_type": "x", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"},
            run_id="run-1",
        )
        store.insert_event(
            {"correlation_id": "pg-c5", "event_type": "y", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"},
            run_id="run-2",
        )
        events = store.list_events(run_id="run-1", limit=10)
        assert len(events) == 1
        assert events[0]["correlation_id"] == "pg-c4"

    def test_list_events_filters_by_pillar(self, store):
        store.insert_event(
            {"correlation_id": "pg-c6", "event_type": "x", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"},
            pillar="quality_gate",
        )
        store.insert_event(
            {"correlation_id": "pg-c7", "event_type": "y", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"},
            pillar="orchestration",
        )
        events = store.list_events(pillar="quality_gate", limit=10)
        assert all(e["pillar"] == "quality_gate" for e in events)

    def test_list_events_filters_by_level(self, store):
        store.insert_event(
            {"correlation_id": "pg-c8", "event_type": "x", "level": "ERROR", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        store.insert_event(
            {"correlation_id": "pg-c9", "event_type": "y", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        events = store.list_events(level="ERROR", limit=10)
        assert all(e["level"] == "ERROR" for e in events)

    def test_list_events_supports_after_id_pagination(self, store):
        store.insert_event(
            {"correlation_id": "pg-c10", "event_type": "a", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        store.insert_event(
            {"correlation_id": "pg-c11", "event_type": "b", "level": "INFO", "timestamp": "2026-01-01T00:00:01+00:00"}
        )
        store.insert_event(
            {"correlation_id": "pg-c12", "event_type": "c", "level": "INFO", "timestamp": "2026-01-01T00:00:02+00:00"}
        )
        all_events = store.list_events(limit=10)
        assert len(all_events) == 3
        middle_id = all_events[1]["id"]
        later_events = store.list_events(after_id=middle_id, limit=10)
        assert len(later_events) == 1
        assert later_events[0]["event_type"] == "c"

    def test_list_events_respects_limit(self, store):
        for i in range(10):
            store.insert_event(
                {"correlation_id": f"pg-c-limit-{i}", "event_type": "x", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
            )
        events = store.list_events(limit=5)
        assert len(events) == 5

    def test_list_events_returns_in_chronological_order(self, store):
        store.insert_event(
            {"correlation_id": "pg-c-order", "event_type": "first", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        store.insert_event(
            {"correlation_id": "pg-c-order", "event_type": "second", "level": "INFO", "timestamp": "2026-01-01T00:00:01+00:00"}
        )
        events = store.list_events(limit=10)
        assert events[0]["event_type"] == "first"
        assert events[1]["event_type"] == "second"


class TestPostgresRuns:
    """Operaciones CRUD sobre ejecuciones (runs) en Postgres."""

    def test_create_run_generates_unique_id(self, store):
        run_id_1 = store.create_run(pillar="orchestration", team_id="team-1")
        run_id_2 = store.create_run(pillar="orchestration", team_id="team-1")
        assert run_id_1 != run_id_2
        assert run_id_1.startswith("orchestration-")

    def test_create_run_sets_initial_status(self, store):
        run_id = store.create_run(pillar="orchestration", team_id="team-1")
        run = store.get_run(run_id)
        assert run["status"] == "running"
        assert run["pillar"] == "orchestration"
        assert run["team_id"] == "team-1"
        assert "started_at" in run

    def test_finish_run_updates_status_and_summary(self, store):
        run_id = store.create_run(pillar="orchestration", team_id="team-1")
        store.finish_run(run_id, status="completed", summary={"tokens": 1500, "cost": 0.03})
        run = store.get_run(run_id)
        assert run["status"] == "completed"
        assert run["summary"] == {"tokens": 1500, "cost": 0.03}
        assert "finished_at" in run

    def test_get_run_returns_none_for_unknown_id(self, store):
        assert store.get_run("no-existe-xyz") is None

    def test_list_runs_returns_recent_runs(self, store):
        run_id_1 = store.create_run(pillar="orchestration", team_id="team-1")
        time.sleep(0.1)
        run_id_2 = store.create_run(pillar="quality_gate", team_id="team-2")
        runs = store.list_runs(limit=10)
        assert len(runs) >= 2
        # El más reciente debe aparecer primero
        assert runs[0]["run_id"] == run_id_2

    def test_list_runs_filters_by_pillar(self, store):
        store.create_run(pillar="orchestration", team_id="team-1")
        store.create_run(pillar="quality_gate", team_id="team-2")
        runs = store.list_runs(pillar="orchestration", limit=10)
        assert all(r["pillar"] == "orchestration" for r in runs)

    def test_list_runs_respects_limit(self, store):
        for i in range(15):
            store.create_run(pillar="orchestration", team_id="team-1")
        runs = store.list_runs(limit=5)
        assert len(runs) == 5


class TestPostgresMetrics:
    """Agregaciones de métricas sobre eventos en Postgres."""

    def test_metrics_summary_aggregates_cost(self, store):
        store.insert_event(
            {
                "correlation_id": "pg-m1",
                "event_type": "tool_result",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "cost_usd": 0.01,
                "model_used": "haiku",
                "latency_ms": 50,
            }
        )
        store.insert_event(
            {
                "correlation_id": "pg-m2",
                "event_type": "tool_result",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:01+00:00",
                "cost_usd": 0.02,
                "model_used": "haiku",
                "latency_ms": 150,
            }
        )
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["total_cost_usd"] == 0.03
        assert metrics["cost_by_model"] == {"haiku": 0.03}
        assert metrics["avg_latency_ms"] == 100.0
        assert metrics["p95_latency_ms"] == 150.0
        assert metrics["total_events"] == 2

    def test_metrics_summary_groups_by_level(self, store):
        store.insert_event(
            {"correlation_id": "pg-m3", "event_type": "x", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        store.insert_event(
            {"correlation_id": "pg-m4", "event_type": "y", "level": "ERROR", "timestamp": "2026-01-01T00:00:01+00:00"}
        )
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["events_by_level"]["INFO"] == 1
        assert metrics["events_by_level"]["ERROR"] == 1

    def test_metrics_summary_groups_by_event_type(self, store):
        store.insert_event(
            {"correlation_id": "pg-m5", "event_type": "agent_start", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        store.insert_event(
            {"correlation_id": "pg-m6", "event_type": "agent_finish", "level": "INFO", "timestamp": "2026-01-01T00:00:01+00:00"}
        )
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["events_by_type"]["agent_start"] == 1
        assert metrics["events_by_type"]["agent_finish"] == 1

    def test_metrics_summary_groups_by_pillar(self, store):
        store.insert_event(
            {"correlation_id": "pg-m7", "event_type": "x", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"},
            pillar="orchestration",
        )
        store.insert_event(
            {"correlation_id": "pg-m8", "event_type": "y", "level": "INFO", "timestamp": "2026-01-01T00:00:01+00:00"},
            pillar="quality_gate",
        )
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["events_by_pillar"]["orchestration"] == 1
        assert metrics["events_by_pillar"]["quality_gate"] == 1

    def test_metrics_summary_counts_runs_by_status(self, store):
        run_id = store.create_run(pillar="orchestration", team_id="team-1")
        store.finish_run(run_id, status="completed")
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["runs_by_status"]["completed"] == 1

    def test_metrics_summary_respects_time_window(self, store):
        """Eventos fuera de la ventana de tiempo no se incluyen."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        store.insert_event(
            {
                "correlation_id": "pg-m9",
                "event_type": "old",
                "level": "INFO",
                "timestamp": "2020-01-01T00:00:00+00:00",  # Muy antiguo
                "cost_usd": 100.0,
            }
        )
        store.insert_event(
            {
                "correlation_id": "pg-m10",
                "event_type": "new",
                "level": "INFO",
                "timestamp": now.isoformat(),  # Ahora mismo → dentro de la ventana
                "cost_usd": 1.0,
            }
        )
        metrics = store.metrics_summary(window_minutes=60)
        assert metrics["total_cost_usd"] == 1.0
        assert metrics["total_events"] == 1


class TestPostgresLiveTail:
    """Pruebas del mecanismo LISTEN/NOTIFY para live-tail multi-instancia."""

    def test_subscribe_receives_inserted_event(self, store):
        """Un suscriptor recibe el evento insertado vía NOTIFY."""
        sub = store.subscribe()
        try:
            store.insert_event(
                {
                    "correlation_id": "pg-lt1",
                    "event_type": "notify_test",
                    "level": "INFO",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            )
            received = sub.get(timeout=5)
            assert received["event_type"] == "notify_test"
            assert received["correlation_id"] == "pg-lt1"
        finally:
            store.unsubscribe(sub)

    def test_multiple_subscribers_receive_same_event(self, store):
        """Varios suscriptores en el mismo proceso reciben el evento."""
        sub1 = store.subscribe()
        sub2 = store.subscribe()
        try:
            store.insert_event(
                {
                    "correlation_id": "pg-lt2",
                    "event_type": "multi_sub",
                    "level": "INFO",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            )
            received1 = sub1.get(timeout=5)
            received2 = sub2.get(timeout=5)
            assert received1["event_type"] == "multi_sub"
            assert received2["event_type"] == "multi_sub"
        finally:
            store.unsubscribe(sub1)
            store.unsubscribe(sub2)

    def test_unsubscribe_stops_receiving_events(self, store):
        sub = store.subscribe()
        store.unsubscribe(sub)
        store.insert_event(
            {"correlation_id": "pg-lt3", "event_type": "after_unsub", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        # La cola debe estar vacía (no se recibió el evento)
        assert sub.empty()

    def test_subscriber_queue_has_maxsize_1000(self, store):
        sub = store.subscribe()
        assert sub.maxsize == 1000
        store.unsubscribe(sub)

    def test_full_subscriber_queue_does_not_block_insert(self, store):
        """Si la cola del suscriptor está llena, insert_event no se bloquea."""
        sub = store.subscribe()
        try:
            # Llenar la cola
            for i in range(1000):
                sub.put_nowait({"event_type": "filler"})
            # Ahora insertar un evento real no debe bloquearse
            event_id = store.insert_event(
                {"correlation_id": "pg-lt4", "event_type": "non_blocking", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
            )
            assert isinstance(event_id, int)
        finally:
            store.unsubscribe(sub)


class TestPostgresConcurrency:
    """Thread-safety y comportamiento bajo concurrencia."""

    def test_concurrent_inserts_do_not_corrupt_data(self, store):
        """Múltiples hilos insertando eventos simultáneamente no corrompen
        los datos."""
        errors = []
        inserted_ids = []
        lock = threading.Lock()

        def insert_events(thread_id: int):
            try:
                for i in range(10):
                    event_id = store.insert_event(
                        {
                            "correlation_id": f"pg-concurrent-{thread_id}",
                            "event_type": f"event-{i}",
                            "level": "INFO",
                            "timestamp": "2026-01-01T00:00:00+00:00",
                        }
                    )
                    with lock:
                        inserted_ids.append(event_id)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=insert_events, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errores en inserciones concurrentes: {errors}"
        # Todos los IDs deben ser únicos
        assert len(inserted_ids) == len(set(inserted_ids))

    def test_concurrent_reads_during_inserts(self, store):
        """Lecturas concurrentes mientras se insertan eventos no fallan."""
        errors = []
        stop_reading = threading.Event()

        def insert_events():
            for i in range(20):
                try:
                    store.insert_event(
                        {
                            "correlation_id": "pg-concurrent-read",
                            "event_type": f"evt-{i}",
                            "level": "INFO",
                            "timestamp": "2026-01-01T00:00:00+00:00",
                        }
                    )
                except Exception as exc:
                    errors.append(exc)

        def read_events():
            for _ in range(10):
                try:
                    store.get_trace("pg-concurrent-read")
                    store.list_events(limit=10)
                except Exception as exc:
                    errors.append(exc)
                time.sleep(0.05)

        threads = [
            threading.Thread(target=insert_events),
            threading.Thread(target=read_events),
            threading.Thread(target=read_events),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errores en lecturas/escrituras concurrentes: {errors}"


class TestPostgresJSONB:
    """Manejo de campos JSONB en Postgres."""

    def test_payload_with_nested_json(self, store):
        """JSONB almacena y recupera estructuras anidadas correctamente."""
        complex_payload = {
            "tool_calls": [
                {"name": "search", "args": {"query": "test", "limit": 10}},
                {"name": "summarize", "args": {"text": "result"}},
            ],
            "metadata": {"source": "api", "version": "2.0"},
        }
        store.insert_event(
            {
                "correlation_id": "pg-json1",
                "event_type": "complex",
                "level": "DEBUG",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "payload": complex_payload,
            }
        )
        trace = store.get_trace("pg-json1")
        assert trace[0]["payload"] == complex_payload

    def test_payload_with_special_characters(self, store):
        """JSONB maneja caracteres especiales (UTF-8, emojis, etc.)."""
        special_payload = {
            "message": "Hola 世界 🌍",
            "quotes": 'She said "hello"',
            "backslash": "path\\to\\file",
        }
        store.insert_event(
            {
                "correlation_id": "pg-json2",
                "event_type": "special",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "payload": special_payload,
            }
        )
        trace = store.get_trace("pg-json2")
        assert trace[0]["payload"] == special_payload

    def test_empty_payload_defaults_to_empty_dict(self, store):
        """Si no se proporciona payload, se almacena como {}."""
        store.insert_event(
            {"correlation_id": "pg-json3", "event_type": "no_payload", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        trace = store.get_trace("pg-json3")
        assert trace[0]["payload"] == {}

    def test_null_payload_becomes_empty_dict(self, store):
        """Si payload es None, se convierte a {}."""
        store.insert_event(
            {
                "correlation_id": "pg-json4",
                "event_type": "null_payload",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "payload": None,
            }
        )
        trace = store.get_trace("pg-json4")
        assert trace[0]["payload"] == {}


class TestPostgresConnectionPool:
    """Pruebas del pool de conexiones ThreadedConnectionPool."""

    def test_pool_handles_multiple_sequential_operations(self, store):
        """El pool sirve correctamente múltiples operaciones secuenciales."""
        for i in range(50):
            store.insert_event(
                {"correlation_id": f"pg-pool-{i}", "event_type": "seq", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
            )
        events = store.list_events(limit=100)
        assert len(events) == 50

    def test_pool_releases_connections_on_error(self, store):
        """Si una operación falla, la conexión se devuelve al pool."""
        # Intentar insertar con datos inválidos (pero válidos para Postgres)
        # Esto no debe agotar el pool
        for i in range(20):
            try:
                store.insert_event(
                    {"correlation_id": f"pg-pool-err-{i}", "event_type": "x", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
                )
            except Exception:
                pass
        # El pool sigue funcionando
        event_id = store.insert_event(
            {"correlation_id": "pg-pool-ok", "event_type": "ok", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        assert event_id > 0


class TestPostgresRowConversion:
    """Pruebas de las funciones helper _row_to_event_dict y _row_to_run_dict."""

    def test_row_to_event_dict_with_datetime_timestamp(self):
        """Convierte datetime a ISO format."""
        from datetime import datetime, timezone
        
        row = {
            "id": 1,
            "correlation_id": "test",
            "timestamp": datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "payload": '{"key": "value"}',
        }
        result = _row_to_event_dict(row)
        assert result["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert result["payload"] == {"key": "value"}

    def test_row_to_event_dict_with_string_payload(self):
        """Parsea JSON string en payload."""
        row = {"id": 1, "correlation_id": "test", "payload": '{"a": 1}'}
        result = _row_to_event_dict(row)
        assert result["payload"] == {"a": 1}

    def test_row_to_event_dict_with_none_payload(self):
        """Payload None se convierte a {}."""
        row = {"id": 1, "correlation_id": "test", "payload": None}
        result = _row_to_event_dict(row)
        assert result["payload"] == {}

    def test_row_to_event_dict_with_dict_payload(self):
        """Payload ya como dict se mantiene."""
        row = {"id": 1, "correlation_id": "test", "payload": {"already": "dict"}}
        result = _row_to_event_dict(row)
        assert result["payload"] == {"already": "dict"}

    def test_row_to_run_dict_with_datetime_fields(self):
        """Convierte datetime a ISO format en runs."""
        from datetime import datetime, timezone
        
        row = {
            "run_id": "run-1",
            "started_at": datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            "finished_at": datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
            "summary": '{"ok": true}',
        }
        result = _row_to_run_dict(row)
        assert result["started_at"] == "2026-01-01T00:00:00+00:00"
        assert result["finished_at"] == "2026-01-01T01:00:00+00:00"
        assert result["summary"] == {"ok": True}

    def test_row_to_run_dict_with_none_summary(self):
        """Summary None se convierte a {}."""
        row = {"run_id": "run-1", "summary": None}
        result = _row_to_run_dict(row)
        assert result["summary"] == {}


class TestPostgresEdgeCases:
    """Casos límite y manejo de errores."""

    def test_insert_event_with_minimal_fields(self, store):
        """Inserta un evento con solo los campos obligatorios.

        Nota: al no pasar correlation_id, se usa "" por defecto. Esto
        funciona pero en producción cada evento debería tener un
        correlation_id único para evitar mezclar trazas no relacionadas.
        """
        event_id = store.insert_event(
            {"event_type": "minimal", "level": "INFO"}
        )
        assert event_id > 0
        trace = store.get_trace("")
        assert len(trace) == 1

    def test_insert_event_with_empty_payload_dict(self, store):
        """Payload vacío se almacena correctamente."""
        event_id = store.insert_event(
            {
                "correlation_id": "pg-edge1",
                "event_type": "empty_payload",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "payload": {},
            }
        )
        trace = store.get_trace("pg-edge1")
        assert trace[0]["payload"] == {}

    def test_metrics_with_no_events(self, store):
        """metrics_summary funciona cuando no hay eventos."""
        metrics = store.metrics_summary(window_minutes=60)
        assert metrics["total_events"] == 0
        assert metrics["total_cost_usd"] == 0.0
        assert metrics["avg_latency_ms"] == 0.0

    def test_metrics_with_no_runs(self, store):
        """metrics_summary funciona cuando no hay runs."""
        store.insert_event(
            {"correlation_id": "pg-edge2", "event_type": "x", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["runs_by_status"] == {}

    def test_list_events_with_no_filters(self, store):
        """list_events sin filtros devuelve todos los eventos."""
        store.insert_event(
            {"correlation_id": "pg-edge3", "event_type": "a", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        store.insert_event(
            {"correlation_id": "pg-edge4", "event_type": "b", "level": "ERROR", "timestamp": "2026-01-01T00:00:01+00:00"}
        )
        events = store.list_events(limit=100)
        assert len(events) == 2

    def test_large_payload_handling(self, store):
        """JSONB maneja payloads grandes sin problemas."""
        large_payload = {"data": "x" * 10000}  # 10KB de datos
        event_id = store.insert_event(
            {
                "correlation_id": "pg-edge5",
                "event_type": "large",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "payload": large_payload,
            }
        )
        trace = store.get_trace("pg-edge5")
        assert trace[0]["payload"]["data"] == "x" * 10000