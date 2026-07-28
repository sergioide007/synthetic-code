"""Tests de contrato para BaseTraceStore: el mismo conjunto de
aserciones se ejecuta contra `TraceStore` (SQLite, siempre disponible) y
contra `PostgresTraceStore` (si `psycopg2` está instalado y
`SINTETICO_TEST_POSTGRES_URL` apunta a un Postgres real y accesible;
si no, se omite limpiamente).

Esto garantiza que ambos backends se comporten de forma idéntica desde
el punto de vista de quien los consume (los routers de la API).
"""

import os
import tempfile

import pytest

from sintetico_api.store import TraceStore


def _make_sqlite_store():
    path = tempfile.mktemp(suffix=".db")
    store = TraceStore(path)
    yield store
    store.close()
    if os.path.exists(path):
        os.remove(path)


def _make_postgres_store():
    psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 no instalado")
    dsn = os.environ.get("SINTETICO_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("Define SINTETICO_TEST_POSTGRES_URL para correr los tests de Postgres")
    from sintetico_api.postgres_store import PostgresTraceStore

    try:
        store = PostgresTraceStore(dsn)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres no accesible en SINTETICO_TEST_POSTGRES_URL: {exc}")
    yield store
    store.close()


@pytest.fixture(params=["sqlite", "postgres"])
def store(request):
    gen = _make_sqlite_store() if request.param == "sqlite" else _make_postgres_store()
    value = next(gen)
    yield value
    try:
        next(gen)
    except StopIteration:
        pass


class TestTraceStoreContract:
    def test_insert_and_retrieve_trace_in_order(self, store):
        store.insert_event(
            {
                "correlation_id": "contract-c1",
                "event_type": "a",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        )
        store.insert_event(
            {
                "correlation_id": "contract-c1",
                "event_type": "b",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:01+00:00",
            }
        )
        trace = store.get_trace("contract-c1")
        assert [e["event_type"] for e in trace] == ["a", "b"]

    def test_run_lifecycle(self, store):
        run_id = store.create_run(pillar="orchestration", team_id="contract-team")
        assert store.get_run(run_id)["status"] == "running"
        store.finish_run(run_id, status="completed", summary={"ok": True})
        run = store.get_run(run_id)
        assert run["status"] == "completed"
        assert run["summary"] == {"ok": True}

    def test_get_run_returns_none_for_unknown_id(self, store):
        assert store.get_run("no-existe") is None

    def test_list_events_filters_by_pillar(self, store):
        store.insert_event(
            {
                "correlation_id": "contract-c2",
                "event_type": "x",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            pillar="quality_gate",
        )
        store.insert_event(
            {
                "correlation_id": "contract-c3",
                "event_type": "y",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            pillar="orchestration",
        )
        events = store.list_events(pillar="quality_gate", limit=100)
        assert all(e["pillar"] == "quality_gate" for e in events)
        assert any(e["correlation_id"] == "contract-c2" for e in events)

    def test_metrics_summary_aggregates_cost(self, store):
        store.insert_event(
            {
                "correlation_id": "contract-c4",
                "event_type": "tool_result",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "cost_usd": 0.01,
                "model_used": "haiku",
                "latency_ms": 50,
            }
        )
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["total_cost_usd"] >= 0.01
        assert "haiku" in metrics["cost_by_model"]

    def test_live_tail_publishes_inserted_event(self, store):
        sub = store.subscribe()
        try:
            store.insert_event(
                {
                    "correlation_id": "contract-c5",
                    "event_type": "z",
                    "level": "INFO",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            )
            received = sub.get(timeout=5)
            assert received["event_type"] == "z"
        finally:
            store.unsubscribe(sub)
