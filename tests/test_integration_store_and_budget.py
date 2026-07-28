"""Tests de integración que verifican el funcionamiento conjunto de
PostgresTraceStore y RedisBudgetBackend en un escenario multi-instancia.

Estos tests simulan dos instancias del servicio compartiendo el mismo
Postgres (trazas) y el mismo Redis (presupuesto), y verifican que:

1. El live-tail multi-instancia (LISTEN/NOTIFY) funciona.
2. El control de presupuesto distribuido (Redis Lua script) es atómico.
3. Ambas operaciones pueden ocurrir concurrentemente sin interferencias.

Requiere:
- `psycopg2` instalado y `SINTETICO_TEST_POSTGRES_URL`
- `redis` instalado y Redis accesible (por defecto redis://localhost:6379/0)
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from sintetico.budget import RedisBudgetBackend, TokenBudget
from sintetico_api.postgres_store import PostgresTraceStore


def _cleanup_postgres(store):
    """Limpia las tablas entre tests."""
    with store._cursor(commit=True) as cur:
        cur.execute("TRUNCATE sintetico_events, sintetico_runs;")


def _make_store() -> PostgresTraceStore:
    psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 no instalado")
    dsn = os.environ.get("SINTETICO_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("Define SINTETICO_TEST_POSTGRES_URL para correr tests de Postgres")
    try:
        store = PostgresTraceStore(dsn)
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Postgres no accesible: {exc}")
    _cleanup_postgres(store)
    return store


def _make_redis_backend() -> RedisBudgetBackend:
    pytest.importorskip("redis", reason="redis-py no instalado")
    from sintetico.budget import RedisUnavailableError
    from urllib.parse import urlparse

    url = os.environ.get("SINTETICO_TEST_REDIS_URL", "redis://localhost:6379/0")
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    try:
        backend = RedisBudgetBackend(host=host, port=port, db=2, socket_timeout=3.0)
    except RedisUnavailableError as exc:
        pytest.skip(f"Redis no accesible: {exc}")
    # Limpiar datos previos
    if backend._available:
        for key in backend.redis.keys("budget:*"):
            backend.redis.delete(key)
    return backend


class TestIntegrationStoreAndBudget:
    """Escenarios de integración entre PostgresTraceStore y RedisBudgetBackend."""

    def test_insert_event_and_check_budget(self):
        """Insertar un evento en Postgres y verificar presupuesto en Redis
        son operaciones independientes que no interfieren."""
        store = _make_store()
        backend = _make_redis_backend()
        try:
            budget = TokenBudget(monthly_budget=100.0, backend=backend, team_id="integration-team")

            # Registrar costo
            assert budget.record_cost(0.05) is True

            # Insertar evento asociado
            event_id = store.insert_event(
                {
                    "correlation_id": "integration-c1",
                    "event_type": "llm_call",
                    "level": "INFO",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "cost_usd": 0.05,
                    "model_used": "haiku",
                }
            )
            assert event_id > 0

            # Verificar que ambos estados son consistentes
            trace = store.get_trace("integration-c1")
            assert len(trace) == 1
            assert trace[0]["cost_usd"] == 0.05
            assert budget.spent == 0.05
        finally:
            store.close()
            backend.redis.close()

    def test_live_tail_cross_instance_with_budget(self):
        """Simula dos instancias: una inserta un evento (y gasta presupuesto),
        la otra recibe el evento vía LISTEN/NOTIFY."""
        store_a = _make_store()
        store_b = _make_store()  # segunda instancia, mismo Postgres
        backend = _make_redis_backend()
        try:
            budget = TokenBudget(monthly_budget=100.0, backend=backend, team_id="integration-live")

            # store_b se suscribe (simula un cliente SSE conectado a instancia B)
            sub = store_b.subscribe()
            try:
                # store_a inserta un evento (simula un agente en instancia A)
                assert budget.record_cost(0.10) is True
                store_a.insert_event(
                    {
                        "correlation_id": "integration-c2",
                        "event_type": "cross_instance",
                        "level": "INFO",
                        "timestamp": "2026-01-01T00:00:00+00:00",
                        "cost_usd": 0.10,
                    }
                )

                # store_b debe recibir el evento vía NOTIFY
                received = sub.get(timeout=5)
                assert received["event_type"] == "cross_instance"
                assert received["correlation_id"] == "integration-c2"
            finally:
                store_b.unsubscribe(sub)
        finally:
            store_a.close()
            store_b.close()
            backend.redis.close()

    def test_concurrent_inserts_and_budget_checks(self):
        """Múltiples hilos insertando eventos y verificando presupuesto
        concurrentemente no corrompen ni Postgres ni Redis."""
        store = _make_store()
        backend = _make_redis_backend()
        try:
            budget = TokenBudget(monthly_budget=50.0, backend=backend, team_id="integration-concurrent")
            errors = []

            def worker(thread_id: int):
                try:
                    for i in range(10):
                        # Verificar presupuesto antes de insertar
                        if budget.record_cost(1.0):
                            store.insert_event(
                                {
                                    "correlation_id": f"integration-con-{thread_id}-{i}",
                                    "event_type": "concurrent",
                                    "level": "INFO",
                                    "timestamp": "2026-01-01T00:00:00+00:00",
                                }
                            )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            assert not errors, f"Errores en operaciones concurrentes: {errors}"
            # Exactamente 50 inserciones exitosas (presupuesto de 50.0)
            assert budget.spent == 50.0
            all_events = store.list_events(limit=100)
            assert len(all_events) == 50
        finally:
            store.close()
            backend.redis.close()

    def test_budget_rejection_does_not_affect_traces(self):
        """Cuando el presupuesto se rechaza, no se inserta ningún evento."""
        store = _make_store()
        backend = _make_redis_backend()
        try:
            budget = TokenBudget(monthly_budget=5.0, backend=backend, team_id="integration-reject")

            # Primer gasto: cabe en el presupuesto
            assert budget.record_cost(5.0) is True
            event_id = store.insert_event(
                {
                    "correlation_id": "integration-reject-1",
                    "event_type": "accepted",
                    "level": "INFO",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            )
            assert event_id > 0

            # Segundo gasto: excede el presupuesto
            assert budget.record_cost(1.0) is False

            # No debe haber un segundo evento
            all_events = store.list_events(limit=10)
            assert len(all_events) == 1
        finally:
            store.close()
            backend.redis.close()

    def test_multiple_teams_shared_store_isolated_budgets(self):
        """Múltiples equipos comparten el mismo PostgresTraceStore pero
        tienen presupuestos aislados en Redis."""
        store = _make_store()
        backend = _make_redis_backend()
        try:
            budget_team_a = TokenBudget(monthly_budget=10.0, backend=backend, team_id="integration-team-a")
            budget_team_b = TokenBudget(monthly_budget=10.0, backend=backend, team_id="integration-team-b")

            # Team A gasta
            budget_team_a.record_cost(10.0)
            store.insert_event(
                {
                    "correlation_id": "integration-team-a-1",
                    "event_type": "team_a_op",
                    "level": "INFO",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
                pillar="orchestration",
            )

            # Team B todavía tiene presupuesto completo
            assert budget_team_b.spent == 0.0
            assert budget_team_b.record_cost(10.0) is True
            store.insert_event(
                {
                    "correlation_id": "integration-team-b-1",
                    "event_type": "team_b_op",
                    "level": "INFO",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                },
                pillar="quality_gate",
            )

            # Team A ya no puede gastar más
            assert budget_team_a.record_cost(1.0) is False

            # Ambos eventos están en Postgres
            events_a = store.list_events(pillar="orchestration", limit=10)
            events_b = store.list_events(pillar="quality_gate", limit=10)
            assert len(events_a) == 1
            assert len(events_b) == 1
        finally:
            store.close()
            backend.redis.close()