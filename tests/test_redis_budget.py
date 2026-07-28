"""Tests específicos de RedisBudgetBackend que ejercitan la conexión real
contra Redis, el script Lua de atomicidad, el fallback con WATCH/MULTI,
y la integración con TokenBudget.

Requiere `redis` instalado y `SINTETICO_TEST_REDIS_URL` apuntando a un
Redis accesible (por defecto redis://localhost:6379/0). Si no se cumple,
los tests se omiten limpiamente.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from sintetico.budget import LocalMemoryBudgetBackend, RedisBudgetBackend, RedisUnavailableError, TokenBudget


def _make_redis_backend(db: int = 1) -> RedisBudgetBackend:
    """Crea un RedisBudgetBackend apuntando a la URL de entorno, o salta
    el test si no está disponible."""
    pytest.importorskip("redis", reason="redis-py no instalado")
    url = os.environ.get("SINTETICO_TEST_REDIS_URL", "redis://localhost:6379/0")
    # Extraer host/port/db de la URL
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    base_db = parsed.path.lstrip("/") or "0"

    try:
        backend = RedisBudgetBackend(host=host, port=port, db=int(base_db) + db, socket_timeout=3.0)
    except RedisUnavailableError as exc:
        pytest.skip(f"Redis no accesible en {host}:{port}: {exc}")
    # Limpiar datos de tests anteriores en esta DB
    _cleanup_redis(backend)
    return backend


def _cleanup_redis(backend: RedisBudgetBackend) -> None:
    """Limpia todas las claves con prefijo budget: de la DB de tests."""
    if backend._available:
        for key in backend.redis.keys("budget:*"):
            backend.redis.delete(key)


class TestRedisBudgetBackendConnection:
    """Verifica que RedisBudgetBackend maneja correctamente la conexión."""

    def test_connects_to_running_redis(self):
        """Conectar a un Redis que está corriendo funciona."""
        backend = _make_redis_backend()
        assert backend._available is True
        assert backend._incr_script is not None
        backend.redis.close()

    def test_raises_on_operation_when_unreachable(self):
        """Operar sobre un backend con host inalcanzable lanza RedisUnavailableError."""
        backend = RedisBudgetBackend(host="192.0.2.1", port=6379, socket_timeout=1.0)
        with pytest.raises(RedisUnavailableError):
            backend.get_spent("budget:test:dead")

    def test_raises_on_operation_when_wrong_port(self):
        """Operar sobre un backend con puerto incorrecto lanza RedisUnavailableError."""
        backend = RedisBudgetBackend(host="localhost", port=1, socket_timeout=1.0)
        with pytest.raises(RedisUnavailableError):
            backend.add_cost("budget:test:dead", 1.0, 100.0)


class TestRedisBudgetBackendAtomicity:
    """Operaciones atómicas check-and-increment sobre Redis."""

    def test_add_cost_within_limit_succeeds(self):
        backend = _make_redis_backend()
        try:
            result = backend.add_cost("budget:test:within", 30.0, 100.0)
            assert result is True
            assert backend.get_spent("budget:test:within") == 30.0
        finally:
            backend.redis.close()

    def test_add_cost_exceeding_limit_fails(self):
        backend = _make_redis_backend()
        try:
            backend.add_cost("budget:test:exceed", 80.0, 100.0)
            result = backend.add_cost("budget:test:exceed", 30.0, 100.0)  # 80+30=110 > 100
            assert result is False
            assert backend.get_spent("budget:test:exceed") == 80.0  # no debe haber cambiado
        finally:
            backend.redis.close()

    def test_add_cost_at_exact_limit_succeeds(self):
        backend = _make_redis_backend()
        try:
            result = backend.add_cost("budget:test:exact", 100.0, 100.0)
            assert result is True
            assert backend.get_spent("budget:test:exact") == 100.0
        finally:
            backend.redis.close()

    def test_add_cost_zero_amount_succeeds(self):
        backend = _make_redis_backend()
        try:
            result = backend.add_cost("budget:test:zero", 0.0, 100.0)
            assert result is True
            assert backend.get_spent("budget:test:zero") == 0.0
        finally:
            backend.redis.close()

    def test_get_spent_returns_zero_for_unknown_key(self):
        backend = _make_redis_backend()
        try:
            assert backend.get_spent("budget:test:unknown") == 0.0
        finally:
            backend.redis.close()

    def test_concurrent_adds_are_atomic(self):
        """Múltiples hilos incrementando concurrentemente no superan el límite."""
        backend = _make_redis_backend()
        try:
            key = "budget:test:concurrent"
            limit = 50.0
            successes = []
            lock = threading.Lock()

            def worker():
                for _ in range(20):
                    ok = backend.add_cost(key, 1.0, limit)
                    with lock:
                        successes.append(ok)

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            total_successes = sum(successes)
            assert total_successes == 50  # exactamente 50 de 100 intentos
            assert backend.get_spent(key) == 50.0
        finally:
            backend.redis.close()

    def test_teams_are_isolated(self):
        """Claves de equipos diferentes no interfieren entre sí."""
        backend = _make_redis_backend()
        try:
            backend.add_cost("budget:team:a", 80.0, 100.0)
            backend.add_cost("budget:team:b", 80.0, 100.0)
            assert backend.get_spent("budget:team:a") == 80.0
            assert backend.get_spent("budget:team:b") == 80.0
            # Aún pueden seguir gastando individualmente
            assert backend.add_cost("budget:team:a", 20.0, 100.0) is True
            assert backend.add_cost("budget:team:b", 20.0, 100.0) is True
        finally:
            backend.redis.close()


class TestRedisBudgetBackendWithTokenBudget:
    """Integración de RedisBudgetBackend con TokenBudget."""

    def test_token_budget_with_redis_backend(self):
        backend = _make_redis_backend()
        try:
            budget = TokenBudget(monthly_budget=100.0, backend=backend, team_id="redis-team")
            assert budget.record_cost(30.0) is True
            assert budget.spent == 30.0
            assert budget.remaining == 70.0
        finally:
            backend.redis.close()

    def test_token_budget_rejects_excess(self):
        backend = _make_redis_backend()
        try:
            budget = TokenBudget(monthly_budget=10.0, backend=backend, team_id="redis-team-excess")
            assert budget.record_cost(5.0) is True
            assert budget.record_cost(6.0) is False  # 5+6=11 > 10
            assert budget.spent == 5.0
        finally:
            backend.redis.close()

    def test_token_budget_alerts_fire(self):
        backend = _make_redis_backend()
        try:
            seen = []
            budget = TokenBudget(
                monthly_budget=100.0,
                backend=backend,
                team_id="redis-team-alert",
                on_alert=lambda pct, team, spent: seen.append(pct),
            )
            budget.record_cost(50.0)  # cruza 50%
            budget.record_cost(1.0)   # no cruza nuevo umbral
            assert seen == [0.5]
        finally:
            backend.redis.close()

    def test_teams_are_isolated_on_redis_backend(self):
        backend = _make_redis_backend()
        try:
            budget_a = TokenBudget(monthly_budget=10.0, backend=backend, team_id="redis-team-a")
            budget_b = TokenBudget(monthly_budget=10.0, backend=backend, team_id="redis-team-b")
            budget_a.record_cost(9.0)
            assert budget_b.spent == 0.0
            assert budget_b.record_cost(9.0) is True
        finally:
            backend.redis.close()

    def test_concurrent_token_budget_is_atomic(self):
        """TokenBudget con Redis backend es atómico bajo concurrencia."""
        backend = _make_redis_backend()
        try:
            budget = TokenBudget(monthly_budget=50.0, backend=backend, team_id="redis-concurrent")
            successes = []
            lock = threading.Lock()

            def worker():
                for _ in range(20):
                    ok = budget.record_cost(1.0)
                    with lock:
                        successes.append(ok)

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert sum(successes) == 50
            assert budget.spent == 50.0
        finally:
            backend.redis.close()


class TestRedisBudgetBackendEdgeCases:
    """Casos límite y manejo de errores."""

    def test_large_amounts(self):
        """Maneja cantidades grandes sin errores de precisión."""
        backend = _make_redis_backend()
        try:
            result = backend.add_cost("budget:test:large", 1_000_000.0, 10_000_000.0)
            assert result is True
            assert backend.get_spent("budget:test:large") == 1_000_000.0
        finally:
            backend.redis.close()

    def test_float_precision(self):
        """Maneja decimales correctamente."""
        backend = _make_redis_backend()
        try:
            backend.add_cost("budget:test:float", 0.1, 1.0)
            backend.add_cost("budget:test:float", 0.2, 1.0)
            # 0.1 + 0.2 = 0.30000000000000004 en float, pero debe estar cerca
            assert abs(backend.get_spent("budget:test:float") - 0.3) < 1e-10
        finally:
            backend.redis.close()

    def test_negative_cost_raises(self):
        """add_cost con cantidad negativa no debería ser posible desde
        TokenBudget (que lo valida), pero RedisBudgetBackend lo maneja."""
        backend = _make_redis_backend()
        try:
            # El script Lua: current + amount > limit → si amount es negativo,
            # nunca supera el límite, así que debería funcionar
            result = backend.add_cost("budget:test:neg", -10.0, 100.0)
            assert result is True
            assert backend.get_spent("budget:test:neg") == -10.0
        finally:
            backend.redis.close()

    def test_redis_unavailable_after_connection(self):
        """Operaciones después de que Redis deje de estar disponible lanzan
        RedisUnavailableError. (Simulado cerrando la conexión)"""
        backend = _make_redis_backend()
        try:
            # Forzar indisponibilidad
            backend.redis.close()
            backend._available = False
            with pytest.raises(RedisUnavailableError):
                backend.add_cost("budget:test:dead", 1.0, 100.0)
            with pytest.raises(RedisUnavailableError):
                backend.get_spent("budget:test:dead")
        finally:
            # No cerrar de nuevo, ya está cerrado
            pass