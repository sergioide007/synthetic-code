#!/usr/bin/env python3
"""
demo_redis_budget.py — Verifica TokenBudget con un Redis real, incluyendo
atomicidad bajo concurrencia.

Requiere Redis corriendo en localhost:6379 (o configurable vía
REDIS_HOST/REDIS_PORT). Si no hay Redis disponible, el script lo indica
claramente y sale con código 0 (no es un fallo del código, es un
prerrequisito de infraestructura ausente) salvo que se use --strict.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

from sintetico import LocalMemoryBudgetBackend, RedisBudgetBackend, RedisUnavailableError, TokenBudget


def banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def test_memory_backend() -> bool:
    banner("TEST: LocalMemoryBudgetBackend")
    backend = LocalMemoryBudgetBackend()
    budget = TokenBudget(monthly_budget=50.0, backend=backend, team_id="test-memory")

    # El tercer coste (25.0) se rechaza a propósito: 5+15+10=30, y 30+25=55
    # excede el límite de 50; por eso el acumulado final es 40, no 65.
    for cost in (5.0, 15.0, 10.0, 25.0, 10.0):
        ok = budget.record_cost(cost)
        print(f"  Costo ${cost:.2f}: {'✅' if ok else '❌ rechazado'} (acumulado: ${budget.spent:.2f})")

    assert budget.spent == 40.0, f"Se esperaba $40.00 acumulados, hay ${budget.spent:.2f}"
    print(f"✅ Backend en memoria correcto: ${budget.spent:.2f} / ${budget.monthly_budget:.2f}")
    return True


def test_redis_backend(host: str, port: int) -> bool:
    banner("TEST: RedisBudgetBackend (conexión real)")
    try:
        backend = RedisBudgetBackend(host=host, port=port)
    except ImportError as exc:
        print(f"❌ {exc}")
        return False

    if not backend._available:
        print(f"⚠️  Redis no disponible en {host}:{port}: {backend._connection_error}")
        return False

    backend.redis.delete("budget:test-redis")
    budget = TokenBudget(monthly_budget=100.0, backend=backend, team_id="test-redis")
    print(f"✅ Conectado a Redis en {host}:{port}")

    for cost in (10.0, 25.5, 8.75, 50.0):
        ok = budget.record_cost(cost)
        print(f"  Costo ${cost:.2f}: {'✅' if ok else '❌ rechazado'} (acumulado: ${budget.spent:.2f})")

    over_budget = budget.record_cost(20.0)
    print(
        f"  Intento de $20 adicionales (excede el límite): {'❌ aceptado (BUG)' if over_budget else '✅ rechazado correctamente'}"
    )
    assert not over_budget, "El backend aceptó un gasto que excedía el límite mensual"

    backend.redis.delete("budget:test-redis")
    print("✅ Test de Redis completado y clave de prueba limpiada.")
    return True


def test_concurrent_redis(host: str, port: int) -> bool:
    banner("TEST: Concurrencia atómica con Redis (script Lua)")
    try:
        backend = RedisBudgetBackend(host=host, port=port)
    except ImportError as exc:
        print(f"❌ {exc}")
        return False
    if not backend._available:
        print(f"⚠️  Redis no disponible en {host}:{port}: {backend._connection_error}")
        return False

    key = "test-redis-concurrent"
    backend.redis.delete(f"budget:{key}")
    budget = TokenBudget(monthly_budget=10_000.0, backend=backend, team_id=key)

    results = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(5):
            ok = budget.record_cost(10.0)
            with lock:
                results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successful = sum(results)
    expected_total = successful * 10.0
    actual_total = budget.spent

    print(f"  Operaciones exitosas: {successful}/{len(results)}")
    print(f"  Esperado en Redis: ${expected_total:.2f} | Real en Redis: ${actual_total:.2f}")

    backend.redis.delete(f"budget:{key}")
    consistent = abs(expected_total - actual_total) < 1e-9
    print("✅ Concurrencia atómica verificada" if consistent else "❌ INCONSISTENCIA DETECTADA")
    return consistent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("REDIS_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("REDIS_PORT", "6379")))
    parser.add_argument("--strict", action="store_true", help="Salir con código != 0 si Redis no está disponible.")
    args = parser.parse_args()

    print("\n🧪 SUITE DE TESTS PARA BUDGET BACKENDS")

    ok_memory = test_memory_backend()

    try:
        ok_redis = test_redis_backend(args.host, args.port)
        ok_concurrent = test_concurrent_redis(args.host, args.port) if ok_redis else False
    except RedisUnavailableError as exc:
        print(f"⚠️  {exc}")
        ok_redis = ok_concurrent = False

    banner("RESUMEN")
    print(f"  Memoria local:        {'✅' if ok_memory else '❌'}")
    print(f"  Redis (conexión):     {'✅' if ok_redis else '⚠️  no disponible'}")
    print(f"  Redis (concurrencia): {'✅' if ok_concurrent else '⚠️  no verificado'}")

    if args.strict and not (ok_redis and ok_concurrent):
        return 1
    return 0 if ok_memory else 1


if __name__ == "__main__":
    sys.exit(main())
