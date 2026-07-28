#!/usr/bin/env python3
"""
demo_pilares.py — Demostración de los 4 pilares del libro (modo simulado).

No requiere API keys: usa costes por tier simulados para poder ejecutarse
en cualquier entorno (incluido CI) y como introducción rápida a los
patrones. Para ver ahorro de costes con tráfico real, ejecuta
`demos/demo_ahorro_tokens.py` con una API key configurada.
"""

from __future__ import annotations

import time

from sintetico import (
    AsyncAIGateway,
    DebtTracker,
    EmergencyOverrideGateway,
    ModelRouter,
    OverrideReason,
    SelfCleaningCodeLoop,
    SemanticCache,
    SwarmTokenOrchestrator,
    TokenBudget,
    TrustScoreCalculator,
    calculate_roas,
    compute_tco,
    get_model_config,
    resolve_model_id,
)

# Coste simulado por tier, sólo para esta demo offline (no llama a ninguna
# API). Se basa en tokens típicos observados por complejidad de consulta.
_SIMULATED_TOKENS_BY_TIER = {"haiku": 50, "sonnet": 100, "opus": 200}


def _simulated_cost(alias: str, tokens: int) -> float:
    cfg = get_model_config(alias)
    # Aproximación 70/30 input/output para esta demo offline.
    input_tokens, output_tokens = int(tokens * 0.7), int(tokens * 0.3)
    return round(
        (input_tokens / 1_000_000) * cfg.cost_per_million_input
        + (output_tokens / 1_000_000) * cfg.cost_per_million_output,
        8,
    )


def demo_orchestration_and_cost() -> dict:
    print("\n" + "=" * 70)
    print("🧠 CASO 1: ORQUESTACIÓN EFICIENTE Y AHORRO DE COSTOS")
    print("=" * 70)
    print("📌 10 consultas con enrutamiento por complejidad + caché (modo simulado)\n")

    router = ModelRouter()
    cache = SemanticCache()
    budget = TokenBudget(100.0)

    queries = [
        "Hola, necesito ayuda",
        "Error en la base de datos",
        "Revisar arquitectura de pagos",
        "Hola, necesito ayuda",
        "Error 500 en API",
        "Diseñar sistema de caché",
        "Saludo inicial",
        "Problema con login",
        "Estrategia de microservicios",
        "Hola, necesito ayuda",
    ]

    total_cost = 0.0
    total_tokens = 0
    cache_hits = 0

    print("🔹 Procesando consultas:\n")
    for i, query in enumerate(queries, 1):
        if cache.get(query):
            cache_hits += 1
            print(f"  [{i:2}] ✅ CACHÉ HIT")
            continue

        alias = router.select_model(query)
        tokens = _SIMULATED_TOKENS_BY_TIER[alias]
        cost = _simulated_cost(alias, tokens)

        if not budget.record_cost(cost):
            print(f"  [{i:2}] ❌ Presupuesto agotado")
            break

        total_cost += cost
        total_tokens += tokens
        print(f"  [{i:2}] Modelo: {resolve_model_id(alias):24} | Tokens: {tokens:3} | Costo: ${cost:.6f}")
        cache.set(query, "respuesta simulada")

    opus_cfg = get_model_config("opus")
    opus_only_cost = (
        total_tokens * (opus_cfg.cost_per_million_input * 0.7 + opus_cfg.cost_per_million_output * 0.3) / 1_000_000
    )
    savings_pct = ((opus_only_cost - total_cost) / opus_only_cost * 100) if opus_only_cost > 0 else 0.0
    tco = compute_tco(total_tokens, total_tokens // 2, 5, 50, 3)

    print("\n📈 MÉTRICAS FINALES:")
    print(f"  • Consultas: {len(queries)}  | Cache hits: {cache_hits} ({cache.hit_rate:.0%})")
    print(f"  • Tokens consumidos: {total_tokens}  | Costo total: ${total_cost:.6f}")
    print(f"  • Presupuesto restante: ${budget.remaining:.2f}")
    print(f"  • Costo estimado si todo fuera Opus: ${opus_only_cost:.6f}")
    print(f"  • Ahorro vs Opus: {savings_pct:.1f}%")
    print(f"  • TCO (con verificación humana): ${tco:.4f}")

    # Proyección de negocio: si esta mezcla de consultas se repitiera a
    # escala de producción, ¿cuánto ahorra el patrón de enrutamiento al
    # mes/año? Deliberadamente NO se inventa aquí un coste de
    # implementación fijo (dependería del sistema de cada lector); se usa
    # `calculate_roas` con implementation_cost=0 para leer sólo el ahorro
    # bruto, y se deja como ejercicio comparar esa cifra contra el coste
    # real de implementar el patrón en cada caso concreto.
    if len(queries) > 0 and total_cost > 0:
        scale_factor = 200_000 / len(queries)
        roas = calculate_roas(
            before_cost=opus_only_cost * scale_factor,
            after_cost=total_cost * scale_factor,
            implementation_cost=0,
        )
        print("\n💼 PROYECCIÓN DE NEGOCIO (extrapolando a 200,000 consultas/mes):")
        print(f"  • Ahorro bruto proyectado: ${roas['monthly_savings']:,.2f}/mes (${roas['annual_savings']:,.2f}/año)")
        print("  • Con esta mezcla de consultas y estos precios, compara esa cifra")
        print("    contra el coste real de implementar el router en tu sistema para")
        print("    decidir si compensa (usa sintetico.economics.calculate_roas con tu")
        print("    propio implementation_cost). El ahorro absoluto depende fuertemente")
        print("    del volumen y de cuánto se diferencian los precios entre modelos.")

    return {"status": "ok", "savings_pct": savings_pct}


def demo_ci_cd_autonomous_quality() -> dict:
    print("\n" + "=" * 70)
    print("🔒 CASO 2: CALIDAD AUTÓNOMA EN CI/CD")
    print("=" * 70)
    print("📌 3 PRs con distinta calidad → TrustScore y decisión de deploy\n")

    calc = TrustScoreCalculator()
    override_gw = EmergencyOverrideGateway()

    prs = [
        {
            "name": "PR #1 - Excelente",
            "pass_rate": 0.95,
            "precision": 0.92,
            "human_approval": 0.98,
            "critical_bugs": 0,
            "consistency": 0.95,
        },
        {
            "name": "PR #2 - Bugs críticos",
            "pass_rate": 0.82,
            "precision": 0.78,
            "human_approval": 0.85,
            "critical_bugs": 2,
            "consistency": 0.70,
        },
        {
            "name": "PR #3 - Baja precisión",
            "pass_rate": 0.72,
            "precision": 0.65,
            "human_approval": 0.80,
            "critical_bugs": 0,
            "consistency": 0.75,
        },
    ]

    exit_codes = []
    for pr in prs:
        res = calc.calculate(
            pr["pass_rate"], pr["precision"], pr["human_approval"], pr["critical_bugs"], pr["consistency"]
        )
        exit_codes.append(res.ci_exit_code)
        status = {0: "✅ APROBADO", 1: "❌ RECHAZADO", 2: "🚫 BLOQUEADO (crítico)"}[res.ci_exit_code]
        print(f"\n{pr['name']}")
        print(f"  TrustScore: {res.score:.3f} ({res.level.value}) | Exit code: {res.ci_exit_code} → {status}")
        print(f"  Recomendación: {res.recommendation}")

        if res.ci_exit_code == 1 and pr["name"].startswith("PR #3"):
            req_id = override_gw.request_override(
                "pipe-123",
                res.recommendation,
                OverrideReason.FALSE_POSITIVE,
                "El agente no conoce el contexto de negocio: falso positivo confirmado por tech lead",
                "DevLead",
            )
            approval = override_gw.approve_override(req_id, "CTO")
            print(f"  🛟 Emergency Override: {approval['status']} (ID: {req_id})")

    print(f"\n📜 Overrides auditados en el histórico: {len(override_gw.override_history)}")
    return {"status": "ok", "exit_codes": exit_codes}


def demo_self_cleaning_code() -> dict:
    print("\n" + "=" * 70)
    print("🧹 CASO 3: CÓDIGO AUTOLIMPIABLE")
    print("=" * 70)
    print("📌 Detección y refactorización determinista de deuda técnica\n")

    dirty_code = """def process_user(data):
    print("Processing user")
    if data.get('active'):
        print("User is active")
    else:
        print("User is inactive")

def process_admin(data):
    print("Processing admin")
    if data.get('active'):
        print("User is active")
    else:
        print("User is inactive")
    return "admin processed"
"""

    print("🔹 Código original:")
    print(f"  Líneas: {len(dirty_code.splitlines())}")
    print("  Problema detectado: 6 llamadas a print() candidatas a unificarse en un logger")

    cleaner = SelfCleaningCodeLoop()
    result = cleaner.clean(dirty_code)

    if result["status"] == "cleaned":
        len(result["code"].splitlines())
        print(f"\n  ✅ Limpieza completada en {result['iterations']} iteración(es)")
        print("  📝 print() → _log() unificado; el código refactorizado compila correctamente")
        compile(result["code"], "<refactored>", "exec")  # verificación explícita, no sólo confianza
        print("  🧪 Verificación estática post-refactor: OK")
    else:
        print(f"  ⚠️ Estado: {result['status']} — {result.get('error', '')}")

    dt = DebtTracker()
    debt = dt.add_debt("code_duplication", "medium", "Código duplicado en funciones de usuario (previo al refactor)")
    print(f"\n  🗂️ Deuda registrada: {debt.id} - {debt.description}")
    print(f"  📋 Prioridad más alta en el backlog: {dt.prioritize()[0].severity}")

    return {"status": "ok", "cleaning_status": result["status"]}


def demo_resilient_async() -> dict:
    print("\n" + "=" * 70)
    print("⚡ CASO 4: ARQUITECTURA RESILIENTE ASÍNCRONA")
    print("=" * 70)
    print("📌 Gateway asíncrono + detección de bucles en enjambre\n")

    async_gw = AsyncAIGateway("platform", 500)
    print("🔹 Enviando 5 tareas pesadas al gateway asíncrono:")
    task_ids = [async_gw.submit_task(f"heavy_task_{i}", {"data": f"dataset_{i}"}) for i in range(5)]
    for i, tid in enumerate(task_ids, 1):
        print(f"  Tarea {i}: {tid} (en cola)")

    print("\n🔹 Recuperando resultados:")
    for tid in task_ids:
        task = async_gw.wait_for_result(tid, timeout=2)
        print(f"  {tid}: {task.status.value} → {task.result or task.error}")
    async_gw.shutdown()

    print("\n🔹 Simulando enjambre de agentes con ciclo de debate:")
    swarm = SwarmTokenOrchestrator("debate_001", total_budget_tokens=400)
    messages = [
        ("AgentA", "AgentB", "Propongo solución X", 50),
        ("AgentB", "AgentA", "No, Y es mejor", 50),
        ("AgentA", "AgentB", "X es más eficiente", 50),
        ("AgentB", "AgentA", "Y es más seguro", 50),
        ("AgentA", "AgentB", "Insisto en X", 50),
        ("AgentB", "AgentA", "Y es mejor, revisa", 50),
    ]
    for from_agent, to_agent, msg, cost in messages:
        decision = swarm.add_message(from_agent, to_agent, msg, cost)
        print(f"  {from_agent}->{to_agent}: {msg[:20]:20} | {decision['decision']}")
        if decision["decision"] == "HALT":
            print(f"  🛑 Bucle detectado y detenido: {decision['reason']}")
            break

    print(f"\n📊 Tokens gastados en enjambre: {swarm.spent_tokens} | Mensajes: {len(swarm.message_history)}")
    return {"status": "ok"}


def main() -> None:
    print("\n🚀 CÓDIGO SINTÉTICO — DEMOSTRACIÓN DE PILARES (modo simulado)")
    print("Para costes reales medidos contra la API, ejecuta demo_ahorro_tokens.py\n")
    start = time.time()

    demo_orchestration_and_cost()
    demo_ci_cd_autonomous_quality()
    demo_self_cleaning_code()
    demo_resilient_async()

    print("\n" + "=" * 70)
    print("🎉 DEMOSTRACIÓN COMPLETADA")
    print("=" * 70)
    print(f"⏱️  Tiempo de ejecución: {time.time() - start:.2f}s")
    print("\n📌 Pilares validados: orquestación, calidad autónoma, autolimpieza, resiliencia.")
    print("📖 Para ahorro de costes con tráfico real: python demos/demo_ahorro_tokens.py\n")


if __name__ == "__main__":
    main()
