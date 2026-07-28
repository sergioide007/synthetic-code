"""
sintetico_api.runner — Ejecuta cada uno de los 4 pilares del libro (y el
agente ReAct de `trazabilidad`) instrumentados de punta a punta, emitiendo
cada paso como un evento estructurado hacia un `TraceStore`.

Como `store.py`, este módulo no depende de FastAPI ni de Pydantic: recibe
y devuelve `dataclasses`/`dict` planos. Esto permite probarlo con pytest
sin necesidad de un cliente HTTP, y mantiene la capa web (routers/) como
un adaptador delgado que sólo traduce HTTP <-> estas funciones.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict

from sintetico import (
    AgentAnalyzer,
    AgentCircuitBreaker,
    DebtTracker,
    EmergencyOverrideGateway,
    ModelRouter,
    OverrideReason,
    SelfCleaningCodeLoop,
    SemanticCache,
    SwarmTokenOrchestrator,
    TokenBudget,
    TrustScoreCalculator,
    get_model_config,
    resolve_model_id,
)
from sintetico.providers import LLMRequest
from trazabilidad import ReActAgentWithLogging, StructuredAgentLogger, list_files, read_file, search_docs

from .store import TraceStore, new_id

__all__ = [
    "run_orchestration_case",
    "run_quality_gate_case",
    "run_self_cleaning_case",
    "run_resilient_swarm_case",
    "run_react_agent",
    "PILLARS",
]

PILLARS = ("orchestration", "quality_gate", "self_cleaning", "resilient_swarm")

_SIMULATED_TOKENS_BY_TIER = {"haiku": 50, "sonnet": 100, "opus": 200}
_DEMO_QUERIES = [
    "Hola, necesito ayuda",
    "Error en la base de datos",
    "Revisar arquitectura de pagos",
    "Hola, necesito ayuda",
    "Error 500 en API",
    "Diseñar sistema de caché",
    "Saludo inicial",
    "Problema con login",
    "Estrategia de microservicios",
]


def _make_logger(store: TraceStore, run_id: str, pillar: str, agent_id: str, team_id: str) -> StructuredAgentLogger:
    return StructuredAgentLogger(
        agent_id=agent_id,
        team_id=team_id,
        output_stream=None,  # no queremos volcar JSON a la consola del servidor
        sink=lambda entry: store.insert_event(entry, run_id=run_id, pillar=pillar),
    )


def _simulated_cost(alias: str, tokens: int) -> float:
    cfg = get_model_config(alias)
    input_tokens, output_tokens = int(tokens * 0.7), int(tokens * 0.3)
    return round(
        (input_tokens / 1_000_000) * cfg.cost_per_million_input
        + (output_tokens / 1_000_000) * cfg.cost_per_million_output,
        8,
    )


# ═══════════════════════════════════════════════════════════════════
# Pilar 1 — Orquestación eficiente
# ═══════════════════════════════════════════════════════════════════


def run_orchestration_case(store: TraceStore, team_id: str = "demo") -> Dict[str, Any]:
    run_id = store.create_run(pillar="orchestration", team_id=team_id)
    logger = _make_logger(store, run_id, "orchestration", "orchestration-agent", team_id)
    correlation_id = new_id("orch")
    logger.start_session(correlation_id, session_id=run_id, user_id=team_id, task_complexity="mixed")

    router = ModelRouter()
    cache = SemanticCache()
    budget = TokenBudget(20.0, team_id=team_id)

    total_cost, total_tokens, cache_hits = 0.0, 0, 0
    step = 0

    for query in _DEMO_QUERIES:
        step += 1
        if cache.get(query):
            cache_hits += 1
            logger.tool_result(
                correlation_id,
                tool_name="semantic_cache",
                success=True,
                result=f"HIT: '{query[:30]}'",
                error=None,
                latency_ms=0.1,
            )
            continue

        alias = router.select_model(query)
        tokens = _SIMULATED_TOKENS_BY_TIER[alias]
        cost = _simulated_cost(alias, tokens)

        logger.reasoning_step(
            correlation_id,
            step,
            thought=f"Enrutando '{query[:35]}' -> {alias}",
            tool_invoked="model_router",
            tool_args={"query": query},
            confidence=0.9,
        )

        if not budget.record_cost(cost):
            logger.security_event(
                correlation_id,
                event_type="budget_exceeded",
                details={"attempted_cost": cost, "remaining": budget.remaining},
                severity="WARNING",
            )
            break

        total_cost += cost
        total_tokens += tokens
        cache.set(query, "respuesta simulada")
        logger.tool_result(
            correlation_id,
            tool_name=f"llm:{resolve_model_id(alias)}",
            success=True,
            result=f"{tokens} tokens",
            error=None,
            latency_ms=tokens * 0.5,
            cost_usd=cost,
            model_used=resolve_model_id(alias),
        )

    opus_cfg = get_model_config("opus")
    opus_only_cost = (
        total_tokens * (opus_cfg.cost_per_million_input * 0.7 + opus_cfg.cost_per_million_output * 0.3) / 1_000_000
    )
    savings_pct = ((opus_only_cost - total_cost) / opus_only_cost * 100) if opus_only_cost > 0 else 0.0

    summary = {
        "queries": len(_DEMO_QUERIES),
        "cache_hits": cache_hits,
        "cache_hit_rate": cache.hit_rate,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "opus_only_cost_usd": round(opus_only_cost, 6),
        "savings_pct": round(savings_pct, 1),
        "budget_remaining_usd": round(budget.remaining, 4),
    }
    logger.finish_session(
        correlation_id,
        status="success",
        message="Orquestación completada",
        tokens_input=int(total_tokens * 0.7),
        tokens_output=int(total_tokens * 0.3),
        cost_usd=total_cost,
    )
    store.finish_run(run_id, status="completed", summary=summary)
    return {"run_id": run_id, "correlation_id": correlation_id, "summary": summary}


# ═══════════════════════════════════════════════════════════════════
# Pilar 2 — Calidad autónoma en CI/CD
# ═══════════════════════════════════════════════════════════════════

_DEMO_PRS = [
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


def run_quality_gate_case(store: TraceStore, team_id: str = "demo") -> Dict[str, Any]:
    run_id = store.create_run(pillar="quality_gate", team_id=team_id)
    logger = _make_logger(store, run_id, "quality_gate", "ci-quality-gate", team_id)
    correlation_id = new_id("ci")
    logger.start_session(correlation_id, session_id=run_id, user_id=team_id, task_complexity="ci_pipeline")

    calc = TrustScoreCalculator()
    override_gw = EmergencyOverrideGateway()
    results = []

    for pr in _DEMO_PRS:
        res = calc.calculate(
            pr["pass_rate"], pr["precision"], pr["human_approval"], pr["critical_bugs"], pr["consistency"]
        )
        logger.decision(
            correlation_id,
            decision=f"exit_code={res.ci_exit_code}",
            rationale=f"{pr['name']}: {res.recommendation}",
            confidence=res.score,
            trust_score=res.score,
        )
        entry = {
            "name": pr["name"],
            "trust_score": res.score,
            "level": res.level.value,
            "ci_exit_code": res.ci_exit_code,
            "recommendation": res.recommendation,
        }

        if res.ci_exit_code == 1:
            req_id = override_gw.request_override(
                run_id,
                res.recommendation,
                OverrideReason.FALSE_POSITIVE,
                "El agente carece de contexto de negocio; confirmado como falso positivo por el tech lead",
                "DevLead",
            )
            approval = override_gw.approve_override(req_id, "CTO")
            logger.security_event(
                correlation_id,
                event_type="emergency_override",
                details={"pr": pr["name"], "override_id": req_id, "status": approval["status"]},
                severity="WARNING",
            )
            entry["override"] = approval["status"]
        results.append(entry)

    summary = {
        "prs_evaluated": len(_DEMO_PRS),
        "results": results,
        "overrides_issued": len(override_gw.override_history),
        "blocked_critical": sum(1 for r in results if r["ci_exit_code"] == 2),
    }
    logger.finish_session(
        correlation_id,
        status="success",
        message="Gate de CI/CD evaluado",
        tokens_input=0,
        tokens_output=0,
        cost_usd=0.0,
    )
    store.finish_run(run_id, status="completed", summary=summary)
    return {"run_id": run_id, "correlation_id": correlation_id, "summary": summary}


# ═══════════════════════════════════════════════════════════════════
# Pilar 3 — Código autolimpiable
# ═══════════════════════════════════════════════════════════════════

_DIRTY_CODE = """def process_user(data):
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


def run_self_cleaning_case(store: TraceStore, team_id: str = "demo") -> Dict[str, Any]:
    run_id = store.create_run(pillar="self_cleaning", team_id=team_id)
    logger = _make_logger(store, run_id, "self_cleaning", "self-cleaning-agent", team_id)
    correlation_id = new_id("clean")
    logger.start_session(correlation_id, session_id=run_id, user_id=team_id, task_complexity="refactor")

    analyzer = AgentAnalyzer()
    smells = analyzer.analyze(_DIRTY_CODE)
    logger.reasoning_step(
        correlation_id,
        1,
        thought=f"Detectados {len(smells)} smell(s) de código",
        tool_invoked="agent_analyzer",
        tool_args=None,
        confidence=0.95,
    )

    cleaner = SelfCleaningCodeLoop()
    start = time.time()
    result = cleaner.clean(_DIRTY_CODE)
    elapsed_ms = (time.time() - start) * 1000

    logger.tool_result(
        correlation_id,
        tool_name="self_cleaning_loop",
        success=(result["status"] == "cleaned"),
        result=result["status"],
        error=result.get("error"),
        latency_ms=elapsed_ms,
    )

    dt = DebtTracker()
    debt = dt.add_debt("code_duplication", "medium", "Código duplicado en funciones de usuario (previo al refactor)")
    logger.decision(
        correlation_id, decision="registrar_deuda", rationale=debt.description, confidence=1.0, trust_score=1.0
    )

    summary = {
        "smells_detected": len(smells),
        "status": result["status"],
        "iterations": result.get("iterations"),
        "debt_registered": debt.id,
        "lines_before": len(_DIRTY_CODE.splitlines()),
        "lines_after": len(result["code"].splitlines()) if result["status"] == "cleaned" else None,
    }
    logger.finish_session(
        correlation_id,
        status="success" if result["status"] == "cleaned" else "partial",
        message=f"Autolimpieza: {result['status']}",
        tokens_input=0,
        tokens_output=0,
        cost_usd=0.0,
    )
    store.finish_run(run_id, status="completed", summary=summary)
    return {
        "run_id": run_id,
        "correlation_id": correlation_id,
        "summary": summary,
        "code_before": _DIRTY_CODE,
        "code_after": result.get("code"),
    }


# ═══════════════════════════════════════════════════════════════════
# Pilar 4 — Arquitectura resiliente (enjambre + circuit breaker)
# ═══════════════════════════════════════════════════════════════════


def run_resilient_swarm_case(store: TraceStore, team_id: str = "demo") -> Dict[str, Any]:
    run_id = store.create_run(pillar="resilient_swarm", team_id=team_id)
    logger = _make_logger(store, run_id, "resilient_swarm", "swarm-orchestrator", team_id)
    correlation_id = new_id("swarm")
    logger.start_session(correlation_id, session_id=run_id, user_id=team_id, task_complexity="multi_agent_debate")

    swarm = SwarmTokenOrchestrator(run_id, total_budget_tokens=400)
    messages = [
        ("AgentA", "AgentB", "Propongo solución X", 50),
        ("AgentB", "AgentA", "No, Y es mejor", 50),
        ("AgentA", "AgentB", "X es más eficiente", 50),
        ("AgentB", "AgentA", "Y es más seguro", 50),
        ("AgentA", "AgentB", "Insisto en X", 50),
        ("AgentB", "AgentA", "Y es mejor, revisa", 50),
    ]
    halted_at = None
    for i, (from_agent, to_agent, msg, cost) in enumerate(messages, 1):
        decision = swarm.add_message(from_agent, to_agent, msg, cost)
        logger.reasoning_step(
            correlation_id,
            i,
            thought=f"{from_agent}->{to_agent}: {msg}",
            tool_invoked="cycle_detector",
            tool_args={"decision": decision["decision"]},
            confidence=0.8,
        )
        if decision["decision"] == "HALT":
            logger.security_event(
                correlation_id,
                event_type="swarm_halted",
                details={"reason": decision["reason"], "step": i},
                severity="WARNING",
            )
            halted_at = i
            break

    breaker = AgentCircuitBreaker(max_cycles=3)
    for i in range(3):
        breaker.record_cycle(thought="mismo pensamiento repetido", tools=["search"], observation=f"obs{i}")
    logger.security_event(
        correlation_id,
        event_type="circuit_breaker_check",
        details={"state": breaker.state, "reason": breaker.failure_reason},
        severity="WARNING" if breaker.is_open() else "INFO",
    )

    summary = {
        "messages_processed": halted_at or len(messages),
        "halted": halted_at is not None,
        "halted_at_step": halted_at,
        "tokens_spent": swarm.spent_tokens,
        "circuit_breaker_open": breaker.is_open(),
        "circuit_breaker_reason": breaker.failure_reason,
    }
    logger.finish_session(
        correlation_id,
        status="success",
        message="Simulación de enjambre completada",
        tokens_input=swarm.spent_tokens,
        tokens_output=0,
        cost_usd=0.0,
    )
    store.finish_run(run_id, status="completed", summary=summary)
    return {"run_id": run_id, "correlation_id": correlation_id, "summary": summary}


# ═══════════════════════════════════════════════════════════════════
# Agente ReAct en vivo (usa proveedor real si hay API key, si no Mock)
# ═══════════════════════════════════════════════════════════════════


def _mock_react_llm_call() -> Callable[[str, dict], dict]:
    """LLM simulado que entiende el formato ReAct del prompt del sistema y
    resuelve en 1-2 ciclos de forma determinista, para poder ver el flujo
    completo en el dashboard sin necesitar una API key."""

    def llm_call(prompt: str, params: dict) -> Dict[str, Any]:
        lowered = prompt.lower()
        if "observation:" not in lowered:
            if "archivo" in lowered or "file" in lowered:
                content = (
                    "Thought: Necesito listar los archivos disponibles.\n"
                    'Action: list_files\nAction Input: {"directory": "."}'
                )
            elif "buscar" in lowered or "search" in lowered or "documentaci" in lowered:
                content = (
                    'Thought: Debo buscar en la documentación.\nAction: search_docs\nAction Input: {"query": "info"}'
                )
            else:
                content = (
                    "Thought: Puedo responder directamente sin usar herramientas.\n"
                    'Action: None\nAction Input: {"result": '
                    '"Respuesta simulada: en un entorno con API key real, aquí '
                    'respondería el modelo Claude real."}'
                )
        else:
            content = (
                "Thought: Ya tengo suficiente información para responder.\n"
                'Action: None\nAction Input: {"result": "Tarea completada con la información obtenida."}'
            )
        return {"content": content, "tokens": 45}

    return llm_call


def _real_react_llm_call(provider) -> Callable[[str, dict], dict]:
    def llm_call(prompt: str, params: dict) -> Dict[str, Any]:
        response = provider.complete(
            LLMRequest(
                system_prompt="",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=params.get("max_tokens", 512),
                temperature=params.get("temperature", 0.2),
            )
        )
        tokens = response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)
        return {"content": response.content, "tokens": tokens}

    return llm_call


def run_react_agent(
    store: TraceStore, query: str, team_id: str = "demo", provider_type: str = "auto"
) -> Dict[str, Any]:
    run_id = store.create_run(pillar="react_agent", team_id=team_id)

    is_real = False
    provider = None
    if provider_type != "mock":
        try:
            import os
            from sintetico.real_providers import create_provider

            if provider_type == "anthropic" or (provider_type == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
                provider = create_provider("anthropic", default_model="haiku")
                is_real = True
            elif provider_type == "openai" or (provider_type == "auto" and os.environ.get("OPENAI_API_KEY")):
                provider = create_provider("openai", default_model="gpt-4o")
                is_real = True
        except Exception:
            provider = None
            is_real = False

    llm_call = _real_react_llm_call(provider) if is_real else _mock_react_llm_call()

    logger = _make_logger(store, run_id, "react_agent", "react-demo-agent", team_id)
    agent = ReActAgentWithLogging(
        llm_call=llm_call,
        tools={"list_files": list_files, "read_file": read_file, "search_docs": search_docs},
        logger=logger,
        max_cycles=5,
    )
    result = agent.run(user_query=query, session_id=run_id, user_id=team_id)

    summary = {**result, "provider": "real" if is_real else "mock"}
    store.finish_run(run_id, status="completed" if result.get("status") == "success" else "partial", summary=summary)
    return {"run_id": run_id, "correlation_id": result.get("correlation_id"), "summary": summary}
