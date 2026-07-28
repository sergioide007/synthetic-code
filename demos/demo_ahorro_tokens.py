#!/usr/bin/env python3
"""
demo_ahorro_tokens.py — Demostración de ahorro de costes con tráfico real.

Mide tokens, latencia y coste REAL de un lote de consultas, comparando el
enrutamiento inteligente (ModelRouter + SemanticCache) contra el coste de
haber usado el modelo más caro (Opus) para todo.

Requiere una API key real en variables de entorno:

    export ANTHROPIC_API_KEY="sk-ant-..."
    python demos/demo_ahorro_tokens.py

    # o, para OpenAI:
    export OPENAI_API_KEY="sk-..."
    python demos/demo_ahorro_tokens.py --provider openai

Sin ninguna key configurada, la demo se ejecuta igualmente en modo
simulación (`MockProvider`) para que el flujo se pueda inspeccionar sin
coste, pero lo indica explícitamente: los números de ahorro en ese modo
no son reales.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List

from sintetico import ModelRouter, SemanticCache, TokenBudget, compute_tco, get_model_config
from sintetico.providers import LLMRequest, MockProvider
from sintetico.real_providers import AuthenticationError, LLMProviderError, create_provider

TEST_QUERIES = [
    ("simple", "Hola, ¿cómo estás?"),
    ("simple", "¿Cuál es la capital de Francia?"),
    ("medium", "Explica el concepto de inversión de dependencias en programación."),
    ("medium", "¿Cómo funciona un circuit breaker en microservicios?"),
    ("complex", "Diseña una arquitectura de microservicios para un sistema de pagos con alta disponibilidad."),
    ("complex", "Analiza los trade-offs entre consistencia fuerte y eventual en bases de datos distribuidas."),
]


@dataclass
class QueryResult:
    complexity: str
    query: str
    model_alias: str
    model_id: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_s: float
    cache_hit: bool = False


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def select_provider(provider_name: str):
    """Crea el proveedor solicitado, o cae a MockProvider con aviso claro."""
    if provider_name == "mock":
        print("📝 Modo simulación explícito (--provider mock): usando MockProvider.")
        return MockProvider(), False

    try:
        if provider_name == "anthropic":
            provider = create_provider("anthropic", default_model="sonnet")
        elif provider_name == "openai":
            provider = create_provider("openai", default_model="gpt-4o")
        else:
            raise ValueError(f"Proveedor no soportado: {provider_name}")
        print(f"✅ Conectado a {provider_name} con API key real.")
        return provider, True
    except AuthenticationError as exc:
        print(f"⚠️  {exc}")
        print("📝 Sin API key disponible: usando MockProvider (los costes NO son reales).")
        return MockProvider(), False
    except ImportError as exc:
        print(f"⚠️  {exc}")
        print("📝 SDK no instalado: usando MockProvider (los costes NO son reales).")
        return MockProvider(), False


def run_queries(
    provider, is_real: bool, router: ModelRouter, cache: SemanticCache, budget: TokenBudget
) -> List[QueryResult]:
    results: List[QueryResult] = []

    for i, (complexity, query) in enumerate(TEST_QUERIES, 1):
        cached = cache.get(query)
        if cached:
            print(f"\n[{i}] 🟢 CACHÉ HIT: '{query[:45]}...'")
            results.append(QueryResult(complexity, query, "cache", "cache", 0, 0, 0.0, 0.0, cache_hit=True))
            continue

        alias = router.select_model(query)
        print(f"\n[{i}] 🔵 Alias enrutado: {alias:8} (complejidad anotada: {complexity})")
        print(f"     Query: '{query[:55]}...'")

        try:
            start = time.time()
            # request.model = alias: el proveedor resuelve el id real y
            # es EXACTAMENTE el modelo que se factura más abajo.
            response = provider.complete(
                LLMRequest(
                    system_prompt="Eres un asistente útil. Responde de forma concisa en español.",
                    messages=[{"role": "user", "content": query}],
                    model=alias,
                    max_tokens=400,
                    temperature=0.3,
                )
            )
            elapsed = time.time() - start

            if is_real and hasattr(provider, "get_last_cost"):
                cost = provider.get_last_cost()
            else:
                cost = 0.0

            tokens_in = response.usage.get("input_tokens", 0)
            tokens_out = response.usage.get("output_tokens", 0)

            cache.set(query, response.content)
            if not budget.record_cost(cost):
                print("     ❌ Presupuesto excedido. Deteniendo la demo.")
                break

            print(
                f"     ⏱️  {elapsed:.2f}s | Modelo real: {response.model_name} | "
                f"Tokens: {tokens_in}+{tokens_out} | Costo: ${cost:.6f}"
            )

            results.append(
                QueryResult(complexity, query, alias, response.model_name, tokens_in, tokens_out, cost, elapsed)
            )

        except LLMProviderError as exc:
            print(f"     ❌ Error del proveedor: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"     ❌ Error inesperado: {exc}")

    return results


def print_summary(results: List[QueryResult], is_real: bool, budget: TokenBudget) -> None:
    banner("📊 RESUMEN DE LA DEMOSTRACIÓN")

    answered = [r for r in results if not r.cache_hit]
    cache_hits = sum(1 for r in results if r.cache_hit)
    total_tokens = sum(r.tokens_in + r.tokens_out for r in answered)
    total_cost = sum(r.cost_usd for r in answered)
    total_time = sum(r.latency_s for r in answered)

    print(f"  • Consultas totales: {len(TEST_QUERIES)}  | Cache hits: {cache_hits}")
    print(f"  • Tokens consumidos: {total_tokens}  | Tiempo total: {total_time:.2f}s")
    print(f"  • Costo total ({'REAL' if is_real else 'simulado, no facturado'}): ${total_cost:.6f}")
    print(f"  • Presupuesto restante: ${budget.remaining:.2f}")

    if not is_real:
        print("\n  ⚠️  Estos números provienen de MockProvider: NO reflejan coste real.")
        print("      Configura ANTHROPIC_API_KEY u OPENAI_API_KEY para medir ahorro real.")
        return

    if total_tokens == 0:
        print("\n  ⚠️  No se completó ninguna consulta real; no hay base para calcular ahorro.")
        return

    # Coste real vs. "todo hubiera ido a Opus" con la MISMA distribución de
    # tokens observada (comparación justa: mismo volumen, distinto modelo).
    opus_cfg = get_model_config("opus")
    avg_input_share = sum(r.tokens_in for r in answered) / total_tokens
    opus_only_cost = (
        total_tokens * avg_input_share * opus_cfg.cost_per_million_input
        + total_tokens * (1 - avg_input_share) * opus_cfg.cost_per_million_output
    ) / 1_000_000

    savings = opus_only_cost - total_cost
    savings_pct = (savings / opus_only_cost * 100) if opus_only_cost > 0 else 0.0

    banner("📈 ANÁLISIS DE AHORRO (coste real medido vs. Opus para todo)")
    print(f"  Costo real con enrutamiento inteligente:  ${total_cost:.6f}")
    print(f"  Costo estimado si todo fuera Opus:        ${opus_only_cost:.6f}")
    print(f"  AHORRO:                                    ${savings:.6f}  ({savings_pct:.1f}%)")

    verify_minutes = sum({"opus": 2, "sonnet": 1, "haiku": 0.5}.get(r.model_alias, 1) for r in answered)
    tco = compute_tco(
        tokens_input=sum(r.tokens_in for r in answered),
        tokens_output=sum(r.tokens_out for r in answered),
        verification_time_minutes=verify_minutes,
        engineer_rate_per_hour=50,
        model_cost_per_million=3.0,
    )
    print(f"\n  💵 TCO (tokens + {verify_minutes:.1f} min de verificación humana a $50/h): ${tco:.4f}")

    if len(answered) >= 3:
        daily_queries = 1000
        avg_cost_per_query = total_cost / len(answered)
        monthly_routed = avg_cost_per_query * daily_queries * 30
        avg_opus_cost_per_query = opus_only_cost / len(answered)
        monthly_opus = avg_opus_cost_per_query * daily_queries * 30
        banner(f"📊 PROYECCIÓN MENSUAL (extrapolando a {daily_queries:,} consultas/día)")
        print(f"  Enrutamiento inteligente: ${monthly_routed:,.2f}/mes")
        print(f"  Solo Opus:                ${monthly_opus:,.2f}/mes")
        print(
            f"  Ahorro mensual estimado:  ${monthly_opus - monthly_routed:,.2f}/mes "
            f"(${(monthly_opus - monthly_routed) * 12:,.2f}/año)"
        )
        print("\n  Nota: la extrapolación asume que la mezcla de complejidad de estas 6")
        print("  consultas es representativa del tráfico real; en producción, valida esto")
        print("  con una muestra mayor antes de reportar la cifra a negocio.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["auto", "anthropic", "openai", "mock"],
        default="auto",
        help="Proveedor a usar. 'auto' detecta la primera API key disponible en el entorno.",
    )
    args = parser.parse_args()

    banner("🚀 DEMOSTRACIÓN REAL DE AHORRO DE COSTOS")

    provider_name = args.provider
    if provider_name == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider_name = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider_name = "openai"
        else:
            provider_name = "mock"

    provider, is_real = select_provider(provider_name)

    router = ModelRouter()
    cache = SemanticCache(max_size=1000)
    budget = TokenBudget(monthly_budget=10.0)

    banner("📊 PROCESANDO CONSULTAS CON ENRUTAMIENTO INTELIGENTE")
    results = run_queries(provider, is_real, router, cache, budget)

    print_summary(results, is_real, budget)

    banner("✅ DEMOSTRACIÓN COMPLETADA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
