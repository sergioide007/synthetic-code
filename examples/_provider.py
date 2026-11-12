"""Shared helper: examples run offline with a deterministic mock by default.

    SINTETICO_PROVIDER=anthropic ANTHROPIC_API_KEY=... python examples/01_gateway_budget_audit.py
    SINTETICO_PROVIDER=openai    OPENAI_API_KEY=...    python examples/01_gateway_budget_audit.py
"""
import os

from sintetico.providers import MockProvider


def get_provider():
    name = os.environ.get("SINTETICO_PROVIDER", "mock").lower()
    if name == "anthropic":
        from sintetico.real_providers import RealAnthropicProvider
        return RealAnthropicProvider()
    if name == "openai":
        from sintetico.real_providers import RealOpenAIProvider
        return RealOpenAIProvider()
    return MockProvider()
