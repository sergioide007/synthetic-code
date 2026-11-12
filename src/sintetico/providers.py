"""
sintetico.providers — Backend-Agnostic pattern for LLM providers
(Chapter 19).

Defines the `LLMProvider` contract that decouples the rest of the system
from the concrete SDK (Anthropic, OpenAI, a local server, etc.). The
*real* implementations that talk to external APIs live in
`sintetico.real_providers`, so that `anthropic`/`openai`/`requests` are
not hard dependencies of this base module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "LLMRequest",
    "LLMResponse",
    "LLMProvider",
    "MockProvider",
    "BackendAgnosticAgent",
    "AllProvidersFailedError",
]


@dataclass
class LLMRequest:
    system_prompt: str
    messages: List[dict]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, int]
    model_name: str


class LLMProvider(ABC):
    """Contract that every language-model provider must fulfil."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    @property
    @abstractmethod
    def cost_per_million_tokens(self) -> float:
        """Approximate input cost, used only for quick estimates.

        For the real cost of a call, use `get_last_cost()` on the real
        providers, which combines the input and output prices with the
        `usage` actually reported by the API.
        """


class MockProvider(LLMProvider):
    """Deterministic provider with no external dependencies, for tests and
    offline demos. It must not be used to measure real cost savings."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        preview = request.system_prompt[:30]
        return LLMResponse(
            content=f"[mock] {preview}",
            usage={"input_tokens": 10, "output_tokens": 5},
            model_name="mock",
        )

    @property
    def cost_per_million_tokens(self) -> float:
        return 0.0


class AllProvidersFailedError(RuntimeError):
    """Raised when the primary provider and every fallback fail."""


class BackendAgnosticAgent:
    """Agent that delegates to a swappable `LLMProvider`, with fallback."""

    def __init__(self, provider: LLMProvider, fallback_providers: Optional[List[LLMProvider]] = None):
        self.provider = provider
        self.fallback_providers = fallback_providers or []

    def process(self, system_prompt: str, user_message: str) -> LLMResponse:
        request = LLMRequest(system_prompt, [{"role": "user", "content": user_message}])
        errors = []
        for provider in [self.provider, *self.fallback_providers]:
            try:
                return provider.complete(request)
            except Exception as exc:  # noqa: BLE001 - any provider failure must be caught here
                errors.append(f"{type(provider).__name__}: {exc}")
        raise AllProvidersFailedError("Todos los proveedores fallaron: " + "; ".join(errors))

    def swap_provider(self, new_provider: LLMProvider) -> None:
        self.provider = new_provider
