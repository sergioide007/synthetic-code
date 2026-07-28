"""
sintetico.providers — Patrón Backend-Agnostic para proveedores de LLM
(Capítulo 19).

Define el contrato `LLMProvider` que desacopla el resto del sistema del
SDK concreto (Anthropic, OpenAI, un servidor local, etc.). Las
implementaciones *reales* que hablan con APIs externas viven en
`sintetico.real_providers` para no forzar `anthropic`/`openai`/`requests`
como dependencias duras de este módulo base.
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
    """Contrato que debe cumplir cualquier proveedor de modelo de lenguaje."""

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    @property
    @abstractmethod
    def cost_per_million_tokens(self) -> float:
        """Coste aproximado de entrada, usado sólo para estimaciones rápidas.

        Para coste real por llamada, usar `get_last_cost()` en los
        proveedores reales, que combina el precio de entrada y salida con
        el `usage` efectivamente reportado por la API.
        """


class MockProvider(LLMProvider):
    """Proveedor determinista sin dependencias externas, para tests y demos
    offline. No debe usarse para medir ahorro de costes real."""

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
    """Se lanza cuando el proveedor principal y todos los de respaldo fallan."""


class BackendAgnosticAgent:
    """Agente que delega en un `LLMProvider` intercambiable, con fallback."""

    def __init__(self, provider: LLMProvider, fallback_providers: Optional[List[LLMProvider]] = None):
        self.provider = provider
        self.fallback_providers = fallback_providers or []

    def process(self, system_prompt: str, user_message: str) -> LLMResponse:
        request = LLMRequest(system_prompt, [{"role": "user", "content": user_message}])
        errors = []
        for provider in [self.provider, *self.fallback_providers]:
            try:
                return provider.complete(request)
            except Exception as exc:  # noqa: BLE001 - queremos capturar cualquier fallo de proveedor
                errors.append(f"{type(provider).__name__}: {exc}")
        raise AllProvidersFailedError("Todos los proveedores fallaron: " + "; ".join(errors))

    def swap_provider(self, new_provider: LLMProvider) -> None:
        self.provider = new_provider
