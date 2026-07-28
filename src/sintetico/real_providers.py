"""
sintetico.real_providers — Integración con APIs reales de LLM.

Soporta Anthropic, OpenAI y modelos locales servidos con una interfaz
compatible con OpenAI (vLLM/Ollama). A diferencia de la v1 del código
(`real_api_utils.py`), este módulo:

- Implementa reintentos con backoff exponencial + jitter (la v1 documentaba
  "maneja reintentos" pero no reintentaba nada).
- Aplica timeouts explícitos en cada llamada de red.
- Distingue errores de autenticación, rate limit y longitud de contexto
  para permitir manejo diferenciado aguas arriba.
- Resuelve el modelo real a partir de `sintetico.model_registry`, evitando
  identificadores de modelo hardcodeados que no existen en la API real.
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

from .model_registry import get_model_config, resolve_model_id
from .providers import LLMProvider, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

__all__ = [
    "LLMProviderError",
    "AuthenticationError",
    "RateLimitError",
    "ContextLengthError",
    "RetryConfig",
    "RealAnthropicProvider",
    "RealOpenAIProvider",
    "LocalLLMProvider",
    "create_provider",
]


# ─── Excepciones ─────────────────────────────────────────────────────
class LLMProviderError(Exception):
    """Error genérico al comunicarse con un proveedor de LLM."""


class AuthenticationError(LLMProviderError):
    """Credenciales ausentes o inválidas. No se debe reintentar."""


class RateLimitError(LLMProviderError):
    """Límite de tasa alcanzado. Reintentable con backoff."""


class ContextLengthError(LLMProviderError):
    """El prompt excede la ventana de contexto del modelo. No reintentable."""


@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0

    def delay_for(self, attempt: int) -> float:
        """Backoff exponencial con jitter completo (evita 'thundering herd')."""
        capped = min(self.max_delay_seconds, self.base_delay_seconds * (2**attempt))
        return random.uniform(0, capped)


def _with_retries(fn, retry_config: RetryConfig, retryable_exceptions: tuple):
    """Ejecuta `fn` reintentando ante excepciones reintentables."""
    last_exc: Optional[Exception] = None
    for attempt in range(retry_config.max_attempts):
        try:
            return fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == retry_config.max_attempts - 1:
                break
            delay = retry_config.delay_for(attempt)
            logger.warning(
                "Intento %s/%s falló (%s). Reintentando en %.2fs",
                attempt + 1,
                retry_config.max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _usage_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    cfg = get_model_config(model_id)
    return round(
        (input_tokens / 1_000_000) * cfg.cost_per_million_input
        + (output_tokens / 1_000_000) * cfg.cost_per_million_output,
        6,
    )


# ─── Proveedor Anthropic ──────────────────────────────────────────────
class RealAnthropicProvider(LLMProvider):
    """Proveedor que llama a la API real de Anthropic.

    El modelo a usar se resuelve por llamada a partir de `request.model`
    (si se especifica) o del `default_model` configurado en el
    constructor; en ambos casos acepta tanto un alias corto ("haiku",
    "sonnet", "opus") como un id completo de modelo.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "sonnet",
        timeout_seconds: float = 60.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise AuthenticationError(
                "ANTHROPIC_API_KEY no configurada. Expórtala como variable de "
                "entorno o pásala explícitamente a RealAnthropicProvider(api_key=...)."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Instala el SDK: pip install anthropic") from exc

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key, timeout=timeout_seconds)
        self.default_model = default_model
        self.retry_config = retry_config or RetryConfig()
        self._last_usage: dict = {}
        self._last_model_id: Optional[str] = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        model_id = resolve_model_id(request.model or self.default_model)

        def _call() -> LLMResponse:
            try:
                response = self.client.messages.create(
                    model=model_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    system=request.system_prompt,
                    messages=request.messages,
                )
            except self._anthropic.AuthenticationError as exc:
                raise AuthenticationError(str(exc)) from exc
            except self._anthropic.RateLimitError as exc:
                raise RateLimitError(str(exc)) from exc
            except self._anthropic.BadRequestError as exc:
                if "context" in str(exc).lower() or "too long" in str(exc).lower():
                    raise ContextLengthError(str(exc)) from exc
                raise LLMProviderError(str(exc)) from exc
            except self._anthropic.APIError as exc:
                raise LLMProviderError(str(exc)) from exc

            content = "".join(block.text for block in response.content if block.type == "text")
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            self._last_usage = usage
            self._last_model_id = model_id
            return LLMResponse(content=content, usage=usage, model_name=model_id)

        try:
            return _with_retries(_call, self.retry_config, (RateLimitError, LLMProviderError))
        except (AuthenticationError, ContextLengthError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en Anthropic tras reintentos: %s", exc)
            raise LLMProviderError(f"Anthropic error: {exc}") from exc

    @property
    def cost_per_million_tokens(self) -> float:
        return get_model_config(self.default_model).cost_per_million_input

    def get_last_cost(self) -> float:
        if not self._last_usage or not self._last_model_id:
            return 0.0
        return _usage_cost(self._last_model_id, self._last_usage["input_tokens"], self._last_usage["output_tokens"])


# ─── Proveedor OpenAI ──────────────────────────────────────────────
class RealOpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
        timeout_seconds: float = 60.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise AuthenticationError("OPENAI_API_KEY no configurada")
        try:
            import openai
        except ImportError as exc:
            raise ImportError("Instala el SDK: pip install openai") from exc

        self._openai = openai
        self.client = openai.OpenAI(api_key=self.api_key, timeout=timeout_seconds)
        self.default_model = default_model
        self.retry_config = retry_config or RetryConfig()
        self._last_usage: dict = {}
        self._last_model_id: Optional[str] = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        model_id = resolve_model_id(request.model or self.default_model)

        def _call() -> LLMResponse:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    messages=[{"role": "system", "content": request.system_prompt}, *request.messages],
                )
            except self._openai.AuthenticationError as exc:
                raise AuthenticationError(str(exc)) from exc
            except self._openai.RateLimitError as exc:
                raise RateLimitError(str(exc)) from exc
            except self._openai.BadRequestError as exc:
                if "context" in str(exc).lower() or "maximum context" in str(exc).lower():
                    raise ContextLengthError(str(exc)) from exc
                raise LLMProviderError(str(exc)) from exc
            except self._openai.APIError as exc:
                raise LLMProviderError(str(exc)) from exc

            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
            self._last_usage = usage
            self._last_model_id = model_id
            return LLMResponse(content=response.choices[0].message.content, usage=usage, model_name=model_id)

        try:
            return _with_retries(_call, self.retry_config, (RateLimitError, LLMProviderError))
        except (AuthenticationError, ContextLengthError):
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Error en OpenAI tras reintentos: %s", exc)
            raise LLMProviderError(f"OpenAI error: {exc}") from exc

    @property
    def cost_per_million_tokens(self) -> float:
        return get_model_config(self.default_model).cost_per_million_input

    def get_last_cost(self) -> float:
        if not self._last_usage or not self._last_model_id:
            return 0.0
        return _usage_cost(self._last_model_id, self._last_usage["input_tokens"], self._last_usage["output_tokens"])


# ─── Proveedor Local (vLLM/Ollama, compatible con OpenAI) ────────────
class LocalLLMProvider(LLMProvider):
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: str = "local",
        timeout_seconds: float = 120.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.base_url = base_url or os.environ.get("LOCALLLM_BASE_URL", "http://localhost:8000/v1")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.retry_config = retry_config or RetryConfig(max_attempts=2)
        self._last_usage: dict = {}
        try:
            import requests
        except ImportError as exc:
            raise ImportError("Instala requests: pip install requests") from exc
        self._requests = requests

    def complete(self, request: LLMRequest) -> LLMResponse:
        model_id = resolve_model_id(request.model or self.model_name)
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model_id,
            "messages": [{"role": "system", "content": request.system_prompt}, *request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        def _call() -> LLMResponse:
            try:
                resp = self._requests.post(url, json=payload, timeout=self.timeout_seconds)
                resp.raise_for_status()
            except self._requests.exceptions.Timeout as exc:
                raise LLMProviderError(f"Timeout tras {self.timeout_seconds}s") from exc
            except self._requests.exceptions.RequestException as exc:
                raise LLMProviderError(str(exc)) from exc

            data = resp.json()
            usage = data.get("usage", {})
            parsed_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
            self._last_usage = parsed_usage
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                usage=parsed_usage,
                model_name=model_id,
            )

        return _with_retries(_call, self.retry_config, (LLMProviderError,))

    @property
    def cost_per_million_tokens(self) -> float:
        return get_model_config(self.model_name).cost_per_million_input

    def get_last_cost(self) -> float:
        if not self._last_usage:
            return 0.0
        return _usage_cost(self.model_name, self._last_usage["input_tokens"], self._last_usage["output_tokens"])


def create_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factoría de proveedores. `provider_type` en {"anthropic", "openai", "local", "mock"}."""
    if provider_type == "anthropic":
        return RealAnthropicProvider(**kwargs)
    if provider_type == "openai":
        return RealOpenAIProvider(**kwargs)
    if provider_type == "local":
        return LocalLLMProvider(**kwargs)
    if provider_type == "mock":
        from .providers import MockProvider

        return MockProvider()
    raise ValueError(f"Tipo de proveedor desconocido: {provider_type}")
