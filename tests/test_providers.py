import sys
import types

import pytest

from sintetico.providers import BackendAgnosticAgent, LLMProvider, LLMRequest, LLMResponse, MockProvider


class FailingProvider(LLMProvider):
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("proveedor caído")

    @property
    def cost_per_million_tokens(self) -> float:
        return 1.0


class TestMockProvider:
    def test_returns_deterministic_response(self):
        response = MockProvider().complete(LLMRequest(system_prompt="hola", messages=[]))
        assert response.model_name == "mock"
        assert response.usage["input_tokens"] > 0


class TestBackendAgnosticAgent:
    def test_falls_back_when_primary_fails(self):
        agent = BackendAgnosticAgent(FailingProvider(), fallback_providers=[MockProvider()])
        response = agent.process("system", "hola")
        assert response.model_name == "mock"

    def test_raises_when_all_providers_fail(self):
        agent = BackendAgnosticAgent(FailingProvider(), fallback_providers=[FailingProvider()])
        with pytest.raises(Exception):
            agent.process("system", "hola")


@pytest.fixture
def fake_anthropic_module(monkeypatch):
    """Inyecta un módulo `anthropic` falso en sys.modules para poder probar
    RealAnthropicProvider sin el SDK real ni acceso a red."""

    class FakeAuthenticationError(Exception):
        pass

    class FakeRateLimitError(Exception):
        pass

    class FakeBadRequestError(Exception):
        pass

    class FakeAPIError(Exception):
        pass

    call_log = {"count": 0, "models_used": []}

    class FakeContentBlock:
        def __init__(self, text):
            self.type = "text"
            self.text = text

    class FakeUsage:
        def __init__(self, input_tokens=100, output_tokens=50):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class FakeMessage:
        def __init__(self, content_text):
            self.content = [FakeContentBlock(content_text)]
            self.usage = FakeUsage()

    class FakeMessages:
        def create(self, model, max_tokens, temperature, system, messages):
            call_log["count"] += 1
            call_log["models_used"].append(model)
            return FakeMessage(f"respuesta simulada para {model}")

    class FakeAnthropic:
        def __init__(self, api_key, timeout=None):
            self.api_key = api_key
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(
        Anthropic=FakeAnthropic,
        AuthenticationError=FakeAuthenticationError,
        RateLimitError=FakeRateLimitError,
        BadRequestError=FakeBadRequestError,
        APIError=FakeAPIError,
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return call_log


class TestRealAnthropicProvider:
    def test_requires_api_key(self, monkeypatch, fake_anthropic_module):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from sintetico.real_providers import AuthenticationError, RealAnthropicProvider

        with pytest.raises(AuthenticationError):
            RealAnthropicProvider(api_key=None)

    def test_resolves_alias_to_real_model_id(self, fake_anthropic_module):
        from sintetico.model_registry import resolve_model_id
        from sintetico.real_providers import RealAnthropicProvider

        provider = RealAnthropicProvider(api_key="fake-key", default_model="haiku")
        response = provider.complete(LLMRequest(system_prompt="s", messages=[{"role": "user", "content": "hola"}]))

        assert response.model_name == resolve_model_id("haiku")
        assert fake_anthropic_module["models_used"] == [resolve_model_id("haiku")]

    def test_request_model_overrides_default(self, fake_anthropic_module):
        from sintetico.model_registry import resolve_model_id
        from sintetico.real_providers import RealAnthropicProvider

        provider = RealAnthropicProvider(api_key="fake-key", default_model="haiku")
        provider.complete(LLMRequest(system_prompt="s", messages=[], model="opus"))

        assert fake_anthropic_module["models_used"] == [resolve_model_id("opus")]

    def test_get_last_cost_matches_the_model_actually_billed(self, fake_anthropic_module):
        """Regresión central: en la v1, el coste se calculaba con el modelo
        elegido por el router pero la llamada real usaba otro modelo
        (el `default_model` fijo del proveedor), por lo que el coste
        reportado no correspondía a lo realmente facturado."""
        from sintetico.model_registry import get_model_config, resolve_model_id
        from sintetico.real_providers import RealAnthropicProvider

        provider = RealAnthropicProvider(api_key="fake-key", default_model="haiku")
        provider.complete(LLMRequest(system_prompt="s", messages=[], model="opus"))

        cfg = get_model_config(resolve_model_id("opus"))
        expected_cost = round(
            (100 / 1_000_000) * cfg.cost_per_million_input + (50 / 1_000_000) * cfg.cost_per_million_output, 6
        )
        assert provider.get_last_cost() == expected_cost

    def test_retries_on_rate_limit_then_succeeds(self, fake_anthropic_module, monkeypatch):
        from sintetico.real_providers import RealAnthropicProvider, RetryConfig

        provider = RealAnthropicProvider(
            api_key="fake-key",
            default_model="haiku",
            retry_config=RetryConfig(max_attempts=3, base_delay_seconds=0.001),
        )

        attempts = {"n": 0}
        original_create = provider.client.messages.create

        def flaky_create(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise sys.modules["anthropic"].RateLimitError("rate limited")
            return original_create(*args, **kwargs)

        provider.client.messages.create = flaky_create
        response = provider.complete(LLMRequest(system_prompt="s", messages=[]))
        assert attempts["n"] == 2
        assert response.content.startswith("respuesta simulada")
