import io, json, runpy
from pathlib import Path
from sintetico.model_registry import MODEL_REGISTRY

SCRIPT = runpy.run_path(str(Path(__file__).resolve().parent.parent / "scripts" / "check_model_registry.py"))


def _fake_opener(pages):
    it = iter(pages)
    def opener(req, timeout=0):
        assert req.get_header("X-api-key") == "k" and req.get_header("Anthropic-version")
        return io.BytesIO(json.dumps(next(it)).encode())
    return opener


def test_pagination_and_missing_ids_logic():
    pages = [{"data": [{"id": "a"}, {"id": "b"}], "has_more": True, "last_id": "b"}, {"data": [{"id": "c"}], "has_more": False}]
    served = SCRIPT["fetch_all_ids"]("k", _fake_opener(pages))
    assert served == ["a", "b", "c"]
    assert SCRIPT["missing_ids"](["a", "z"], served) == ["z"]


def test_registry_entries_are_sane():
    for alias, cfg in MODEL_REGISTRY.items():
        assert cfg.id and cfg.provider in {"anthropic", "openai", "local"}
        assert 0 < cfg.cost_per_million_input <= cfg.cost_per_million_output * 1.0001 or cfg.provider == "local"
        assert cfg.max_context_tokens > 0 and cfg.tier in {"fast", "balanced", "quality"}


def test_capability_profiles_only_name_registered_models():
    from sintetico.adapters import PROFILES
    for profile in PROFILES.values():
        for alias in profile.recommended + profile.avoid:
            assert alias in MODEL_REGISTRY, f"{profile.name} names unregistered model {alias}"


def test_openai_catalog_matches_published_prices():
    # Standard-tier prices published by OpenAI on 2026-11-13 (input, output per 1M tokens).
    expected = {"gpt-6-astra": (10.00, 50.00), "gpt-6.1-sol": (2.00, 10.00), "gpt-6-luna": (0.10, 0.50), "gpt-4o": (2.50, 10.00)}
    for alias, (inp, out) in expected.items():
        cfg = MODEL_REGISTRY[alias]
        assert (cfg.provider, cfg.cost_per_million_input, cfg.cost_per_million_output) == ("openai", inp, out)
    assert "gpt-4-turbo" not in MODEL_REGISTRY and "gpt-3.5-turbo" not in MODEL_REGISTRY


def test_openai_kwargs_suit_reasoning_and_gpt4_models():
    from sintetico.real_providers import openai_sampling_kwargs
    assert openai_sampling_kwargs("gpt-6.1-sol", 100, 0.2) == {"max_completion_tokens": 100}
    assert openai_sampling_kwargs("gpt-6-astra", 100, 0.2) == {"max_completion_tokens": 100}
    assert openai_sampling_kwargs("gpt-4o", 100, 0.2) == {"max_completion_tokens": 100, "temperature": 0.2}
