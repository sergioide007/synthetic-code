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
