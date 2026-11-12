"""
sintetico.model_registry — catalog of models and their cost per million tokens.

The model identifiers and prices of an LLM provider change often (new
versions, launch discounts, retirement of legacy models). That is why
this catalog:

1. Lives in a single place, instead of being scattered as string
   literals across several demo files (as in v1 of this code, where
   "claude-haiku-4-5" was hardcoded in four different places with
   inconsistent identifiers and without the date suffix the real API
   requires, so every real call failed with "model not found").
2. Can be overridden with environment variables, so that the book's
   examples keep working when Anthropic publishes new models, with no
   code change.

Prices were checked against Anthropic's public documentation (see the
dated note below). Anthropic may change them: before using them for
business decisions, confirm at https://docs.claude.com/en/docs/about-claude/pricing
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

__all__ = ["ModelConfig", "MODEL_REGISTRY", "get_model_config", "resolve_model_id"]


@dataclass(frozen=True)
class ModelConfig:
    id: str
    provider: str
    cost_per_million_input: float
    cost_per_million_output: float
    max_context_tokens: int
    tier: str  # "fast" | "balanced" | "quality"


# Logical alias -> configuration. The alias ("haiku", "sonnet", "opus") is
# what the rest of the system uses (ModelRouter, demos); the real `id` sent
# to the API is resolved here and can be overridden with the environment
# variables SINTETICO_MODEL_<ALIAS>, with no code change.
_DEFAULTS: Dict[str, ModelConfig] = {
    # Anthropic values below were read from the official docs page
    # https://platform.claude.com/docs/en/models/overview on 2026-11-01 ("Claude API ID" and "Pricing" rows).
    # Haiku is listed there as "From $0.10 / $0.50": confirm your tier on the pricing page. Re-verify with
    # `python scripts/check_model_registry.py` (queries the Models API with your key) before relying on cost numbers.
    "haiku": ModelConfig("claude-haiku-5-5", "anthropic", 0.10, 0.50, 1_000_000, "fast"),
    "sonnet": ModelConfig("claude-sonnet-5-5", "anthropic", 2.00, 10.00, 1_000_000, "balanced"),
    "opus": ModelConfig("claude-opus-5-5", "anthropic", 4.00, 20.00, 1_000_000, "quality"),
    "gpt-4o": ModelConfig("gpt-4o", "openai", 5.00, 15.00, 128_000, "balanced"),
    "gpt-4-turbo": ModelConfig("gpt-4-turbo", "openai", 10.00, 30.00, 128_000, "quality"),
    "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", "openai", 0.50, 1.50, 16_385, "fast"),
    "local": ModelConfig("llama3-70b", "local", 0.10, 0.10, 8_192, "quality"),
}


def _build_registry() -> Dict[str, ModelConfig]:
    """Build the final registry, indexed by real model `id`, applying
    overrides from environment variables such as
    `SINTETICO_MODEL_HAIKU=<model id>`.
    """
    registry: Dict[str, ModelConfig] = {}
    for alias, cfg in _DEFAULTS.items():
        env_var = f"SINTETICO_MODEL_{alias.upper().replace('-', '_')}"
        model_id = os.environ.get(env_var, cfg.id)
        resolved = ModelConfig(
            model_id,
            cfg.provider,
            cfg.cost_per_million_input,
            cfg.cost_per_million_output,
            cfg.max_context_tokens,
            cfg.tier,
        )
        registry[model_id] = resolved
        registry[alias] = resolved  # also reachable by its short alias
    return registry


MODEL_REGISTRY: Dict[str, ModelConfig] = _build_registry()


def resolve_model_id(alias: str) -> str:
    """Translate a logical alias ("haiku", "sonnet", "opus") into the real
    model id to send to the API, honoring environment overrides."""
    return get_model_config(alias).id


def get_model_config(model_id_or_alias: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_id_or_alias]
    except KeyError as exc:
        opciones = sorted(set(MODEL_REGISTRY.keys()))
        raise ValueError(f"Modelo '{model_id_or_alias}' no registrado. Opciones: {opciones}") from exc
