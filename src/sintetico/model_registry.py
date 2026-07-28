"""
sintetico.model_registry — Catálogo de modelos y su coste por millón de tokens.

Los identificadores y precios de los modelos de un proveedor de LLM
cambian con frecuencia (nuevas versiones, descuentos de lanzamiento,
retirada de modelos legacy). Por eso este catálogo:

1. Vive en un único lugar, en vez de estar disperso como literales de
   texto en varios ficheros de demo (como ocurría en la v1 del código,
   donde "claude-haiku-4-5" aparecía hardcodeado en cuatro sitios
   distintos con identificadores inconsistentes y, además, sin el sufijo
   de fecha que exige la API real → toda llamada real fallaba con
   "model not found").
2. Es sobreescribible por variables de entorno, para que el ejemplo del
   libro siga funcionando cuando Anthropic publique nuevos modelos sin
   tener que tocar código.

Precios verificados en la documentación pública de Anthropic (julio 2026).
Anthropic puede cambiarlos: antes de usarlos para decisiones de negocio,
confirma en https://docs.claude.com/en/docs/about-claude/pricing
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


# Alias lógico -> configuración. El alias ("haiku", "sonnet", "opus") es lo
# que usa el resto del sistema (ModelRouter, demos); el `id` real que se
# envía a la API se resuelve aquí y puede sobreescribirse con variables de
# entorno SINTETICO_MODEL_<ALIAS> sin tocar código.
_DEFAULTS: Dict[str, ModelConfig] = {
    "haiku": ModelConfig("claude-haiku-4-5-20251001", "anthropic", 1.00, 5.00, 200_000, "fast"),
    "sonnet": ModelConfig("claude-sonnet-5", "anthropic", 3.00, 15.00, 1_000_000, "balanced"),
    "opus": ModelConfig("claude-opus-4-8", "anthropic", 5.00, 25.00, 1_000_000, "quality"),
    "gpt-4o": ModelConfig("gpt-4o", "openai", 5.00, 15.00, 128_000, "balanced"),
    "gpt-4-turbo": ModelConfig("gpt-4-turbo", "openai", 10.00, 30.00, 128_000, "quality"),
    "gpt-3.5-turbo": ModelConfig("gpt-3.5-turbo", "openai", 0.50, 1.50, 16_385, "fast"),
    "local": ModelConfig("llama3-70b", "local", 0.10, 0.10, 8_192, "quality"),
}


def _build_registry() -> Dict[str, ModelConfig]:
    """Construye el registro final, indexado por `id` real de modelo,
    aplicando overrides desde variables de entorno del tipo
    `SINTETICO_MODEL_HAIKU=claude-haiku-4-5-20251001`.
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
        registry[alias] = resolved  # también accesible por alias corto
    return registry


MODEL_REGISTRY: Dict[str, ModelConfig] = _build_registry()


def resolve_model_id(alias: str) -> str:
    """Traduce un alias lógico ("haiku", "sonnet", "opus") al id real de
    modelo que se debe enviar a la API, respetando overrides de entorno."""
    return get_model_config(alias).id


def get_model_config(model_id_or_alias: str) -> ModelConfig:
    try:
        return MODEL_REGISTRY[model_id_or_alias]
    except KeyError as exc:
        opciones = sorted(set(MODEL_REGISTRY.keys()))
        raise ValueError(f"Modelo '{model_id_or_alias}' no registrado. Opciones: {opciones}") from exc
