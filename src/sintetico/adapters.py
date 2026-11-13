"""Prompt/output adapters: being backend-agnostic also means adapting *behavior* (book Chapters 19-20)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["OutputParseError", "OutputAdapter", "PromptAdapter", "CapabilityProfile", "PROFILES"]


class OutputParseError(ValueError):
    pass


class OutputAdapter:
    """Normalize raw model output: strip reasoning tags / markdown fences, extract JSON."""

    _TAGS = re.compile(r"<(thinking|think|reasoning)>.*?</\1>", re.S | re.I)
    MAX_CHARS = 1_000_000
    MAX_ATTEMPTS = 200
    _FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.S)

    @classmethod
    def clean(cls, raw: str) -> str:
        return cls._TAGS.sub("", raw).strip()

    @classmethod
    def extract_json(cls, raw: str) -> Any:
        text = cls.clean(raw[: cls.MAX_CHARS])
        candidates = [m.group(1) for m in cls._FENCE.finditer(text)] + [text]
        decoder = json.JSONDecoder()
        for cand in candidates:
            tries = 0
            for i, ch in enumerate(cand):
                if ch in "{[":
                    tries += 1
                    if tries > cls.MAX_ATTEMPTS:      # pathological input: stop instead of going quadratic
                        break
                    try:
                        obj, _ = decoder.raw_decode(cand[i:])
                        return obj
                    except (json.JSONDecodeError, RecursionError):   # deep nesting must not crash the caller
                        continue
        raise OutputParseError("No valid JSON found in model output")


class PromptAdapter:
    """Rephrase the *same* instruction in the style each model family follows best."""

    FAMILIES = ("claude", "gpt", "gemini", "local")

    @classmethod
    def adapt(cls, prompt: str, family: str, format_example: Optional[str] = None) -> str:
        family = family.lower()
        if family not in cls.FAMILIES:
            raise ValueError(f"Unknown family '{family}'. Use one of {cls.FAMILIES}")
        if family == "claude":
            return f"<instructions>\n{prompt.strip()}\n</instructions>"
        if family == "gpt":
            out = prompt.strip()
            if format_example:
                out += f"\n\nRespond using exactly this format:\n{format_example}"
            return out
        if family == "gemini":
            lines = [ln.strip() for ln in re.split(r"[.\n]+", prompt) if ln.strip()]
            return "Follow these criteria:\n" + "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
        return prompt.strip()


@dataclass(frozen=True)
class CapabilityProfile:
    name: str
    recommended: List[str]
    avoid: List[str] = field(default_factory=list)
    note: str = ""


# Aliases refer to sintetico.model_registry. Treat as defaults to revisit each quarter.
PROFILES: Dict[str, CapabilityProfile] = {
    "complex_reasoning": CapabilityProfile("complex_reasoning", ["opus", "gpt-6-astra"], ["gpt-6-luna", "local"]),
    "simple_fast": CapabilityProfile("simple_fast", ["haiku", "gpt-6-luna"], ["opus", "gpt-6-astra"], "cheap tier is enough"),
    "strict_json": CapabilityProfile("strict_json", ["sonnet", "gpt-6.1-sol"], ["local"], "always validate + retry"),
}
