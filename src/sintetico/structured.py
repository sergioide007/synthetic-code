"""Structured outputs: stop parsing free text, validate against a schema (book Chapter 4).

``AgentStep`` and ``Hypothesis``/``HypothesisBatch`` are the schemas described in the book. ``complete_structured``
asks a provider for JSON that matches a Pydantic model and **validates it before the caller ever sees it**:

* if the provider has a native mechanism (``provider.complete_structured``: forced tool use on Anthropic,
  ``response_format=json_schema`` on OpenAI) the schema is enforced by the API;
* otherwise the schema is put in the prompt, and an invalid answer is sent back to the model together with the
  validation errors (bounded number of attempts) instead of being silently coerced.

Either way the final check is Pydantic, so a bad value (``"high"`` where a 0..1 number is required) fails at the
exact point where it was produced, not three functions later.

Requires ``pydantic>=2`` (``pip install "codigo-sintetico[structured]"``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError('sintetico.structured needs pydantic>=2: pip install "codigo-sintetico[structured]"') from exc

from .adapters import OutputAdapter, OutputParseError
from .providers import LLMProvider, LLMRequest, LLMResponse

__all__ = ["AgentStep", "Hypothesis", "HypothesisBatch", "StructuredOutputError", "StructuredResult", "complete_structured",
           "anthropic_complete_structured", "openai_complete_structured"]

T = TypeVar("T", bound=BaseModel)
_NO_ACTION = {"", "none", "null", "n/a"}


class AgentStep(BaseModel):
    """One ReAct step. ``action is None`` means the agent considers the task finished
    (the final answer travels in ``action_input["result"]``)."""

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(min_length=1, description="Short reasoning for this step")
    action: Optional[str] = Field(default=None, description="Tool name, or null when finished")
    action_input: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool, or {'result': ...} when finished")

    @field_validator("thought")
    @classmethod
    def _thought_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("thought must not be blank")
        return v

    @field_validator("action")
    @classmethod
    def _normalise_no_action(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return None if v.lower() in _NO_ACTION else v

    @property
    def is_final(self) -> bool:
        return self.action is None


class Hypothesis(BaseModel):
    """A candidate reasoning step with the model's own promise estimate. ``score`` must really be a number in [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    thought: str = Field(min_length=1)
    score: float = Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("thought")
    @classmethod
    def _thought_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("thought must not be blank")
        return v


class HypothesisBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypotheses: List[Hypothesis] = Field(min_length=1, max_length=10)


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, attempts: int, last_error: str, last_raw: str) -> None:
        super().__init__(message)
        self.attempts, self.last_error, self.last_raw = attempts, last_error, last_raw


@dataclass
class StructuredResult(Generic[T]):
    value: T
    attempts: int
    usage: Dict[str, int] = field(default_factory=dict)
    raw: str = ""


def _add_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
    for k, v in (usage or {}).items():
        total[k] = total.get(k, 0) + int(v)


def _compact_errors(exc: ValidationError, limit: int = 5) -> str:
    parts = []
    for e in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in e.get("loc", ())) or "<root>"
        parts.append(f"{loc}: {e.get('msg')}")
    more = len(exc.errors()) - limit
    return "; ".join(parts) + (f"; (+{more} more)" if more > 0 else "")


def complete_structured(provider: LLMProvider, request: LLMRequest, model_cls: Type[T], *, max_attempts: int = 3,
                        schema_name: Optional[str] = None) -> StructuredResult[T]:
    """Return a **validated** ``model_cls`` instance, or raise :class:`StructuredOutputError`.

    Provider exceptions (network, auth, rate limits) are NOT swallowed: providers already implement their own retries."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    schema = model_cls.model_json_schema()
    name = schema_name or model_cls.__name__
    native = getattr(provider, "complete_structured", None)
    system = request.system_prompt
    if native is None:
        system = (f"{system}\n\n" if system else "") + (
            "Respond with ONE JSON object and nothing else (no prose, no markdown fences) that validates against this JSON Schema:\n"
            + json.dumps(schema, separators=(",", ":")))
    messages = [dict(m) for m in request.messages]
    usage: Dict[str, int] = {}
    last_error, last_raw = "", ""
    for attempt in range(1, max_attempts + 1):
        req = LLMRequest(system, messages, request.model, request.temperature, request.max_tokens)
        resp: LLMResponse = native(req, schema, name) if native is not None else provider.complete(req)
        _add_usage(usage, resp.usage)
        last_raw = resp.content or ""
        try:
            value = model_cls.model_validate(OutputAdapter.extract_json(last_raw))
            return StructuredResult(value, attempt, usage, last_raw)
        except OutputParseError:
            last_error = "the answer was not valid JSON"
        except ValidationError as exc:
            last_error = _compact_errors(exc)
        messages = messages + [
            {"role": "assistant", "content": last_raw[:2000] or "(empty)"},
            {"role": "user", "content": f"Your previous answer was rejected: {last_error}. Reply again with ONLY valid JSON that matches the schema."},
        ]
    raise StructuredOutputError(f"no valid {name} after {max_attempts} attempts: {last_error}", max_attempts, last_error, last_raw)


# --------------------------------------------------------------------------------------------- native provider hooks
def anthropic_complete_structured(provider: Any, request: LLMRequest, json_schema: Dict[str, Any], name: str) -> LLMResponse:
    """Anthropic: offer one tool whose ``input_schema`` is the schema and ask for exactly one call to it.

    ``tool_choice`` is ``auto`` because the Claude 5.5 models reject a forced choice (``any``/``tool``) with HTTP 400,
    and no ``temperature`` is sent because they reject sampling parameters too. If the model answers in text instead,
    the text is returned and ``complete_structured`` validates it and retries with the error, as for any provider."""
    from .real_providers import LLMProviderError, resolve_model_id

    model_id = resolve_model_id(request.model or provider.default_model)
    tool = {"name": name[:64], "description": "Return the result in exactly this structure.", "input_schema": json_schema}
    system = (f"{request.system_prompt}\n\n" if request.system_prompt else "") + f"Answer by calling the `{tool['name']}` tool exactly once."
    try:
        response = provider.client.messages.create(
            model=model_id, max_tokens=request.max_tokens, system=system,
            messages=request.messages, tools=[tool], tool_choice={"type": "auto"})
    except provider._anthropic.APIError as exc:
        raise LLMProviderError(str(exc)) from exc
    block = next((b for b in response.content if getattr(b, "type", "") == "tool_use"), None)
    content = json.dumps(block.input) if block is not None else "".join(getattr(b, "text", "") for b in response.content)
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return LLMResponse(content=content, usage=usage, model_name=model_id)


def openai_complete_structured(provider: Any, request: LLMRequest, json_schema: Dict[str, Any], name: str) -> LLMResponse:
    """OpenAI: ``response_format=json_schema`` (non-strict, because Pydantic schemas with defaults are not strict-compatible)."""
    from .real_providers import LLMProviderError, openai_sampling_kwargs, resolve_model_id

    model_id = resolve_model_id(request.model or provider.default_model)
    try:
        response = provider.client.chat.completions.create(
            model=model_id, **openai_sampling_kwargs(model_id, request.max_tokens, request.temperature),
            messages=[{"role": "system", "content": request.system_prompt}, *request.messages],
            response_format={"type": "json_schema", "json_schema": {"name": name[:64], "schema": json_schema, "strict": False}})
    except provider._openai.APIError as exc:
        raise LLMProviderError(str(exc)) from exc
    usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    return LLMResponse(content=response.choices[0].message.content or "", usage=usage, model_name=model_id)
