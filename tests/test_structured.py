import json
from types import SimpleNamespace
import pytest
pytest.importorskip("pydantic")
from sintetico.providers import LLMProvider, LLMRequest, LLMResponse
from sintetico.real_providers import LLMProviderError, RealAnthropicProvider, RealOpenAIProvider
from sintetico.structured import (AgentStep, Hypothesis, HypothesisBatch, StructuredOutputError, anthropic_complete_structured,
                                  complete_structured, openai_complete_structured)


class Scripted(LLMProvider):
    """Returns the next canned answer; records every request it receives."""
    def __init__(self, *answers, usage=(10, 5)):
        self.answers, self.requests, self.usage = list(answers), [], usage
    def complete(self, request):
        self.requests.append(request)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return LLMResponse(a if isinstance(a, str) else json.dumps(a), {"input_tokens": self.usage[0], "output_tokens": self.usage[1]}, "scripted")
    @property
    def cost_per_million_tokens(self): return 0.0


def req(model="m"): return LLMRequest("sys", [{"role": "user", "content": "go"}], model, 0.0, 100)


# ------------------------------------------------------------------------------------------------ schemas
def test_agent_step_normalises_no_action_and_rejects_blank_thought_and_extras():
    for none in ("none", "None", " NULL ", "", "n/a", None):
        assert AgentStep(thought="t", action=none).action is None and AgentStep(thought="t", action=none).is_final
    s = AgentStep(thought="  look it up  ", action=" search ", action_input={"q": 1})
    assert (s.thought, s.action, s.is_final) == ("look it up", "search", False)
    for bad in ({"thought": "   "}, {"thought": ""}, {"thought": "t", "surprise": 1}, {"thought": "t", "action_input": "not-a-dict"}):
        with pytest.raises(Exception):
            AgentStep.model_validate(bad)


@pytest.mark.parametrize("score,ok", [(0, True), (0.0, True), (1, True), (1.0, True), (0.5, True), (-0.01, False), (1.01, False),
                                      ("high", False), ("0.7", False), (True, False), (float("nan"), False), (float("inf"), False), (None, False)])
def test_hypothesis_score_must_be_a_real_number_between_zero_and_one(score, ok):
    if ok:
        assert Hypothesis(thought="t", score=score).score == float(score)
    else:
        with pytest.raises(Exception):
            Hypothesis.model_validate({"thought": "t", "score": score})


def test_batch_size_bounds():
    one = {"thought": "a", "score": 0.1}
    assert len(HypothesisBatch.model_validate({"hypotheses": [one]}).hypotheses) == 1
    for bad in ({"hypotheses": []}, {"hypotheses": [one] * 11}, {}):
        with pytest.raises(Exception):
            HypothesisBatch.model_validate(bad)


# ----------------------------------------------------------------------------------- complete_structured
def test_valid_first_try_with_fences_and_prose_and_schema_in_prompt():
    p = Scripted('Sure!\n```json\n{"thought": "hi", "action": null}\n```')
    r = complete_structured(p, req(), AgentStep)
    assert r.value.thought == "hi" and r.attempts == 1 and r.usage == {"input_tokens": 10, "output_tokens": 5}
    assert '"thought"' in p.requests[0].system_prompt and p.requests[0].system_prompt.startswith("sys")      # schema appended to the system prompt
    assert (p.requests[0].model, p.requests[0].temperature, p.requests[0].max_tokens) == ("m", 0.0, 100)


def test_invalid_answer_is_sent_back_with_the_validation_errors_then_succeeds():
    bad = {"hypotheses": [{"thought": "a", "score": "high"}]}
    good = {"hypotheses": [{"thought": "a", "score": 0.9}]}
    p = Scripted(bad, good)
    r = complete_structured(p, req(), HypothesisBatch)
    assert r.attempts == 2 and r.value.hypotheses[0].score == 0.9 and r.usage == {"input_tokens": 20, "output_tokens": 10}
    retry = p.requests[1].messages
    assert retry[0] == {"role": "user", "content": "go"}                                                   # original conversation preserved
    assert retry[1]["role"] == "assistant" and "high" in retry[1]["content"]                                 # what the model said
    assert retry[2]["role"] == "user" and "hypotheses.0.score" in retry[2]["content"]                         # exactly where it was wrong
    assert len(p.requests[0].messages) == 1                                                                   # the caller's list was not mutated


def test_non_json_then_json():
    p = Scripted("I think the answer is yes", {"thought": "ok"})
    r = complete_structured(p, req(), AgentStep)
    assert r.attempts == 2 and "not valid JSON" in p.requests[1].messages[-1]["content"]


def test_gives_up_after_max_attempts_with_a_precise_error():
    p = Scripted("nope", {"thought": ""}, {"thought": "x", "bogus": 1})
    with pytest.raises(StructuredOutputError) as e:
        complete_structured(p, req(), AgentStep, max_attempts=3)
    assert e.value.attempts == 3 and "bogus" in e.value.last_error and e.value.last_raw and len(p.requests) == 3
    with pytest.raises(ValueError):
        complete_structured(p, req(), AgentStep, max_attempts=0)


def test_provider_errors_are_not_swallowed():
    with pytest.raises(LLMProviderError):
        complete_structured(Scripted(LLMProviderError("rate limited")), req(), AgentStep)


def test_native_provider_path_does_not_pollute_the_prompt():
    class Native(Scripted):
        def complete_structured(self, request, schema, name):
            self.requests.append((request, schema, name))
            return LLMResponse(json.dumps({"thought": "native"}), {"input_tokens": 3, "output_tokens": 2}, "n")
    p = Native()
    r = complete_structured(p, req(), AgentStep, schema_name="step")
    request, schema, name = p.requests[0]
    assert r.value.thought == "native" and request.system_prompt == "sys" and name == "step" and schema["title"] == "AgentStep"


# ------------------------------------------------------------------------- native SDK hooks (fake SDK clients)
class _APIError(Exception): ...


def _anthropic_provider(response=None, error=None):
    calls = []
    def create(**kw):
        calls.append(kw)
        if error: raise error
        return response
    p = object.__new__(RealAnthropicProvider)
    p.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    p._anthropic = SimpleNamespace(APIError=_APIError)
    p.default_model = "sonnet"
    return p, calls


def test_anthropic_native_forces_a_single_tool_call_and_returns_its_input_as_json():
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="ignored"), SimpleNamespace(type="tool_use", input={"thought": "t", "action": None})],
                           usage=SimpleNamespace(input_tokens=11, output_tokens=7))
    p, calls = _anthropic_provider(resp)
    out = p.complete_structured(LLMRequest("sys", [{"role": "user", "content": "x"}], "sonnet", 0.0, 50), {"type": "object"}, "AgentStep")
    kw = calls[0]
    assert kw["tools"] == [{"name": "AgentStep", "description": "Return the result in exactly this structure.", "input_schema": {"type": "object"}}]
    assert kw["tool_choice"] == {"type": "tool", "name": "AgentStep"} and kw["system"] == "sys" and kw["max_tokens"] == 50
    assert json.loads(out.content) == {"thought": "t", "action": None} and out.usage == {"input_tokens": 11, "output_tokens": 7}
    assert AgentStep.model_validate_json(out.content).is_final


def test_anthropic_native_falls_back_to_text_and_maps_sdk_errors():
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text='{"thought": "plain"}')], usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    p, _ = _anthropic_provider(resp)
    assert p.complete_structured(req("sonnet"), {}, "X").content == '{"thought": "plain"}'
    p, _ = _anthropic_provider(error=_APIError("boom"))
    with pytest.raises(LLMProviderError):
        anthropic_complete_structured(p, req("sonnet"), {}, "X")


def test_openai_native_sends_a_json_schema_response_format():
    calls = []
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"thought": "o"}'))], usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2))
    p = object.__new__(RealOpenAIProvider)
    p.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    p._openai = SimpleNamespace(APIError=_APIError)
    p.default_model = "gpt-4o"
    out = p.complete_structured(req("gpt-4o"), {"type": "object"}, "Step")
    kw = calls[0]
    assert kw["response_format"] == {"type": "json_schema", "json_schema": {"name": "Step", "schema": {"type": "object"}, "strict": False}}
    assert kw["messages"][0] == {"role": "system", "content": "sys"} and out.usage == {"input_tokens": 4, "output_tokens": 2}
    p.client.chat.completions.create = lambda **kw: (_ for _ in ()).throw(_APIError("x"))
    with pytest.raises(LLMProviderError):
        openai_complete_structured(p, req("gpt-4o"), {}, "Step")
