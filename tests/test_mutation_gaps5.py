"""Behaviours of structured.py / reasoning_resilient.py that mutation testing showed were unpinned."""
import json
from types import SimpleNamespace
import pytest
pytest.importorskip("pydantic")
from sintetico.providers import LLMProvider, LLMRequest, LLMResponse
from sintetico.real_providers import RealAnthropicProvider, RealOpenAIProvider, resolve_model_id
from sintetico.reasoning_resilient import ReActAgentResilient, ReActStepRecord, TreeOfThoughtsResilient, text_similarity
from sintetico.structured import AgentStep, HypothesisBatch, StructuredOutputError, complete_structured


class Scripted(LLMProvider):
    def __init__(self, *answers, tokens=(1, 1)):
        self.answers, self.requests, self.tokens = list(answers), [], tokens
    def complete(self, request):
        self.requests.append(request)
        a = self.answers.pop(0)
        return LLMResponse(a if isinstance(a, str) else json.dumps(a), {"input_tokens": self.tokens[0], "output_tokens": self.tokens[1]}, "s")
    @property
    def cost_per_million_tokens(self): return 0.0


def req(model="m"): return LLMRequest("sys", [{"role": "user", "content": "go"}], model, 0.0, 100)
def step(thought, action=None, **inp): return {"thought": thought, "action": action, "action_input": inp}
def bad_batch(n): return {"hypotheses": [{"thought": f"h{i}", "score": "x"} for i in range(n)]}


# ------------------------------------------------------------------------------------------------ structured
@pytest.mark.parametrize("n_bad,suffix", [(1, None), (5, None), (6, "(+1 more)"), (8, "(+3 more)")])
def test_error_feedback_lists_at_most_five_problems_and_says_how_many_more(n_bad, suffix):
    with pytest.raises(StructuredOutputError) as e:
        complete_structured(Scripted(bad_batch(n_bad)), req(), HypothesisBatch, max_attempts=1)
    err = e.value.last_error
    assert err.count("hypotheses.") == min(n_bad, 5) and (suffix in err if suffix else "more)" not in err)


def test_default_is_three_attempts_and_a_long_invalid_answer_is_truncated_in_the_echo():
    p = Scripted("a" * 5000, "", "c")
    with pytest.raises(StructuredOutputError) as e:
        complete_structured(p, req(), AgentStep)                                          # default max_attempts
    assert e.value.attempts == 3 and len(p.requests) == 3
    assert len(p.requests[1].messages[1]["content"]) == 2000 and p.requests[2].messages[3]["content"] == "(empty)"


def test_native_hooks_use_the_provider_default_model_and_cap_names_at_64_chars():
    long_name = "N" * 100
    seen = {}
    def a_create(**kw):
        seen["a"] = kw
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={})], usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    a = object.__new__(RealAnthropicProvider); a.client = SimpleNamespace(messages=SimpleNamespace(create=a_create))
    a._anthropic = SimpleNamespace(APIError=Exception); a.default_model = "sonnet"
    a.complete_structured(LLMRequest("s", [], None, 0.0, 10), {}, long_name)
    assert seen["a"]["model"] == resolve_model_id("sonnet") and seen["a"]["tools"][0]["name"] == "N" * 64 and seen["a"]["tool_choice"]["name"] == "N" * 64
    def o_create(**kw):
        seen["o"] = kw
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    o = object.__new__(RealOpenAIProvider); o.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=o_create)))
    o._openai = SimpleNamespace(APIError=Exception); o.default_model = "gpt-4o"
    out = o.complete_structured(LLMRequest("s", [], None, 0.0, 10), {}, long_name)
    assert seen["o"]["model"] == resolve_model_id("gpt-4o") and seen["o"]["response_format"]["json_schema"]["name"] == "N" * 64 and out.content == ""


# ------------------------------------------------------------------------------------------- ReAct defaults / bounds
def test_documented_defaults():
    a = ReActAgentResilient(Scripted(), {})
    assert (a.max_cycles, a.similarity_threshold, a.repeat_limit, a.max_attempts, a.observation_limit, a.max_total_tokens, a.model) == (8, 0.85, 2, 3, 2000, None, None)
    t = TreeOfThoughtsResilient(Scripted())
    assert (t.branching, t.max_attempts, t.model) == (3, 3, None)


def test_request_parameters_sent_to_the_model():
    p = Scripted(step("x", None, result="r"))
    ReActAgentResilient(p, {}, model="mm").run("q")
    r = p.requests[0]
    assert (r.model, r.temperature, r.max_tokens) == ("mm", 0.0, 1024) and r.messages[0]["content"] == "Task: q"
    tp = Scripted({"hypotheses": [{"thought": "a", "score": 0.5}]})
    TreeOfThoughtsResilient(tp, model="tm").solve_detailed("p", depth=1)
    assert (tp.requests[0].model, tp.requests[0].temperature, tp.requests[0].max_tokens) == ("tm", 0.7, 1024)


def test_the_similarity_threshold_is_inclusive():
    thoughts = ["checking inventory for item 4411", "checking inventory for item 4412", "checking inventory for item 4413"]
    s = min(text_similarity(a, b) for a, b in ((thoughts[0], thoughts[1]), (thoughts[1], thoughts[2]), (thoughts[0], thoughts[2])))
    recs = [ReActStepRecord(t, "x", {"i": i}, str(i)) for i, t in enumerate(thoughts)]
    assert "semantic loop" in ReActAgentResilient(Scripted(), {}, similarity_threshold=s)._should_stop(recs)
    assert ReActAgentResilient(Scripted(), {}, similarity_threshold=s + 1e-6)._should_stop(recs) is None


def test_token_budget_is_exclusive_at_the_limit():
    mk = lambda: Scripted(*[step(f"{'abcdefgh'[i] * (i + 5)} unique {i * 'z'}", "t", n=i) for i in range(3)], step("done", None, result="r"), tokens=(300, 200))
    ok = ReActAgentResilient(mk(), {"t": lambda n: n}, max_total_tokens=500).run("q")                  # 1 step = 500 tokens: not MORE than 500
    assert ok.status == "budget" and len(ok.steps) == 2                                               # stops after the 2nd step (1000 > 500), not the 1st
    assert ReActAgentResilient(mk(), {"t": lambda n: n}, max_total_tokens=5000).run("q").status == "success"


def test_the_circuit_breaker_is_the_last_line_of_defence_against_tool_flooding():
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()
    thoughts = [f"{words[i % 26]} {words[(i * 7 + 3) % 26]} {words[(i * 11 + 5) % 26]} number {i * 97}" for i in range(40)]
    p = Scripted(*[step(t, "search", q=f"query-{i}") for i, t in enumerate(thoughts)], step("done", None, result="r"))
    r = ReActAgentResilient(p, {"search": lambda q: f"result for {q}"}, max_cycles=40).run("q")
    assert r.status == "stopped" and r.stop_reason.startswith("circuit breaker open: Flooding: search 30/30") and len(r.steps) == 30      # none of the other two layers fired


# ----------------------------------------------------------------------------------------------------- Tree of Thoughts
class Fixed(LLMProvider):
    def __init__(self, n_levels=6): self.requests = []
    def complete(self, request):
        self.requests.append(request)
        d = request.messages[0]["content"].count("->")
        return LLMResponse(json.dumps({"hypotheses": [{"thought": f"L{d}a", "score": 0.9}, {"thought": f"L{d}b", "score": 0.8}, {"thought": f"L{d}c", "score": 0.1}]}), {}, "f")
    @property
    def cost_per_million_tokens(self): return 0.0


def test_default_depth_beam_and_branching_clamp():
    prov = Fixed()
    res = TreeOfThoughtsResilient(prov).solve_detailed("p")                                 # depth=3, beam_width=2 by default
    assert len(res.path) == 4 and res.nodes_expanded == 1 + 2 + 2 and len(prov.requests) == 5
    one = TreeOfThoughtsResilient(Fixed(), branching=0)
    assert one.branching == 1 and "exactly 1" in one.SYSTEM.format(n=one.branching)
    assert len(one.solve_detailed("p", depth=1).path) == 2


def test_failed_attempts_count_as_model_calls():
    res = TreeOfThoughtsResilient(Scripted("junk", "junk"), max_attempts=2).solve_detailed("p")
    assert res.model_calls == 2 and res.nodes_expanded == 0 and res.path == ["Start"] and res.best_score == 1.0
