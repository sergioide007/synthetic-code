import json, os, subprocess, sys
import pytest
pytest.importorskip("pydantic")
from sintetico.providers import LLMProvider, LLMResponse
from sintetico.reasoning import ThoughtNode, TreeOfThoughts
from sintetico.reasoning_resilient import ReActAgentResilient, ReActStepRecord, TreeOfThoughtsResilient, text_similarity


class Scripted(LLMProvider):
    def __init__(self, *answers, tokens=(10, 5)):
        self.answers, self.requests, self.tokens = list(answers), [], tokens
    def complete(self, request):
        self.requests.append(request)
        a = self.answers.pop(0)
        return LLMResponse(a if isinstance(a, str) else json.dumps(a), {"input_tokens": self.tokens[0], "output_tokens": self.tokens[1]}, "s")
    @property
    def cost_per_million_tokens(self): return 0.0


def step(thought, action=None, **inp): return {"thought": thought, "action": action, "action_input": inp}
def final(text="done"): return step("I have the answer", None, result=text)


def search(query: str) -> str:
    """Look something up."""
    return f"found:{query}"


# ----------------------------------------------------------------------------------------------- ReAct
def test_happy_path_feeds_the_observation_back_and_returns_the_final_answer():
    p = Scripted(step("need data", "search", query="kafka"), final("42"))
    r = ReActAgentResilient(p, {"search": search}, model="m").run("what?")
    assert (r.status, r.answer, len(r.steps), r.stop_reason) == ("success", "42", 2, None)
    assert r.steps[0].observation == "found:kafka" and r.structured_attempts == 2 and r.usage == {"input_tokens": 20, "output_tokens": 10}
    second = p.requests[1].messages
    assert [m["role"] for m in second] == ["user", "assistant", "user"] and second[-1]["content"] == "Observation: found:kafka"
    assert json.loads(second[1]["content"])["action"] == "search"                                      # the step the model produced
    assert "- search: Look something up." in p.requests[0].system_prompt and p.requests[0].temperature == 0.0 and p.requests[0].model == "m"


def test_final_answer_falls_back_to_the_thought_when_no_result_is_given():
    r = ReActAgentResilient(Scripted(step("The capital is Paris", None)), {}).run("q")
    assert r.status == "success" and r.answer == "The capital is Paris"


def test_a_model_that_wraps_json_in_prose_or_fences_no_longer_breaks_the_agent():
    wrapped = "Here you go!\n```json\n" + json.dumps(final("ok")) + "\n```\nHope that helps."
    assert ReActAgentResilient(Scripted(wrapped), {}).run("q").answer == "ok"


def test_malformed_step_is_corrected_by_re_asking_not_by_guessing():
    p = Scripted("Thought: I should search\nAction: search", step("retry ok", None, result="y"))
    r = ReActAgentResilient(p, {"search": search}).run("q")
    assert r.status == "success" and r.structured_attempts == 2 and "not valid JSON" in p.requests[1].messages[-1]["content"]


def test_persistently_invalid_output_ends_as_invalid_output_not_an_exception():
    r = ReActAgentResilient(Scripted("x", "y", "z"), {}, max_attempts=3).run("q")
    assert r.status == "invalid_output" and "no valid AgentStep after 3 attempts" in r.stop_reason and r.answer is None


def test_unknown_tool_and_tool_exceptions_become_observations_the_agent_can_recover_from():
    def boom(x): raise RuntimeError("db down")
    def strict(a, b): return a + b
    p = Scripted(step("first try an unknown tool", "nope"), step("now a tool that crashes", "boom", x=1), step("then wrong arguments entirely", "strict", a=1), final())
    r = ReActAgentResilient(p, {"boom": boom, "strict": strict, "search": search}).run("q")
    obs = [s.observation for s in r.steps[:3]]
    assert "unknown tool 'nope'. Available: boom, search, strict" in obs[0] and obs[1] == "ERROR: RuntimeError: db down" and "bad arguments for 'strict'" in obs[2]
    assert r.status == "success"


def test_tool_output_is_truncated_and_structured_results_are_json_encoded():
    r = ReActAgentResilient(Scripted(step("t", "big"), step("t", "obj"), final()), {"big": lambda: "x" * 5000, "obj": lambda: {"a": [1, 2]}}, observation_limit=100).run("q")
    assert len(r.steps[0].observation) == 100 and r.steps[1].observation == '{"a": [1, 2]}'


def test_same_tool_same_args_same_result_is_stopped_on_the_third_identical_call():
    calls = []
    def stuck(q): calls.append(q); return "same"
    thoughts = ["checking inventory", "look at the stock levels", "try fetching the numbers once more"]        # dissimilar on purpose
    p = Scripted(*[step(t, "stuck", q="a") for t in thoughts], final())
    r = ReActAgentResilient(p, {"stuck": stuck}).run("q")
    assert r.status == "stopped" and "repeated 3 times" in r.stop_reason and len(calls) == 3 and len(p.requests) == 3


def test_repeating_a_call_is_fine_while_the_state_keeps_changing():
    counter = iter(range(100))
    p = Scripted(*[step(f"step {i} distinct wording {'x' * i * 7}", "tick") for i in range(4)], final())
    r = ReActAgentResilient(p, {"tick": lambda: next(counter)}).run("q")
    assert r.status == "success" and len(r.steps) == 5


def test_different_arguments_reset_the_tool_signature_streak():
    p = Scripted(step("alpha one", "search", query="a"), step("beta two is different", "search", query="b"),
                 step("gamma three also different", "search", query="a"), final())
    assert ReActAgentResilient(p, {"search": lambda query: "same"}).run("q").status == "success"


def test_semantic_loop_is_caught_even_when_ids_and_timestamps_change_the_text():
    p = Scripted(*[step(f"Still checking the payment gateway status at 10:0{i} txn-{1000 + i}", "search", query=str(i)) for i in range(5)], final())
    r = ReActAgentResilient(p, {"search": lambda query: f"r{query}"}).run("q")
    assert r.status == "stopped" and "semantic loop" in r.stop_reason and len(r.steps) == 3


def test_should_stop_layers_directly():
    agent = ReActAgentResilient(Scripted(), {})
    rec = lambda t, a="x", i=None, o="o": ReActStepRecord(t, a, i or {}, o)
    assert agent._should_stop([]) is None and agent._should_stop([rec("a")]) is None and agent._should_stop([rec("a"), rec("b")]) is None
    assert "semantic" in agent._should_stop([rec("same thought"), rec("same thought"), rec("same thought")])
    assert agent._should_stop([rec("one thing", o="1"), rec("something else entirely", o="2"), rec("a third unrelated idea", o="3")]) is None
    x = [rec(f"distinct {c * 9}", i={"k": 1}) for c in "abc"]
    assert "repeated 3 times" in agent._should_stop(x) and agent._should_stop(x[:2]) is None
    assert agent._should_stop([rec("p"), rec("q"), ReActStepRecord("final", None, {}, "")]) is None            # a final step is never a tool repeat


def test_similarity_helper_bounds():
    assert text_similarity("abc", "abc") == 1.0 and text_similarity("abc", "xyz") < 0.2 and 0.0 <= text_similarity("", "abc") <= 1.0
    assert text_similarity("checking txn-1001", "checking txn-1002") > 0.85


def test_token_budget_and_max_cycles_are_enforced():
    p = Scripted(*[step(f"{'abcdefghij'[i]}" * 20, "search", query=str(i)) for i in range(3)], tokens=(600, 600))
    r = ReActAgentResilient(p, {"search": search}, max_total_tokens=1000).run("q")
    assert r.status == "budget" and len(r.steps) == 1
    p2 = Scripted(*[step("p" * (i + 1) * 9, "search", query=str(i)) for i in range(3)])
    r2 = ReActAgentResilient(p2, {"search": search}, max_cycles=2).run("q")
    assert r2.status == "max_cycles" and len(r2.steps) == 2 and "2 cycles" in r2.stop_reason


# -------------------------------------------------------------------------------------- Tree of Thoughts
def batch(*items): return {"hypotheses": [{"thought": t, "score": s} for t, s in items]}


class TreeProvider(LLMProvider):
    """Answers by looking at the 'Reasoning so far' line of the prompt, like a model would."""
    def __init__(self, table): self.table, self.requests = table, []
    def complete(self, request):
        self.requests.append(request)
        so_far = request.messages[0]["content"].split("Reasoning so far: ")[1].split("\n")[0]
        ans = self.table[so_far]
        return LLMResponse(ans if isinstance(ans, str) else json.dumps(ans), {"input_tokens": 1, "output_tokens": 1}, "t")
    @property
    def cost_per_million_tokens(self): return 0.0


def test_beam_search_follows_the_best_scores_and_returns_the_path():
    table = {"Start": batch(("A", 0.4), ("B", 0.9), ("C", 0.1)), "Start -> B": batch(("B1", 0.5), ("B2", 0.95)), "Start -> A": batch(("A1", 0.99))}
    tot = TreeOfThoughtsResilient(TreeProvider(table), branching=3)
    res = tot.solve_detailed("problem", depth=2, beam_width=2)
    assert res.best_score == 0.99 and res.path == ["Start", "A", "A1"]                    # beam_width=2 keeps B(0.9) and A(0.4); A1(0.99) beats B2(0.95)
    assert res.nodes_expanded == 3 and res.model_calls == 3 and res.failures == [] and tot.last_result is res
    assert tot.solve("problem", 2, 2) == "Start -> A -> A1" and isinstance(tot, TreeOfThoughts)


def test_beam_width_prunes_branches_before_expanding_them():
    table = {"Start": batch(("A", 0.4), ("B", 0.9), ("C", 0.1)), "Start -> B": batch(("B1", 0.5))}
    prov = TreeProvider(table)
    TreeOfThoughtsResilient(prov).solve_detailed("p", depth=2, beam_width=1)
    assert len(prov.requests) == 2                                                          # only the best branch (B) is expanded


def test_a_text_score_is_rejected_at_the_source_and_the_model_is_asked_again():
    table = {"Start": [batch(("A", "high")), batch(("A", 0.7))]}
    class Seq(TreeProvider):
        def complete(self, request):
            self.requests.append(request)
            return LLMResponse(json.dumps(self.table["Start"][len(self.requests) - 1]), {}, "t")
    prov = Seq(table)
    res = TreeOfThoughtsResilient(prov).solve_detailed("p", depth=1)
    assert res.path == ["Start", "A"] and res.best_score == 0.7 and res.model_calls == 2
    assert "hypotheses.0.score" in prov.requests[1].messages[-1]["content"]


def test_a_branch_that_cannot_be_generated_is_recorded_and_the_search_continues():
    table = {"Start": batch(("A", 0.9), ("B", 0.8)), "Start -> A": "garbage", "Start -> B": batch(("B1", 0.7))}
    res = TreeOfThoughtsResilient(TreeProvider(table), max_attempts=2).solve_detailed("p", depth=2, beam_width=2)
    assert res.path == ["Start", "B", "B1"] and len(res.failures) == 1 and "'A'" in res.failures[0] and res.nodes_expanded == 2
    all_bad = TreeOfThoughtsResilient(TreeProvider({"Start": "nope"}), max_attempts=1).solve_detailed("p")
    assert all_bad.path == ["Start"] and all_bad.nodes_expanded == 0 and len(all_bad.failures) == 1


def test_duplicates_are_dropped_branching_is_enforced_and_an_external_judge_can_override_scores():
    table = {"Start": batch(("Same", 0.9), ("same", 0.8), ("Other", 0.1), ("Third", 0.2), ("Fourth", 0.3))}
    prov = TreeProvider(table)
    tot = TreeOfThoughtsResilient(prov, branching=2)
    assert tot._generate_children(ThoughtNode("Start", 1.0), "p") == ["Same", "Other"]            # dedupe (case-insensitive) then cap
    assert "exactly 2" in prov.requests[0].system_prompt
    judged = TreeOfThoughtsResilient(TreeProvider(table), branching=5, evaluator=lambda thought, problem: 1.0 if thought == "Fourth" else 0.0)
    assert judged.solve_detailed("p", depth=1, beam_width=1).path == ["Start", "Fourth"]


def test_default_evaluator_of_the_base_class_is_reproducible_across_processes():
    code = "from sintetico.reasoning import TreeOfThoughts as T;print(T._default_evaluator('x','y'),T().solve('p'))"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": s}).stdout for s in ("1", "2", "3", "random")}
    assert len(outs) == 1, outs
