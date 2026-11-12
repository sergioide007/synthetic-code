"""Resilient reasoning agents built on structured outputs (book Chapter 4).

* :class:`ReActAgentResilient` -- ReAct loop whose every step is a schema-validated :class:`AgentStep`. There is no
  ``Thought:/Action:`` regex to break. Loop protection is layered: thought similarity, **tool signature** (same tool +
  same arguments + same observation), and the ``AgentCircuitBreaker``.
* :class:`TreeOfThoughtsResilient` -- beam search whose hypotheses arrive as a validated :class:`HypothesisBatch`
  (score is a real number in [0, 1], not text parsed inside a silent ``try/except``).

Both take an :class:`~sintetico.providers.LLMProvider`, so they run offline with a scripted provider and online with
Anthropic / OpenAI / a local server unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .providers import LLMProvider, LLMRequest
from .reasoning import ThoughtNode, TreeOfThoughts
from .resilience import AgentCircuitBreaker
from .structured import AgentStep, Hypothesis, HypothesisBatch, StructuredOutputError, complete_structured

__all__ = ["ReActResult", "ReActStepRecord", "ReActAgentResilient", "ToTResult", "TreeOfThoughtsResilient", "text_similarity"]


def text_similarity(a: str, b: str) -> float:
    """Normalised similarity in [0, 1]: Levenshtein (rapidfuzz) when installed, ``difflib`` otherwise."""
    if a == b:
        return 1.0
    try:
        from rapidfuzz.distance import Levenshtein

        return float(Levenshtein.normalized_similarity(a, b))
    except ImportError:
        import difflib

        return difflib.SequenceMatcher(None, a, b).ratio()


@dataclass
class ReActStepRecord:
    thought: str
    action: Optional[str]
    action_input: Dict[str, Any]
    observation: str = ""


@dataclass
class ReActResult:
    status: str                                   # success | stopped | max_cycles | invalid_output | budget
    answer: Optional[str]
    steps: List[ReActStepRecord] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    structured_attempts: int = 0                  # total model calls, including re-asks after invalid JSON


def _signature(action: Optional[str], args: Mapping[str, Any]) -> str:
    return f"{action}:{json.dumps(args, sort_keys=True, default=str)}"


class ReActAgentResilient:
    SYSTEM = ("You solve tasks step by step using tools. On every turn answer with ONE step: your short thought, the tool to call "
              "(`action`) and its arguments (`action_input`). When you can answer, set `action` to null and put the final answer in "
              "`action_input.result`.\nAvailable tools:\n{tools}")

    def __init__(self, provider: LLMProvider, tools: Mapping[str, Callable[..., Any]], *, max_cycles: int = 8,
                 similarity_threshold: float = 0.85, repeat_limit: int = 2, max_attempts: int = 3, model: Optional[str] = None,
                 max_total_tokens: Optional[int] = None, observation_limit: int = 2000) -> None:
        self.provider, self.tools = provider, dict(tools)
        self.max_cycles, self.similarity_threshold = max_cycles, similarity_threshold
        self.repeat_limit, self.max_attempts, self.model = repeat_limit, max_attempts, model
        self.max_total_tokens, self.observation_limit = max_total_tokens, observation_limit

    # ------------------------------------------------------------------ layered loop protection
    def _should_stop(self, steps: List[ReActStepRecord], breaker: Optional[AgentCircuitBreaker] = None) -> Optional[str]:
        """Return a reason to abort, or None. Three independent layers; any one is enough."""
        if len(steps) >= 3:
            a, b, c = (s.thought for s in steps[-3:])
            if min(text_similarity(a, b), text_similarity(b, c), text_similarity(a, c)) >= self.similarity_threshold:
                return f"semantic loop: last 3 thoughts are >= {self.similarity_threshold:.0%} similar"
        # tool signature: same tool, same arguments AND same observation = no measurable state change
        last = steps[-1] if steps else None
        if last is not None and last.action is not None:
            sig = (_signature(last.action, last.action_input), last.observation)
            streak = 0
            for s in reversed(steps):
                if s.action is not None and (_signature(s.action, s.action_input), s.observation) == sig:
                    streak += 1
                else:
                    break
            if streak > self.repeat_limit:
                return f"same tool call repeated {streak} times with no change in the result: {last.action}"
        if breaker is not None and breaker.is_open():
            return f"circuit breaker open: {breaker.failure_reason}"
        return None

    # ------------------------------------------------------------------------- tool execution
    def _run_tool(self, step: AgentStep) -> str:
        if step.action is None:                      # callers only pass tool steps; refuse rather than assert
            return "ERROR: no tool requested"
        fn = self.tools.get(step.action)
        if fn is None:
            return f"ERROR: unknown tool '{step.action}'. Available: {', '.join(sorted(self.tools)) or '(none)'}"
        try:
            out = fn(**step.action_input)
        except TypeError as exc:
            return f"ERROR: bad arguments for '{step.action}': {exc}"
        except Exception as exc:  # noqa: BLE001 - a failing tool is information for the agent, not a crash
            return f"ERROR: {type(exc).__name__}: {exc}"
        text = out if isinstance(out, str) else json.dumps(out, default=str)
        return text[: self.observation_limit]

    def run(self, query: str) -> ReActResult:
        tools_doc = "\n".join(f"- {n}: {(f.__doc__ or 'no description').strip().splitlines()[0]}" for n, f in sorted(self.tools.items()))
        system = self.SYSTEM.format(tools=tools_doc or "(none)")
        messages: List[Dict[str, str]] = [{"role": "user", "content": f"Task: {query}"}]
        result = ReActResult(status="max_cycles", answer=None)
        breaker = AgentCircuitBreaker()
        for _ in range(self.max_cycles):
            try:
                got = complete_structured(self.provider, LLMRequest(system, messages, self.model, 0.0, 1024), AgentStep,
                                          max_attempts=self.max_attempts)
            except StructuredOutputError as exc:
                result.status, result.stop_reason = "invalid_output", str(exc)
                return result
            result.structured_attempts += got.attempts
            for k, v in got.usage.items():
                result.usage[k] = result.usage.get(k, 0) + v
            step = got.value
            rec = ReActStepRecord(step.thought, step.action, step.action_input)
            result.steps.append(rec)
            if step.is_final:
                res = step.action_input.get("result")
                result.status, result.answer = "success", str(res) if res is not None else step.thought
                return result
            rec.observation = self._run_tool(step)
            breaker.record_cycle(step.thought, [step.action or ""], rec.observation)
            if self.max_total_tokens is not None and sum(result.usage.values()) > self.max_total_tokens:
                result.status, result.stop_reason = "budget", f"token budget {self.max_total_tokens} exceeded"
                return result
            reason = self._should_stop(result.steps, breaker)
            if reason:
                result.status, result.stop_reason = "stopped", reason
                return result
            messages += [{"role": "assistant", "content": step.model_dump_json()},
                         {"role": "user", "content": f"Observation: {rec.observation}"}]
        result.stop_reason = f"no final answer after {self.max_cycles} cycles"
        return result


# ======================================================================================== Tree of Thoughts
@dataclass
class ToTResult:
    path: List[str]
    best_score: float
    nodes_expanded: int
    failures: List[str] = field(default_factory=list)   # nodes whose hypotheses could not be generated/validated
    model_calls: int = 0

    def as_text(self) -> str:
        return " -> ".join(self.path)


class TreeOfThoughtsResilient(TreeOfThoughts):
    """Beam search over *validated* hypotheses. ``solve`` keeps the base signature (returns the path as text);
    use ``solve_detailed`` for scores, failures and call counts.

    ``evaluator`` (optional) re-scores each hypothesis with an independent judge; the model's own ``score`` is used
    otherwise. Generation failures never abort the search: that branch is skipped and recorded in ``failures``."""

    def __init__(self, provider: LLMProvider, *, branching: int = 3, model: Optional[str] = None, max_attempts: int = 3,
                 evaluator: Optional[Callable[[str, str], float]] = None) -> None:
        super().__init__(llm_call=lambda prompt: "", evaluator=evaluator)
        self.provider, self.branching, self.model, self.max_attempts = provider, max(1, branching), model, max_attempts
        self._external_evaluator = evaluator
        self.last_result: Optional[ToTResult] = None

    SYSTEM = ("You propose candidate next reasoning steps for a problem. Return exactly {n} DISTINCT hypotheses. For each, `score` is your "
              "honest estimate (a number between 0.0 and 1.0) of how promising that step is.")

    def _generate_hypotheses(self, node: ThoughtNode, problem: str) -> Tuple[List[Hypothesis], int]:
        path: List[str] = []
        cur: Optional[ThoughtNode] = node
        while cur is not None:
            path.append(cur.thought)
            cur = cur.parent
        user = f"Problem: {problem}\nReasoning so far: {' -> '.join(reversed(path))}\nPropose the next step."
        got = complete_structured(self.provider, LLMRequest(self.SYSTEM.format(n=self.branching), [{"role": "user", "content": user}],
                                                            self.model, 0.7, 1024), HypothesisBatch, max_attempts=self.max_attempts)
        seen, out = set(), []
        for h in got.value.hypotheses:
            key = h.thought.lower()
            if key not in seen:
                seen.add(key)
                out.append(h)
        return out[: self.branching], got.attempts

    def _generate_children(self, node: ThoughtNode, problem: str) -> List[str]:   # base-class compatible view
        return [h.thought for h in self._generate_hypotheses(node, problem)[0]]

    def solve_detailed(self, problem: str, depth: int = 3, beam_width: int = 2) -> ToTResult:
        root = ThoughtNode("Start", 1.0)
        active, expanded, calls, failures = [root], 0, 0, []
        for level in range(depth):
            candidates: List[ThoughtNode] = []
            for node in active:
                try:
                    hyps, attempts = self._generate_hypotheses(node, problem)
                except StructuredOutputError as exc:
                    failures.append(f"depth {level}, '{node.thought[:40]}': {exc.last_error}")
                    calls += exc.attempts
                    continue
                calls += attempts
                expanded += 1
                for h in hyps:
                    score = self._external_evaluator(h.thought, problem) if self._external_evaluator else h.score
                    child = ThoughtNode(h.thought, float(score), node)
                    node.children.append(child)
                    candidates.append(child)
            if not candidates:
                break
            candidates.sort(key=lambda c: c.score, reverse=True)
            active = candidates[:beam_width]
        best = max(active, key=lambda c: c.score, default=root)
        path = []
        cur: Optional[ThoughtNode] = best
        while cur is not None:
            path.append(cur.thought)
            cur = cur.parent
        self.last_result = ToTResult(list(reversed(path)), best.score, expanded, failures, calls)
        return self.last_result

    def solve(self, problem: str, depth: int = 3, beam_width: int = 2) -> str:
        return self.solve_detailed(problem, depth, beam_width).as_text()
