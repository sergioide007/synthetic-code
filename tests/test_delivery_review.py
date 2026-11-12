import pytest
from sintetico.adapters import OutputAdapter, OutputParseError, PromptAdapter, PROFILES
from sintetico.delivery import (Answer, AgentArchitectureReviewer, DeployGate, DeployMetrics, DeployPhase,
                                PipelineDecision, ProductionPipeline, ReviewStatus, SwarmOrchestrator, DEFAULT_QUESTIONS)
from sintetico.guards import DetectionGuard
from sintetico.providers import LLMProvider, LLMResponse
from sintetico.quality import ShadowAgent
from sintetico.review import CodeReviewAgent, ConfidenceEscalationReviewer, HumanInLoopReviewer, BugFinding


class Canned(LLMProvider):
    def __init__(self, text): self.text = text
    def complete(self, request): return LLMResponse(self.text, {"input_tokens": 1, "output_tokens": 1}, "m")
    @property
    def cost_per_million_tokens(self): return 0.0


def test_extract_json_from_fence_prose_and_reasoning_tags():
    assert OutputAdapter.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert OutputAdapter.extract_json('Sure! Here it is: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    assert OutputAdapter.extract_json('<thinking>{"x": 0}</thinking>{"y": 2}') == {"y": 2}
    with pytest.raises(OutputParseError):
        OutputAdapter.extract_json("no json here {oops")


def test_prompt_adapter_families():
    assert PromptAdapter.adapt("Do X.", "claude").startswith("<instructions>")
    assert "exactly this format" in PromptAdapter.adapt("Do X.", "gpt", '{"k": 1}')
    assert PromptAdapter.adapt("Be brief. Cite sources.", "gemini").splitlines()[1] == "1. Be brief"
    with pytest.raises(ValueError):
        PromptAdapter.adapt("x", "alien")
    assert "opus" in PROFILES["complex_reasoning"].recommended


def test_deploy_gate_advances_only_with_evidence():
    g = DeployGate()
    assert g.traffic_percent == 10
    assert g.advance(DeployMetrics(pass_rate=0.79, error_rate=0.0)) is DeployPhase.CANARY
    assert g.advance(DeployMetrics(pass_rate=0.80, error_rate=0.10)) is DeployPhase.SLOW_ROLLOUT
    assert g.advance(DeployMetrics(pass_rate=0.90, error_rate=0.0, trust_score=0.59)) is DeployPhase.SLOW_ROLLOUT
    assert g.advance(DeployMetrics(pass_rate=0.85, error_rate=0.0, trust_score=0.60)) is DeployPhase.FULL
    assert g.traffic_percent == 100


@pytest.mark.parametrize("m", [DeployMetrics(0.9, 0.21), DeployMetrics(0.69, 0.0), DeployMetrics(0.9, 0.0, 0.39), DeployMetrics(0.9, 0.0, 1.0, 5.1)])
def test_any_single_bad_signal_rolls_back(m):
    g = DeployGate()
    bad, reasons = g.should_rollback(m)
    assert bad and reasons
    assert g.rollback_if_needed(m) and g.traffic_percent == 0


def test_healthy_metrics_do_not_roll_back():
    assert DeployGate().should_rollback(DeployMetrics(0.9, 0.05, 0.8, 1.0)) == (False, [])


def test_architecture_review_status_ladder():
    r = AgentArchitectureReviewer()
    ids = [q.id for q in DEFAULT_QUESTIONS]
    all_ok = {i: Answer.COMPLIANT for i in ids}
    assert r.review(all_ok).status is ReviewStatus.APPROVED
    assert r.review({**all_ok, "srp_agents": Answer.NON_COMPLIANT}).status is ReviewStatus.APPROVED_WITH_CHANGES
    assert r.review({**all_ok, "pii_filter": Answer.NON_COMPLIANT}).status is ReviewStatus.REJECTED
    assert r.review({**all_ok, "eval_set": Answer.NON_COMPLIANT}).status is ReviewStatus.REJECTED
    partial = dict(all_ok); partial.pop("cost_tracking")
    assert r.review(partial).status is ReviewStatus.NEEDS_INFORMATION


def test_pipeline_decisions():
    ok = ProductionPipeline(lambda t: "good answer", ShadowAgent(strategy="constraints", constraints=["secret"]))
    assert ok.run("t").decision is PipelineDecision.AUTO_APPROVED
    bad = ProductionPipeline(lambda t: "leaks the secret", ShadowAgent(strategy="constraints", constraints=["secret"]))
    assert bad.run("t").decision is PipelineDecision.REJECTED
    low = ProductionPipeline(lambda t: "good", ShadowAgent(strategy="constraints"), confidence_threshold=0.95)
    assert low.run("t").decision is PipelineDecision.ESCALATED_TO_HUMAN


def test_pipeline_escalates_when_shadow_crashes():
    def boom(prompt): raise RuntimeError("llm down")
    p = ProductionPipeline(lambda t: "x", ShadowAgent(llm_call=boom, strategy="adversarial"))
    r = p.run("t")
    assert r.decision is PipelineDecision.ESCALATED_TO_HUMAN and "llm down" in r.note


def test_shadow_agent_understands_english_and_spanish():
    en = ShadowAgent(llm_call=lambda p: "I found no problems", strategy="adversarial").validate("t", "o")
    es = ShadowAgent(llm_call=lambda p: "no problemas", strategy="adversarial").validate("t", "o")
    assert en.verdict.value == es.verdict.value == "approve"
    assert ShadowAgent(strategy="consistency").validate("t", "There is a contradiction here").verdict.value == "reject"


def test_swarm_trust_and_approval_gates():
    calls = []
    agents = {r: (lambda s, r=r: calls.append(r) or f"{r} did {s}") for r in ("analyst", "writer", "deployer")}
    sw = SwarmOrchestrator(agents, {"analyst": 0.9, "writer": 0.5, "deployer": 0.9}, roles_requiring_approval={"deployer"})
    rep = sw.run("job", lambda t: [("analyst", "a"), ("writer", "b"), ("deployer", "c"), ("ghost", "d")])
    assert list(rep.results) == ["0:analyst"] and calls == ["analyst"]
    assert "trust gate" in rep.skipped["1:writer"] and "approval" in rep.skipped["2:deployer"] and "no such agent" in rep.skipped["3:ghost"]
    sw2 = SwarmOrchestrator(agents, {"deployer": 0.9}, {"deployer"}, approver=lambda role, task: True)
    assert "0:deployer" in sw2.run("j", lambda t: [("deployer", "x")]).results
    assert "[0:analyst] analyst did a" in rep.consolidated()


def test_code_review_parses_and_survives_garbage():
    good = '{"bugs": [{"line": 3, "type": "off-by-one", "severity": "CRITICAL", "description": "d", "test": "t", "confidence": 0.9}, {"nope": 1}]}'
    f = CodeReviewAgent(Canned(good)).review("diff")
    assert len(f) == 1 and f[0].severity == "critical" and f[0].line == 3
    assert CodeReviewAgent(Canned("totally not json")).review("diff") == []
    assert CodeReviewAgent(Canned('{"bugs": [{"line": 1, "severity": "weird"}]}')).review("d")[0].severity == "medium"


def test_human_in_loop_blocks_on_critical():
    notified = []
    rev = HumanInLoopReviewer(CodeReviewAgent(Canned('{"bugs": [{"line": 1, "type": "x", "severity": "critical", "description": "d"}]}')), notified.append)
    out = rev.review("diff")
    assert out.blocking and len(notified) == 1
    quiet = HumanInLoopReviewer(CodeReviewAgent(Canned('{"bugs": []}')), notified.append).review("diff")
    assert not quiet.blocking


def test_confidence_escalation():
    fs = [BugFinding(1, "a", "low", "", confidence=c) for c in (0.9, 0.7, 0.69, 0.1)]
    r = ConfidenceEscalationReviewer(0.7)
    auto, esc = r.split(fs)
    assert (len(auto), len(esc)) == (2, 2) and r.escalation_rate(fs) == 0.5 and r.escalation_rate([]) == 0.0


def test_detection_guard_blocks_irreversible_regardless_of_confidence():
    g = DetectionGuard()
    assert g.check(patch_text="ALTER TABLE users DROP COLUMN email").blocked_irreversible
    assert g.check(patch_text="issue refund for order").blocked_irreversible
    assert not g.check(patch_text="raise pool size from 20 to 50").pause_for_human


def test_detection_guard_time_and_scope_limits():
    g = DetectionGuard()
    assert g.check(users_affected_pct=51).pause_for_human
    assert not g.check(users_affected_pct=50).pause_for_human
    assert g.check(minutes_unresolved=16).pause_for_human
    assert g.check(hypothesis_validated=False, minutes_since_hypothesis=6).pause_for_human
    assert not g.check(hypothesis_validated=True, minutes_since_hypothesis=60).pause_for_human


def test_detection_guard_exact_boundaries_and_defaults():
    g = DetectionGuard()
    # thresholds are strict '>' : exactly at the limit does NOT pause, a hair above does
    assert not g.check(users_affected_pct=50.0).pause_for_human and g.check(users_affected_pct=50.1).pause_for_human
    assert not g.check(minutes_unresolved=15.0).pause_for_human and g.check(minutes_unresolved=15.1).pause_for_human
    assert not g.check(hypothesis_validated=False, minutes_since_hypothesis=5.0).pause_for_human
    assert g.check(hypothesis_validated=False, minutes_since_hypothesis=5.1).pause_for_human
    # defaults: calling with nothing is calm; an hypothesis is "validated" unless stated otherwise
    quiet = g.check()
    assert (quiet.pause_for_human, quiet.blocked_irreversible, quiet.reasons) == (False, False, [])
    assert not g.check(minutes_since_hypothesis=999).pause_for_human
    # custom limits are honoured
    strict = DetectionGuard(max_users_affected_pct=10, max_minutes_unresolved=1, max_minutes_unvalidated=0.5)
    assert strict.check(users_affected_pct=11).pause_for_human and strict.check(minutes_unresolved=2).pause_for_human
    assert strict.check(hypothesis_validated=False, minutes_since_hypothesis=0.6).pause_for_human
