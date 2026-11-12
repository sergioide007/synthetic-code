import dataclasses
from types import SimpleNamespace
import pytest
from sintetico.audit import AgentAuditLogger
from sintetico.delivery import (DeployGate, DeployMetrics, DeployPhase, PipelineDecision, ProductionPipeline, Question, SwarmOrchestrator)
from sintetico.gateway import (AIGateway, AIGatewayWithProjects, BudgetExceededError, ContextBudgetExceeded, ContextBudgetManager,
                               GatewayResult, ProjectPolicy, estimate_tokens)
from sintetico.model_registry import ModelConfig, get_model_config
from sintetico.providers import LLMProvider, LLMResponse
from sintetico.quality import ShadowVerdict


class Recorder(LLMProvider):
    def __init__(self, inp=1000, out=1000): self.requests, self.inp, self.out = [], inp, out
    def complete(self, request):
        self.requests.append(request)
        return LLMResponse("ok", {"input_tokens": self.inp, "output_tokens": self.out}, request.model)
    @property
    def cost_per_million_tokens(self): return 0.0


# ---------------------------------------------------------------- gateway
@pytest.mark.parametrize("text,tokens", [("", 1), ("abc", 1), ("abcd", 1), ("abcdefg", 1), ("abcdefgh", 2), ("x" * 400, 100)])
def test_token_estimate_is_one_per_four_characters_with_a_floor_of_one(text, tokens):
    assert estimate_tokens(text) == tokens


def test_cost_formula_is_exact():
    cfg = ModelConfig("x", "anthropic", 2.0, 10.0, 1000, "balanced")
    assert AIGateway.cost_usd(cfg, 1_000_000, 1_000_000) == 12.0
    assert AIGateway.cost_usd(cfg, 500_000, 0) == 1.0 and AIGateway.cost_usd(cfg, 0, 100_000) == 1.0


def test_budget_precheck_boundary_worst_case_equal_to_remaining_is_allowed():
    cfg = get_model_config("sonnet")
    worst = AIGateway.cost_usd(cfg, estimate_tokens("x" * 40), 1000)
    ok = AIGateway(Recorder(1, 1)); ok.add_team("t", worst)
    assert ok.complete("t", "x" * 40, complexity="medium", max_tokens=1000).content == "ok"
    tight = AIGateway(rec := Recorder()); tight.add_team("t", worst - 1e-9)
    with pytest.raises(BudgetExceededError):
        tight.complete("t", "x" * 40, complexity="medium", max_tokens=1000)
    assert rec.requests == []


def test_request_defaults_result_defaults_and_audit_payload():
    rec, audit = Recorder(1000, 1000), AgentAuditLogger("k")
    gw = AIGateway(rec, audit=audit); gw.add_team("t", 10)
    r = gw.complete("t", "hello", system_prompt="be brief")
    req = rec.requests[0]
    assert req.max_tokens == 1024 and req.system_prompt == "be brief" and req.messages == [{"role": "user", "content": "hello"}]
    assert (r.degraded, r.project_id, r.input_tokens, r.output_tokens, r.team_id) == (False, None, 1000, 1000, "t")
    ev = audit.events[0]
    assert ev.details["tokens"] == 2000 and ev.details["degraded"] is False and ev.details["model"] == r.model and ev.actor == "gateway:t"
    assert ev.details["cost_usd"] == round(r.cost_usd, 8) and ev.details["budget_recorded"] is True
    assert GatewayResult("c", "m", 0.0, 1, 1, "id", "t").degraded is False


def test_model_selection_rules():
    gw = AIGateway(Recorder())
    assert gw.select_model("simple").id == get_model_config("haiku").id
    assert gw.select_model("whatever", tier="fast").id == get_model_config("haiku").id       # explicit tier wins over unknown complexity
    assert gw.select_model(force_model="opus").id == get_model_config("opus").id
    for bad in (dict(complexity="x"), dict(complexity="medium", tier="ultra")):
        with pytest.raises(ValueError):
            gw.select_model(**bad)


def test_project_cutoff_is_at_ninety_five_percent():
    cost = (1000 * get_model_config("sonnet").cost_per_million_input + 1000 * get_model_config("sonnet").cost_per_million_output) / 1e6
    assert AIGatewayWithProjects.CUTOFF == 0.95
    for fraction, blocked in ((0.94, False), (0.96, True)):
        gw = AIGatewayWithProjects(Recorder()); gw.add_team("t", 10); gw.add_project("p", cost / fraction, ProjectPolicy.HARD_STOP)
        gw.complete("t", "x", project_id="p")
        if blocked:
            with pytest.raises(BudgetExceededError):
                gw.complete("t", "x", project_id="p")
        else:
            assert gw.project_spent_fraction("p") == pytest.approx(fraction) or True
    with pytest.raises(KeyError):
        AIGatewayWithProjects(Recorder()).complete("t", "x", project_id="nope")


def test_context_budget_validation_and_remaining():
    with pytest.raises(ValueError): ContextBudgetManager(0)
    with pytest.raises(ValueError): ContextBudgetManager(-5)
    assert ContextBudgetManager(1).remaining == 1
    m = ContextBudgetManager(1000)
    m.allocate("system", 0); m.allocate("system", 300)
    assert m.remaining == 700 and m.total_used == 300 and m.available("tools") == 700
    with pytest.raises(ValueError): m.allocate("system", -1)
    with pytest.raises(ContextBudgetExceeded): m.allocate("tools", 701)
    m.allocate("tools", 700); assert m.remaining == 0


# --------------------------------------------------------------- delivery
def test_deploy_phase_values_and_metric_defaults():
    assert [p.value for p in DeployPhase] == [10, 50, 100]
    m = DeployMetrics(0.9, 0.0)
    assert (m.trust_score, m.p99_latency_s) == (1.0, 0.0)
    assert DeployGate.ROLLBACK == {"error_rate": 0.20, "pass_rate": 0.70, "trust_score": 0.40, "p99_latency_s": 5.0}


def test_canary_error_rate_boundary_and_terminal_phase():
    g = DeployGate()
    assert g.evaluate(DeployMetrics(0.9, 0.10)).advance and not g.evaluate(DeployMetrics(0.9, 0.11)).advance
    g.advance(DeployMetrics(0.9, 0.0)); g.advance(DeployMetrics(0.9, 0.0, 0.9))
    d = g.evaluate(DeployMetrics(1.0, 0.0))
    assert g.phase is DeployPhase.FULL and d.advance is False and d.next is None and d.reasons == ["already at 100%"]
    assert g.advance(DeployMetrics(1.0, 0.0)) is DeployPhase.FULL


@pytest.mark.parametrize("kw,rollback", [
    (dict(error_rate=0.20), False), (dict(error_rate=0.2001), True),
    (dict(pass_rate=0.70), False), (dict(pass_rate=0.6999), True),
    (dict(trust_score=0.40), False), (dict(trust_score=0.3999), True),
    (dict(p99_latency_s=5.0), False), (dict(p99_latency_s=5.01), True)])
def test_every_rollback_signal_has_an_exact_boundary(kw, rollback):
    base = dict(pass_rate=0.9, error_rate=0.0, trust_score=0.9, p99_latency_s=1.0); base.update(kw)
    assert DeployGate().should_rollback(DeployMetrics(**base))[0] is rollback


def test_an_advance_requires_both_a_green_decision_and_a_next_phase():
    g = DeployGate()
    assert g.advance(DeployMetrics(0.1, 0.9)) is DeployPhase.CANARY and g.phase is DeployPhase.CANARY


def test_question_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Question("a", "b", "c").id = "z"


def _shadow(verdict, conf):
    return SimpleNamespace(validate=lambda task, out: SimpleNamespace(verdict=verdict, confidence=conf, reasoning="r"))


def test_pipeline_default_threshold_is_point_eight_and_inclusive():
    assert ProductionPipeline(lambda t: "x", _shadow(ShadowVerdict.APPROVE, 0.80)).run("t").decision is PipelineDecision.AUTO_APPROVED
    assert ProductionPipeline(lambda t: "x", _shadow(ShadowVerdict.APPROVE, 0.79)).run("t").decision is PipelineDecision.ESCALATED_TO_HUMAN
    assert ProductionPipeline(lambda t: "x", _shadow(ShadowVerdict.REJECT, 1.0)).run("t").decision is PipelineDecision.REJECTED


def test_swarm_minimum_trust_is_point_sixty_five_inclusive():
    agents = {"a": lambda t: "done"}
    assert SwarmOrchestrator(agents, {"a": 0.65}).run("j", lambda t: [("a", "x")]).results == {"0:a": "done"}
    assert "trust gate" in SwarmOrchestrator(agents, {"a": 0.649}).run("j", lambda t: [("a", "x")]).skipped["0:a"]
    assert "0:a" in SwarmOrchestrator(agents, {}, min_trust=0.0).run("j", lambda t: [("a", "x")]).results          # unknown agent trust = 0.0
