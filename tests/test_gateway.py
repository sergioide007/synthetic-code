import pytest
from sintetico.audit import AgentAuditLogger
from sintetico.model_registry import get_model_config
from sintetico.gateway import (AIGateway, AIGatewayWithProjects, BudgetExceededError, ContextBudgetExceeded,
                               ContextBudgetManager, ProjectPolicy)
from sintetico.providers import LLMProvider, LLMResponse


class FixedUsage(LLMProvider):
    """Reports exactly 1,000 input + 1,000 output tokens per call."""
    def __init__(self): self.models = []
    def complete(self, request):
        self.models.append(request.model)
        return LLMResponse("ok", {"input_tokens": 1000, "output_tokens": 1000}, request.model)
    @property
    def cost_per_million_tokens(self): return 0.0


def _sonnet_call_cost():
    c = get_model_config("sonnet")
    return (1000 * c.cost_per_million_input + 1000 * c.cost_per_million_output) / 1e6


def make(budget=10.0, cls=AIGateway):
    p = FixedUsage()
    gw = cls(p, audit=AgentAuditLogger("k"))
    gw.add_team("t", budget)
    return gw, p


def test_routes_by_complexity_and_computes_cost():
    gw, p = make()
    r = gw.complete("t", "hi", complexity="simple")
    cfg = get_model_config("haiku")
    assert r.model == cfg.id and r.cost_usd == pytest.approx((1000 * cfg.cost_per_million_input + 1000 * cfg.cost_per_million_output) / 1e6)
    gw.complete("t", "hi", complexity="complex")
    assert p.models[-1] == get_model_config("opus").id


def test_unknown_complexity_and_team():
    gw, _ = make()
    with pytest.raises(ValueError):
        gw.complete("t", "x", complexity="galactic")
    with pytest.raises(KeyError):
        gw.complete("nobody", "x")


def test_budget_blocks_before_calling_provider():
    gw, p = make(budget=0.0001)
    with pytest.raises(BudgetExceededError):
        gw.complete("t", "x" * 100, complexity="complex", max_tokens=4000)
    assert p.models == []                      # provider never called


def test_budgets_are_isolated_per_team():
    gw, _ = make()
    gw.add_team("other", 10.0)
    gw.complete("t", "x")
    assert gw.budgets["t"].spent > 0 and gw.budgets["other"].spent == 0


def test_every_call_is_audited_and_chain_is_valid():
    gw, _ = make()
    for _ in range(3):
        gw.complete("t", "x")
    rep = gw.audit.export_report()
    assert rep["events"] == 3 and rep["chain_valid"] and rep["total_cost_usd"] > 0


def test_provider_mapping_by_vendor():
    anth, oai = FixedUsage(), FixedUsage()
    gw = AIGateway({"anthropic": anth, "openai": oai})
    gw.add_team("t", 10)
    gw.complete("t", "x", complexity="simple")
    gw.complete("t", "x", model="gpt-4o")
    assert len(anth.models) == 1 and len(oai.models) == 1
    gw2 = AIGateway({"anthropic": anth})
    gw2.add_team("t", 10)
    with pytest.raises(KeyError):
        gw2.complete("t", "x", model="gpt-4o")      # no provider configured for openai


def test_project_hard_stop():
    gw, _ = make(cls=AIGatewayWithProjects)
    gw.add_project("p", _sonnet_call_cost() / 0.97, ProjectPolicy.HARD_STOP)   # one call = 97% of the project budget
    gw.complete("t", "x", project_id="p")
    with pytest.raises(BudgetExceededError):
        gw.complete("t", "x", project_id="p")


def test_project_degrade_forces_cheapest_tier_and_flags_result():
    gw, _ = make(cls=AIGatewayWithProjects)
    gw.add_project("p", _sonnet_call_cost() / 0.97, ProjectPolicy.DEGRADE)
    first = gw.complete("t", "x", project_id="p")
    second = gw.complete("t", "x", project_id="p", complexity="complex")
    assert not first.degraded and second.degraded and second.model == get_model_config("haiku").id


def test_context_budget_caps_rag_at_half():
    m = ContextBudgetManager(1000, {"system": 200})
    assert m.available("rag") == 500
    m.allocate("rag", 500)
    with pytest.raises(ContextBudgetExceeded):
        m.allocate("rag", 1)
    with pytest.raises(ContextBudgetExceeded):
        m.allocate("system", 201)


def test_fit_chunks_keeps_leading_chunks_only():
    m = ContextBudgetManager(400, rag_max_fraction=0.5)      # rag cap = 200 tokens = ~800 chars
    chunks = ["a" * 400, "b" * 400, "c" * 400]
    assert m.fit_chunks(chunks) == chunks[:2]
