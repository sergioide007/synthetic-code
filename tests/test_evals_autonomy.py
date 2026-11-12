import json
import pytest
from sintetico.autonomy import (ActivitySpec, DomainRiskProfile, DynamicThresholdRACIGate, Verifiability,
                                compute_max_allowed_autonomy, threshold_for_error_cost)
from sintetico.evals import CostOptimizerWithEvals, EvalCase, EvalSet
from sintetico.governance import RACILevel


def test_evalset_pass_rate_and_exception_safety():
    es = EvalSet([EvalCase("a", expected_contains=["A"]), EvalCase("b", expected_contains=["X"]), EvalCase("boom")])
    def agent(s):
        if s == "boom": raise RuntimeError("x")
        return s.upper()
    r = es.run(agent)
    assert (r.total, r.passed) == (3, 1) and r.pass_rate == pytest.approx(1 / 3)


def test_forbidden_regex_and_critical():
    es = EvalSet([EvalCase("q", forbidden_contains=["secret"], expected_regex=r"\d{3}", critical=True)])
    assert es.run(lambda s: "code 123").critical_failures == 0
    assert es.run(lambda s: "secret 123").critical_failures == 1
    assert es.run(lambda s: "no digits").critical_failures == 1


def test_evalset_from_json(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"name": "n", "version": 3, "cases": [{"input": "hi", "expected_contains": ["hi"]}]}))
    es = EvalSet.from_json(p)
    assert es.name == "n" and es.version == "3" and es.run(lambda s: s).pass_rate == 1.0


def test_empty_evalset_rejected():
    with pytest.raises(ValueError):
        EvalSet([])


def test_cost_optimizer_blocks_regressions_even_if_cheaper():
    es = EvalSet([EvalCase(str(i), expected_contains=[str(i)]) for i in range(10)])
    opt = CostOptimizerWithEvals(es, max_accuracy_drop=0.02)
    opt.measure_baseline(lambda s: s, cost_per_call=1.0)
    bad = opt.validate_candidate(lambda s: "" if s == "3" else s, cost_per_call=0.1)
    ok = opt.validate_candidate(lambda s: s, cost_per_call=0.5)
    assert not bad.approved and bad.accuracy_drop == pytest.approx(0.1)
    assert ok.approved and ok.cost_saving_pct == pytest.approx(50.0)
    with pytest.raises(RuntimeError):
        CostOptimizerWithEvals(es).validate_candidate(lambda s: s, 1.0)


def test_max_autonomy_follows_verifiability():
    P = lambda v: DomainRiskProfile("d", v, 0.6)
    assert compute_max_allowed_autonomy(P(Verifiability.NOT_VERIFIABLE)) == 0.0
    assert compute_max_allowed_autonomy(P(Verifiability.SLOW_MANUAL)) == 0.1
    assert compute_max_allowed_autonomy(P(Verifiability.QUICK_MANUAL)) == 0.6
    assert compute_max_allowed_autonomy(P(Verifiability.AUTOMATED)) == pytest.approx(0.72)   # book example


@pytest.mark.parametrize("cost,thr", [(5, .2), (10, .2), (11, .4), (30, .4), (31, .6), (60, .6), (61, .8), (100, .8), (101, .95)])
def test_threshold_tiers(cost, thr):
    assert threshold_for_error_cost(cost) == thr


def test_dynamic_gate_same_agent_different_outcomes():
    gate = DynamicThresholdRACIGate({
        "scan_logs": ActivitySpec(RACILevel.RESPONSIBLE, "sre-lead", 5),
        "notify_breach": ActivitySpec(RACILevel.RESPONSIBLE, "dpo", 95),
        "approve_merge": ActivitySpec(RACILevel.INFORMED, "tech-lead", 20),
        "bad_design": ActivitySpec(RACILevel.ACCOUNTABLE, "n/a", 1),
    })
    assert gate.authorize("scan_logs", 0.25).allowed
    d = gate.authorize("notify_breach", 0.25)
    assert not d.allowed and d.requires_human_approval and d.owner == "dpo"
    assert gate.authorize("notify_breach", 0.9).allowed
    assert not gate.authorize("approve_merge", 1.0).allowed
    assert not gate.authorize("bad_design", 1.0).allowed
    with pytest.raises(KeyError):
        gate.authorize("nope", 1.0)
