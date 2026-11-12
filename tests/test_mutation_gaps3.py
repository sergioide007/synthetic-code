import json
import pytest
from sintetico.evals import CostOptimizerWithEvals, EvalCase, EvalReport, EvalResult, EvalSet


def test_custom_check_callable_must_pass_and_combines_with_declarative_rules():
    assert EvalCase("q", check=lambda o: o.startswith("ok")).passes("ok!")
    assert not EvalCase("q", check=lambda o: o.startswith("ok")).passes("nope")
    assert not EvalCase("q", check=lambda o: True, expected_contains=["x"]).passes("y")       # all rules must hold
    assert EvalCase("q").passes("anything") and EvalCase("q").critical is False and EvalCase("q").tags == ()


def test_report_numbers_and_dict_shape():
    r = EvalReport([EvalResult("a", "", True), EvalResult("b", "", False, critical=True), EvalResult("c", "", False)])
    assert (r.total, r.passed, r.critical_failures) == (3, 1, 1)
    assert r.to_dict() == {"total": 3, "passed": 1, "pass_rate": 0.3333, "critical_failures": 1}
    assert EvalReport([]).pass_rate == 0.0 and EvalResult("a", "", True).critical is False


def test_from_json_keeps_critical_flag_and_tags(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"cases": [{"input": "a", "critical": True, "tags": ["pii"]}, {"input": "b"}]}))
    es = EvalSet.from_json(p)
    assert [c.critical for c in es.cases] == [True, False] and list(es.cases[0].tags) == ["pii"] and list(es.cases[1].tags) == []
    assert es.name == "e" and es.version == "1"


def _opt(drop_cases_failed, cost, max_drop=None):
    es = EvalSet([EvalCase(str(i), expected_contains=[str(i)]) for i in range(50)])      # each failed case = 2% accuracy
    opt = CostOptimizerWithEvals(es) if max_drop is None else CostOptimizerWithEvals(es, max_drop)
    opt.measure_baseline(lambda s: s, cost_per_call=1.0)
    broken = {str(i) for i in range(drop_cases_failed)}
    return opt.validate_candidate(lambda s: "" if s in broken else s, cost_per_call=cost)


def test_accuracy_drop_boundary_is_inclusive_at_two_percent_by_default():
    assert _opt(1, 0.5).approved is True                  # exactly 0.02 drop: allowed
    d = _opt(2, 0.5)
    assert d.approved is False and d.accuracy_drop == pytest.approx(0.04) and "dropped" in d.reason
    assert _opt(2, 0.5, max_drop=0.04).approved is True and _opt(3, 0.5, max_drop=0.04).approved is False


def test_a_candidate_that_saves_nothing_is_rejected_whatever_its_quality():
    same = _opt(0, 1.0)
    assert same.approved is False and same.cost_saving_pct == 0 and same.reason == "no cost saving"
    worse = _opt(0, 1.5)
    assert worse.approved is False and worse.cost_saving_pct == pytest.approx(-50.0)
    ok = _opt(0, 0.9)
    assert ok.approved is True and ok.cost_saving_pct == pytest.approx(10.0) and ok.baseline_accuracy == ok.candidate_accuracy == 1.0


def test_zero_cost_baseline_cannot_claim_savings():
    es = EvalSet([EvalCase("a")])
    opt = CostOptimizerWithEvals(es); opt.measure_baseline(lambda s: s, 0.0)
    assert opt.validate_candidate(lambda s: s, 0.0).approved is False
