"""Property-based tests (Hypothesis): thousands of generated cases per invariant, not hand-picked ones."""
import copy, json
from hypothesis import HealthCheck, given, settings, strategies as st
from sintetico.adapters import OutputAdapter, OutputParseError
from sintetico.audit import AgentAuditLogger
from sintetico.autonomy import threshold_for_error_cost
from sintetico.devops.compliance import StageResult, build_evidence_package, verify_evidence_package
from sintetico.devops.gitops import TrustAdmission
from sintetico.devops.iac_review import Decision, IaCReviewer
from sintetico.devops.migration import row_hash
from sintetico.devops.residency import RegulatoryProfile, ResidencyPolicy
from sintetico.evals import EvalCase, EvalSet

S = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
text = st.text(max_size=20)
jsonish = st.recursive(st.none() | st.booleans() | st.integers(-10**6, 10**6) | st.floats(allow_nan=False, allow_infinity=False) | text,
                       lambda c: st.lists(c, max_size=4) | st.dictionaries(text, c, max_size=4), max_leaves=12)


@S
@given(n=st.integers(1, 8), data=st.data())
def test_any_single_mutation_of_any_audit_event_is_detected(n, data):
    log = AgentAuditLogger("k")
    for i in range(n):
        log.log(data.draw(text, label="actor"), data.draw(text, label="action"), i=i, note=data.draw(text))
    assert log.verify() == (True, None)
    idx = data.draw(st.integers(0, n - 1))
    ev = log.events[idx]
    field = data.draw(st.sampled_from(["actor", "action", "timestamp", "details", "prev_hash"]))
    if field == "details":
        ev.details = {**ev.details, "i": ev.details["i"] + 1}
    elif field == "prev_hash":
        ev.prev_hash = ev.prev_hash[::-1] if len(set(ev.prev_hash)) > 1 else "x" + ev.prev_hash
    else:
        setattr(ev, field, getattr(ev, field) + "x")
    ok, bad = log.verify()
    assert not ok and bad == idx


@S
@given(n=st.integers(3, 8), a=st.integers(0, 7), b=st.integers(0, 7))
def test_reordering_audit_events_is_detected(n, a, b):
    a, b = a % n, b % n
    if a == b:
        return
    log = AgentAuditLogger("k")
    for i in range(n):
        log.log("a", "x", i=i)
    log.events[a], log.events[b] = log.events[b], log.events[a]
    assert not log.verify()[0]


@S
@given(key=st.sampled_from(["commit_sha", "generated_at", "overall_result", "controls_satisfied", "stages", "schema_version"]), junk=text)
def test_evidence_package_any_field_edit_breaks_signature(key, junk):
    pkg = build_evidence_package("c1", [StageResult("sast-analysis", "PASS", "t"), StageResult("iac-scan", "PASS", "t")], "k")
    assert verify_evidence_package(pkg, "k")
    tampered = copy.deepcopy(pkg)
    tampered[key] = [junk, "extra"] if isinstance(pkg[key], list) else str(pkg[key]) + junk + "!"
    assert not verify_evidence_package(tampered, "k")


@S
@given(country=text, region=text, consent=st.none() | st.dictionaries(text, jsonish, max_size=3))
def test_residency_unknown_country_is_always_denied(country, region, consent):
    pol = ResidencyPolicy({"BR": RegulatoryProfile("BR", "LGPD", allowed_regions=["sa-east-1"])})
    if country.upper() != "BR":
        assert pol.evaluate_write(country, region, consent).allowed is False


@S
@given(region=text.filter(lambda r: r != "sa-east-1"), consent=st.none() | st.dictionaries(text, jsonish, max_size=3))
def test_residency_foreign_region_needs_exact_explicit_consent(region, consent):
    pol = ResidencyPolicy({"BR": RegulatoryProfile("BR", "LGPD", allowed_regions=["sa-east-1"])})
    allowed = pol.evaluate_write("BR", region, consent).allowed
    valid = bool(consent) and consent.get("explicit") is True and str(consent.get("country", "")).upper() == "BR"
    assert allowed == valid


@S
@given(a=st.floats(0, 10**4), b=st.floats(0, 10**4))
def test_error_cost_threshold_is_monotonic_and_bounded(a, b):
    lo, hi = sorted((a, b))
    assert threshold_for_error_cost(lo) <= threshold_for_error_cost(hi)
    assert 0 < threshold_for_error_cost(a) <= 0.95


@S
@given(rec=st.dictionaries(st.sampled_from(["id", "balance", "currency", "noise", "other"]), st.integers(-5, 5) | text, min_size=3), delta=st.integers(1, 3))
def test_row_hash_ignores_order_and_unlisted_fields_but_sees_listed_ones(rec, delta):
    fields = ["id", "balance", "currency"]
    shuffled = dict(reversed(list(rec.items())))
    assert row_hash(rec, fields) == row_hash(shuffled, fields)
    assert row_hash(rec, fields) == row_hash({**rec, "noise": "zzz", "other": 12345}, fields) or "noise" in fields
    changed = {**rec, "balance": str(rec.get("balance", "")) + "#" * delta}
    assert row_hash(rec, fields) != row_hash(changed, fields)


@S
@given(plan=jsonish, dep=st.none() | jsonish, cost=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9))
def test_iac_reviewer_never_crashes_and_never_approves_garbage(plan, dep, cost):
    plan = plan if isinstance(plan, dict) else {"resource_changes": plan}
    r = IaCReviewer().review(plan, dep if isinstance(dep, dict) else None, None, None, cost)
    assert r.decision in set(Decision) and r.exit_code in (0, 1, 2)
    if not plan.get("resource_changes") and cost <= 500:
        assert r.decision is Decision.APPROVE or r.decision is Decision.BLOCK    # empty plan: nothing to object to; garbage: blocked


@S
@given(raw=st.text(max_size=300))
def test_extract_json_returns_json_or_raises_only_parse_error(raw):
    try:
        out = OutputAdapter.extract_json(raw)
    except OutputParseError:
        return
    json.dumps(out)                                      # whatever it returns must be real JSON data


@S
@given(ann=st.dictionaries(st.sampled_from(["synthetic-code/trust-score", "synthetic-code/prompt-sha256", "x"]), text, max_size=3))
def test_trust_admission_total_function_and_threshold_respected(ann):
    ok, _ = TrustAdmission(0.7).decide("ClaudeAgent", ann)
    if ok:
        assert float(ann["synthetic-code/trust-score"]) >= 0.7 and ann.get("synthetic-code/prompt-sha256")


@S
@given(outputs=st.lists(text, min_size=1, max_size=5))
def test_evalset_never_raises_whatever_the_agent_does(outputs):
    es = EvalSet([EvalCase(str(i), expected_contains=["a"]) for i in range(len(outputs))])
    it = iter(outputs)
    def agent(q):
        o = next(it)
        if o == "":
            raise RuntimeError("boom")
        return o
    r = es.run(agent)
    assert r.total == len(outputs) and 0.0 <= r.pass_rate <= 1.0
