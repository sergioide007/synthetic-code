import json, threading, urllib.error, urllib.request
import pytest
from sintetico.audit import AgentAuditLogger
from sintetico.devops.gitops import TrustAdmission, serve
from sintetico.devops.iac_review import Decision, IaCReviewer, InstanceMetrics
from sintetico.devops.incident import (ApprovalRequired, GuardedToolbox, InMemoryQueue, IncidentSession, ToolPolicy, inspect_dlq, replay_dlq)
from sintetico.devops.migration import DualWriteRepository, MigrationAdvisor, Phase, row_hash, verify_batch

TAGS = {"env": "prod", "owner": "pay", "compliance": "pci"}


def rds_plan(before="db.r6g.2xlarge", after="db.r6g.large", **extra):
    return {"resource_changes": [{"address": "aws_db_instance.ledger", "type": "aws_db_instance", "change": {
        "actions": ["update"], "before": {"identifier": "ledger", "instance_class": before, "multi_az": True, "tags": TAGS, **extra.get("b", {})},
        "after": {"identifier": "ledger", "instance_class": after, "multi_az": True, "tags": TAGS, **extra.get("a", {})}}}]}


def test_etl_working_set_blocks_the_downsize_that_cost_a_fine():
    m = {"ledger": InstanceMetrics(p99_cpu_pct=94, min_freeable_memory_gb=36)}      # working set 64-36 = 28 GiB
    r = IaCReviewer().review(rds_plan(), metrics=m)
    assert r.decision is Decision.BLOCK and r.exit_code == 1 and any("working set" in x for x in r.reasons)


def test_safe_downsize_with_good_headroom_is_approved():
    r = IaCReviewer().review(rds_plan(), metrics={"ledger": InstanceMetrics(12, 56)})      # working set 8 GiB fits in 16
    assert r.decision is Decision.APPROVE and r.exit_code == 0


def test_downsize_without_metrics_needs_a_human():
    r = IaCReviewer().review(rds_plan())
    assert r.decision is Decision.REVIEW and r.exit_code == 2 and r.required_approvers


def test_unknown_instance_class_is_not_guessed():
    assert IaCReviewer().review(rds_plan("db.weird.1", "db.weird.2")).decision is Decision.REVIEW


def test_dependency_map_drives_review_burst_and_strict():
    dep = {"regulatory_sensitivity": "STRICT", "downstream_consumers": [{"service": "ledger-batch", "breach_cost_usd": 12000}],
           "infrastructure": {"burst_window": "23:00-02:30"}}
    m = {"ledger": InstanceMetrics(10, 58)}
    r = IaCReviewer().review(rds_plan(), dep, m, change_hour_utc=23)
    text = " | ".join(r.reasons)
    assert r.decision is Decision.REVIEW and "STRICT" in text and "breach cost" in text and "burst window" in text
    assert set(r.required_approvers) >= {"principal-engineer", "security-lead"}
    assert "burst window" not in " | ".join(IaCReviewer().review(rds_plan(), dep, m, change_hour_utc=10).reasons)


def test_security_group_open_db_port_blocks_and_multi_az_removal_on_pci():
    sg = {"resource_changes": [{"address": "aws_security_group.db", "type": "aws_security_group", "change": {
        "actions": ["update"], "before": {}, "after": {"ingress": [{"from_port": 5432, "to_port": 5432, "cidr_blocks": ["0.0.0.0/0"]}]}}}]}
    assert IaCReviewer().review(sg).decision is Decision.BLOCK
    ok = {"resource_changes": [{"address": "aws_security_group.web", "type": "aws_security_group", "change": {
        "actions": ["update"], "before": {}, "after": {"ingress": [{"from_port": 443, "to_port": 443, "cidr_blocks": ["0.0.0.0/0"]}]}}}]}
    assert IaCReviewer().review(ok).decision is Decision.APPROVE
    pci = rds_plan("db.r6g.large", "db.r6g.large", b={"multi_az": True}, a={"multi_az": False})
    assert IaCReviewer().review(pci, {"compliance_frameworks": ["PCI-DSS-4.0"]}).decision is Decision.BLOCK


def test_missing_tags_cost_delta_and_noop_plans():
    plan = {"resource_changes": [{"address": "aws_s3_bucket.x", "type": "aws_s3_bucket", "change": {"actions": ["create"], "before": None, "after": {"tags": {"env": "p"}}}}]}
    r = IaCReviewer().review(plan)
    assert r.decision is Decision.REVIEW and "missing required tags" in r.reasons[0]
    assert IaCReviewer().review({"resource_changes": []}, monthly_cost_delta_usd=501).decision is Decision.REVIEW
    assert IaCReviewer().review({"resource_changes": [{"type": "aws_db_instance", "change": {"actions": ["no-op"]}}]}).decision is Decision.APPROVE


def make_toolbox(approval=None, max_calls=60, clock=None):
    audit = AgentAuditLogger("k")
    kw = {"clock": clock} if clock else {}
    session = IncidentSession("inc-1", "payments", audit, max_tool_calls=max_calls, **kw)
    ex = {"logs": lambda q: f"logs for {q}", "replay": lambda n: f"replayed {n}", "boom": lambda: 1 / 0}
    pol = [ToolPolicy("logs"), ToolPolicy("replay", "WRITE", True, True, ("finance-lead", "eng-lead")), ToolPolicy("boom")]
    return GuardedToolbox(session, pol, ex, approval), audit


def test_read_tool_runs_and_is_audited():
    tb, audit = make_toolbox()
    out = tb.invoke("logs", q="5xx")
    assert out.ok and out.result == "logs for 5xx" and audit.events[-1].action == "tool_call" and audit.verify()[0]


def test_financial_tool_denied_without_approval_and_allowed_with():
    tb, _ = make_toolbox()
    d = tb.invoke("replay", n=5)
    assert not d.ok and d.escalate and "approval required" in d.error
    tb2, audit2 = make_toolbox(approval=lambda policy, params: policy.name == "replay")
    assert tb2.invoke("replay", n=5).ok
    assert [e.action for e in audit2.events][:2] == ["human_approval", "tool_call"]


def test_loop_detection_and_unknown_tool_and_tool_errors():
    tb, _ = make_toolbox()
    results = [tb.invoke("logs", q="same") for _ in range(4)]
    assert results[0].ok and results[1].ok and not results[2].ok and "loop" in results[2].error
    assert not tb.invoke("nope").ok
    tb2, audit = make_toolbox()
    out = tb2.invoke("boom")
    assert not out.ok and "ZeroDivisionError" in out.error and audit.events[-1].action == "tool_error"


def test_tool_call_and_time_budgets():
    tb, _ = make_toolbox(max_calls=2)
    tb.invoke("logs", q="a"); tb.invoke("logs", q="b")
    assert "budget" in tb.invoke("logs", q="c").error
    t = [0.0]
    tb2, _ = make_toolbox(clock=lambda: t[0])
    t[0] = 901.0
    assert "time budget" in tb2.invoke("logs", q="x").error


def test_dlq_inspect_is_read_only_and_replay_needs_approver_and_notifies():
    dlq = InMemoryQueue([{"id": i} for i in range(5)])
    assert inspect_dlq(dlq, 2) == {"depth": 5, "sample": [{"id": 0}, {"id": 1}]} and len(dlq) == 5
    target, notified = InMemoryQueue(), []
    with pytest.raises(ApprovalRequired):
        replay_dlq(dlq, target, None, notified.append)
    assert len(dlq) == 5 and notified == []
    s = replay_dlq(dlq, target, "finance-lead", notified.append, max_messages=3)
    assert s == {"replayed": 3, "remaining": 2, "approver": "finance-lead"} and notified == [s]
    assert [x["id"] for x in target.peek(10)] == [0, 1, 2]                    # order preserved


FIELDS = ["id", "balance", "currency"]


def test_row_hash_detects_content_even_when_counts_match():
    a = {"id": "1", "balance": 100, "currency": "USD", "ignored": "x"}
    assert row_hash(a, FIELDS) == row_hash({**a, "ignored": "y"}, FIELDS)
    assert row_hash(a, FIELDS) != row_hash({**a, "balance": 101}, FIELDS)
    assert row_hash({"id": 1, "balance": 5, "currency": "U"}, FIELDS) == row_hash({"id": "1", "balance": "5", "currency": "U"}, FIELDS)


def test_verify_batch_reports_diff_fields_and_missing():
    p = {"1": {"id": "1", "balance": 10, "currency": "USD"}, "2": {"id": "2", "balance": 20, "currency": "USD"}, "3": {"id": "3", "balance": 5, "currency": "USD"}}
    s = {"1": dict(p["1"]), "2": {**p["2"], "balance": 21}}
    r = verify_batch(["1", "2", "3"], p, s, FIELDS)
    assert r.mismatches == [{"id": "2", "fields": ["balance"]}] and r.missing_in_secondary == ["3"]
    assert r.mismatch_rate == pytest.approx(2 / 3) and not r.clean
    assert verify_batch(["1"], p, s, FIELDS).clean


def test_dual_write_never_fails_the_primary_and_dlq_is_recoverable():
    primary, secondary, healthy = {}, {}, [False]
    def write2(i, rec):
        if not healthy[0]: raise ConnectionError("sql down")
        secondary[i] = rec
    repo = DualWriteRepository(primary, write2)
    repo.write("a", {"id": "a", "balance": 1}); repo.write("b", {"id": "b", "balance": 2})
    assert set(primary) == {"a", "b"} and secondary == {} and len(repo.dlq) == 2 and "ConnectionError" in repo.dlq[0]["error"]
    assert repo.reprocess_dlq() == {"recovered": 0, "still_failing": 2}
    healthy[0] = True
    assert repo.reprocess_dlq() == {"recovered": 2, "still_failing": 0} and set(secondary) == {"a", "b"}


def test_advisor_phases_and_human_cutover():
    adv = MigrationAdvisor()
    assert adv.evaluate(Phase.SHADOW, 3, 0).action == "HOLD"
    assert adv.evaluate(Phase.SHADOW, 7, 100).action == "ADVANCE"
    assert adv.evaluate(Phase.DUAL_WRITE, 14, 11).action == "HOLD"
    hold = adv.evaluate(Phase.VERIFY, 14, 0, consecutive_clean_batches=3, mismatch_rate=0.0001, rollback_tested=True)
    assert hold.action == "HOLD" and any("mismatch" in b for b in hold.blockers)
    assert "rollback" in " ".join(adv.evaluate(Phase.VERIFY, 14, 0, 3, 0.0, False).blockers)
    cut = adv.evaluate(Phase.VERIFY, 14, 0, 3, 0.0, True)
    assert cut.action == "PROPOSE_CUTOVER" and cut.requires_human and cut.approvers == ["principal-engineer", "cto"] and cut.evidence["rollback_tested"]
    assert adv.evaluate(Phase.CUTOVER, 1, 0).action == "COMPLETE"


def review(kind, annotations):
    return {"request": {"uid": "u1", "kind": {"kind": kind}, "object": {"metadata": {"annotations": annotations}}}}


def test_trust_admission_decisions():
    g = TrustAdmission(0.70)
    P = "synthetic-code/"
    ok = {P + "trust-score": "0.81", P + "prompt-sha256": "abcdef123456", P + "eval-set-version": "v2"}
    assert g.review(review("ClaudeAgent", ok))["response"]["allowed"]
    low = g.review(review("ClaudeAgent", {**ok, P + "trust-score": "0.69"}))["response"]
    assert not low["allowed"] and "0.69" in low["status"]["message"]
    assert not g.review(review("ClaudeAgent", {}))["response"]["allowed"]
    assert not g.review(review("ClaudeAgent", {**ok, P + "trust-score": "high"}))["response"]["allowed"]
    assert not g.review(review("ClaudeAgent", {**ok, P + "trust-score": "1.7"}))["response"]["allowed"]
    assert not g.review(review("ClaudeAgent", {P + "trust-score": "0.9"}))["response"]["allowed"]     # no prompt hash
    assert g.review(review("ConfigMap", {}))["response"]["allowed"]                                      # not gated
    assert g.review(review("ClaudeAgent", ok))["response"]["uid"] == "u1"


def test_admission_webhook_over_http():
    srv = serve(port=0, min_trust=0.7)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/validate"
        body = json.dumps(review("ClaudeAgent", {"synthetic-code/trust-score": "0.2"})).encode()
        with urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"})) as r:
            resp = json.load(r)
        assert resp["response"]["allowed"] is False
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(url, b"not json", {"Content-Type": "application/json"}))
        assert e.value.code == 400
        e.value.close()
    finally:
        srv.shutdown(); srv.server_close()
