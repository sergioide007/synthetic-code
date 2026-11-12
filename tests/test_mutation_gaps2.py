"""Second batch of behaviours that mutation testing showed were unpinned (boundaries, per-item coverage, exact codes)."""
import http.client, json, threading
import pytest
from sintetico.devops.compliance import EvidenceStore, StageResult, build_evidence_package, verify_evidence_package
from sintetico.devops.gitops import MAX_BODY_BYTES, TrustAdmission, serve
from sintetico.devops.iac_review import DB_PORTS, Decision, IaCReviewer, InstanceMetrics
from sintetico.devops.migration import BatchReport, row_hash
from sintetico.devops.residency import RegulatoryProfile

TAGS = {"env": "p", "owner": "o", "compliance": "c"}


def rds(before="db.r6g.2xlarge", after="db.r6g.large", rtype="aws_db_instance", actions=("update",), **kw):
    b = {"identifier": "d", "instance_class": before, "multi_az": kw.get("b_az", True), "tags": TAGS}
    a = {"identifier": "d", "instance_class": after, "multi_az": kw.get("a_az", True), "tags": TAGS}
    return {"resource_changes": [{"address": f"{rtype}.d", "type": rtype, "change": {"actions": list(actions), "before": b, "after": a}}]}


# ---- IaC: every database port, exact boundaries, conditions that must all hold
@pytest.mark.parametrize("port", sorted(DB_PORTS))
def test_each_database_port_is_blocked_when_open_to_the_world(port):
    plan = {"resource_changes": [{"address": "sg", "type": "aws_security_group", "change": {"actions": ["create"], "before": None,
            "after": {"ingress": [{"from_port": port, "to_port": port, "protocol": "tcp", "cidr_blocks": ["0.0.0.0/0"]}]}}}]}
    assert IaCReviewer().review(plan).decision is Decision.BLOCK
    plan["resource_changes"][0]["change"]["after"]["ingress"][0].update(from_port=8080, to_port=8080)
    assert IaCReviewer().review(plan).decision is Decision.APPROVE            # a non-database port stays allowed


def test_memory_headroom_boundary_is_exactly_eighty_percent():
    # before 64 GiB, after 16 GiB: working set = 64 - free. 80% of 16 = 12.8 GiB
    at_limit = IaCReviewer().review(rds(), metrics={"d": InstanceMetrics(10, 64 - 12.8)})
    over = IaCReviewer().review(rds(), metrics={"d": InstanceMetrics(10, 64 - 12.9)})
    assert at_limit.decision is Decision.APPROVE and over.decision is Decision.BLOCK


def test_cpu_thresholds_p99_exactly_40_and_80():
    ok = lambda cpu: IaCReviewer().review(rds(), metrics={"d": InstanceMetrics(cpu, 60)}).decision
    assert ok(40) is Decision.APPROVE and ok(40.1) is Decision.REVIEW and ok(79.9) is Decision.REVIEW and ok(80) is Decision.BLOCK


def test_breach_cost_and_monthly_cost_boundaries():
    m = {"d": InstanceMetrics(5, 60)}
    dep = lambda c: {"downstream_consumers": [{"service": "s", "breach_cost_usd": c}]}
    assert IaCReviewer().review(rds(), dep(5000), m).decision is Decision.APPROVE
    assert IaCReviewer().review(rds(), dep(5001), m).decision is Decision.REVIEW
    assert IaCReviewer().review(rds(), None, m, None, 500).decision is Decision.APPROVE
    assert IaCReviewer().review(rds(), None, m, None, 500.01).decision is Decision.REVIEW


def test_burst_window_only_matters_when_a_change_hour_is_known():
    dep = {"infrastructure": {"burst_window": "23:00-02:30"}}
    m = {"d": InstanceMetrics(5, 60)}
    assert IaCReviewer().review(rds(), dep, m, None).decision is Decision.APPROVE
    assert IaCReviewer().review(rds(), dep, m, 23).decision is Decision.REVIEW and IaCReviewer().review(rds(), dep, m, 1).decision is Decision.REVIEW
    assert IaCReviewer().review(rds(), dep, m, 2).decision is Decision.REVIEW           # window is [23:00, 02:00) in whole hours
    assert IaCReviewer().review(rds(), dep, m, 3).decision is Decision.APPROVE and IaCReviewer().review(rds(), dep, m, 22).decision is Decision.APPROVE
    day = {"infrastructure": {"burst_window": "09:00-17:00"}}
    assert IaCReviewer().review(rds(), day, m, 9).decision is Decision.REVIEW and IaCReviewer().review(rds(), day, m, 17).decision is Decision.APPROVE


def test_resize_rules_apply_only_to_rds_updates():
    m = {"d": InstanceMetrics(99, 1)}
    assert IaCReviewer().review(rds(rtype="aws_instance"), metrics=m).decision is Decision.APPROVE        # not an RDS instance
    assert IaCReviewer().review(rds(actions=("create",)), metrics=m).decision is Decision.APPROVE         # creation is not a resize
    assert IaCReviewer().review(rds("db.r6g.large", "db.r6g.2xlarge"), metrics=m).decision is Decision.APPROVE   # upsizing is fine


def test_multi_az_removal_is_blocked_only_for_pci_scoped_resources():
    pci = {"compliance_frameworks": ["PCI-DSS-4.0"]}
    same = ("db.r6g.large", "db.r6g.large")
    assert IaCReviewer().review(rds(*same, b_az=True, a_az=False), pci).decision is Decision.BLOCK
    assert IaCReviewer().review(rds(*same, b_az=True, a_az=False), {"compliance_frameworks": ["SOC2"]}).decision is Decision.APPROVE
    assert IaCReviewer().review(rds(*same, b_az=True, a_az=True), pci).decision is Decision.APPROVE
    assert IaCReviewer().review(rds(*same, b_az=False, a_az=False), pci).decision is Decision.APPROVE


# ---- compliance / residency
def test_unsigned_or_wrong_signature_never_verifies_and_query_filters_exactly():
    assert verify_evidence_package({}, "k") is False and verify_evidence_package({"hmac_sha256": ""}, "k") is False
    pkg = build_evidence_package("a", [StageResult("sast-analysis", "PASS", "t")], "k")
    pkg["hmac_sha256"] = "0" * 64
    assert verify_evidence_package(pkg, "k") is False


def test_query_by_result_returns_the_right_commits(tmp_path):
    s = EvidenceStore(tmp_path)
    s.write(build_evidence_package("good", [StageResult("sast-analysis", "PASS", "t")], "k"))
    s.write(build_evidence_package("bad", [StageResult("iac-scan", "FAIL", "t")], "k"))
    assert [r["commit_sha"] for r in s.query(result="PASS")] == ["good"] and [r["commit_sha"] for r in s.query(result="FAIL")] == ["bad"]
    assert [r["commit_sha"] for r in s.query(control="PCI-DSS-6.3")] == ["good"]


def test_profile_defaults_and_overrides_for_breach_notification(tmp_path):
    assert RegulatoryProfile.from_dict({"country": "XX"}).breach_notification_hours == 72
    assert RegulatoryProfile.from_dict({"country": "XX", "breach_notification": {"deadline_hours": 24}}).breach_notification_hours == 24
    assert RegulatoryProfile("XX", "r").breach_notification_hours == 72


# ---- migration helpers
def test_batch_report_clean_requires_both_lists_empty_and_hash_ignores_field_order():
    assert BatchReport(3).clean
    assert not BatchReport(3, mismatches=[{"id": "1"}]).clean and not BatchReport(3, missing_in_secondary=["1"]).clean
    assert BatchReport(0).mismatch_rate == 0.0 and BatchReport(4, mismatches=[{}], missing_in_secondary=["x"]).mismatch_rate == 0.5
    rec = {"a": 1, "b": 2}
    assert row_hash(rec, ["a", "b"]) == row_hash(rec, ["b", "a"])


# ---- GitOps gate: boundaries and exact HTTP codes
def _ann(score, sha="ab"):
    return {"synthetic-code/trust-score": score, "synthetic-code/prompt-sha256": sha}


def test_trust_score_range_and_threshold_boundaries():
    g = TrustAdmission(0.0)
    assert g.decide("Agent", _ann("0"))[0] and g.decide("Agent", _ann("1.0"))[0]
    assert not g.decide("Agent", _ann("1.0001"))[0] and not g.decide("Agent", _ann("-0.0001"))[0]
    d = TrustAdmission()                                                    # default minimum is 0.70
    assert d.decide("Agent", _ann("0.70"))[0] and not d.decide("Agent", _ann("0.6999"))[0]
    assert TrustAdmission(0.9).decide("ClaudeAgent", _ann("0.9"))[0]


def _post(srv, path, length, body=b""):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    try:
        c.putrequest("POST", path); c.putheader("Content-Length", str(length)); c.endheaders(body)
        return c.getresponse().status
    finally:
        c.close()


def test_webhook_status_codes_are_exact():
    srv = serve(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        good = json.dumps({"request": {"uid": "1", "kind": {"kind": "ConfigMap"}}}).encode()
        assert _post(srv, "/validate", len(good), good) == 200
        assert _post(srv, "/other", 0) == 404
        assert _post(srv, "/validate", 0) == 200                              # empty body is treated as {}
        assert _post(srv, "/validate", MAX_BODY_BYTES + 1) == 413 and _post(srv, "/validate", -1) == 413
        assert _post(srv, "/validate", 3, b"{{{") == 400
    finally:
        srv.shutdown(); srv.server_close()


# ---- values pinned to external reality (literals on purpose: a test that reads the constant it checks proves nothing)
def test_database_ports_and_instance_memory_match_the_real_world():
    from sintetico.devops.iac_review import RDS_MEMORY_GB
    assert DB_PORTS == {1433, 1521, 3306, 5432, 6379, 27017}            # SQL Server, Oracle, MySQL, PostgreSQL, Redis, MongoDB
    assert RDS_MEMORY_GB == {"db.r6g.large": 16, "db.r6g.xlarge": 32, "db.r6g.2xlarge": 64, "db.r6g.4xlarge": 128, "db.r6g.8xlarge": 256,
                             "db.r5.large": 16, "db.r5.xlarge": 32, "db.r5.2xlarge": 64, "db.r5.4xlarge": 128,
                             "db.t3.medium": 4, "db.t3.large": 8}


def test_webhook_body_limit_is_exactly_one_mebibyte():
    assert MAX_BODY_BYTES == 1_048_576
    srv = serve(port=0)                                                 # default minimum trust must be 0.70
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        body = b'{"request": {"uid": "1", "kind": {"kind": "ConfigMap"}}}'
        body = body + b" " * (1_048_576 - len(body))                    # exactly at the limit: accepted
        assert _post(srv, "/validate", 1_048_576, body) == 200
        assert _post(srv, "/validate", 1_048_577) == 413                # one byte over: refused before reading
        def decide(score):
            payload = json.dumps({"request": {"uid": "1", "kind": {"kind": "Agent"}, "object": {"metadata": {"annotations": _ann(score)}}}}).encode()
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
            try:
                c.request("POST", "/validate", payload, {"Content-Type": "application/json"})
                return json.loads(c.getresponse().read())["response"]["allowed"]
            finally:
                c.close()
        assert decide("0.70") is True and decide("0.69") is False
    finally:
        srv.shutdown(); srv.server_close()


@pytest.mark.parametrize("window,hour,inside", [
    ("23:30-02:00", 22, False), ("23:30-02:00", 23, True), ("23:30-02:00", 0, True), ("23:30-02:00", 1, True), ("23:30-02:00", 2, False),
    ("10:15-10:45", 9, False), ("10:15-10:45", 10, True), ("10:15-10:45", 11, False),
    ("09:00-17:00", 8, False), ("09:00-17:00", 9, True), ("09:00-17:00", 16, True), ("09:00-17:00", 17, False),
])
def test_burst_window_overlap_is_minute_accurate_and_conservative(window, hour, inside):
    from sintetico.devops.iac_review import _hour_overlaps_window, _parse_window
    assert _hour_overlaps_window(_parse_window(window), hour) is inside
    assert _parse_window("garbage") is None and _parse_window("25-x") is None
