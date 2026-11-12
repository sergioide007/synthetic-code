"""Contract tests against PRIMARY sources (real tool output / official type definitions), not against our own assumptions."""
import json
from pathlib import Path
import pytest
from sintetico import cli
from sintetico.devops.gitops import TrustAdmission
from sintetico.devops.iac_review import Decision, IaCReviewer

FIX = Path(__file__).parent / "fixtures"
PLAN = json.loads((FIX / "real_opentofu_plan.json").read_text())
K8S = json.loads((FIX / "k8s_admission_v1_fields.json").read_text())


def test_real_opentofu_plan_has_the_shape_the_reviewer_assumes():
    by_action = {tuple(rc["change"]["actions"]): rc for rc in PLAN["resource_changes"]}
    assert set(by_action) == {("create",), ("update",), ("delete",)}
    assert by_action[("create",)]["change"]["before"] is None            # create: before is null
    assert by_action[("delete",)]["change"]["after"] is None             # delete: after is null
    assert isinstance(by_action[("update",)]["change"]["before"], dict) and isinstance(by_action[("update",)]["change"]["after"], dict)
    assert all({"address", "type", "change"} <= set(rc) for rc in PLAN["resource_changes"])


def test_reviewer_handles_real_plan_including_null_before_and_after():
    r = IaCReviewer().review(PLAN)
    assert r.decision is Decision.APPROVE                                  # terraform_data is not a risk resource


def test_cli_accepts_the_real_plan_file(capsys):
    assert cli.main(["iac-review", "--plan", str(FIX / "real_opentofu_plan.json")]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "APPROVE"


def _review(operation, annotations, kind="ClaudeAgent"):
    obj = None if operation == "DELETE" else {"metadata": {"annotations": annotations}}
    return {"request": {"uid": "u-1", "kind": {"group": "ai.example.com", "version": "v1", "kind": kind}, "operation": operation,
                        "object": obj, "oldObject": {"metadata": {"annotations": {}}} if operation in ("DELETE", "UPDATE") else None}}


def test_response_matches_official_admission_v1_types():
    gate = TrustAdmission(0.7)
    for allowed_case in (_review("CREATE", {"synthetic-code/trust-score": "0.9", "synthetic-code/prompt-sha256": "ab"}), _review("CREATE", {})):
        out = gate.review(allowed_case)
        assert set(out) <= set(K8S["AdmissionReview"]["fields"]) | {"apiVersion", "kind"}      # TypeMeta is inlined
        resp = out["response"]
        assert set(resp) <= set(K8S["AdmissionResponse"]["fields"])
        assert set(K8S["AdmissionResponse"]["required"]) <= set(resp)                           # uid and allowed
        assert resp["uid"] == "u-1" and isinstance(resp["allowed"], bool)


def test_delete_is_never_blocked_even_for_untrusted_agents():
    """Found by reading the Go types: on DELETE `object` is null, so annotation checks would deny every removal."""
    gate = TrustAdmission(0.7)
    assert gate.review(_review("DELETE", {}))["response"]["allowed"] is True
    assert gate.review(_review("CONNECT", {}))["response"]["allowed"] is True
    assert gate.review(_review("CREATE", {}))["response"]["allowed"] is False
    assert gate.review(_review("UPDATE", {"synthetic-code/trust-score": "0.1"}))["response"]["allowed"] is False


def _sg(rtype, after):
    return {"resource_changes": [{"address": f"{rtype}.x", "type": rtype, "change": {"actions": ["create"], "before": None, "after": after}}]}


OPEN, DB = ["0.0.0.0/0"], 5432


@pytest.mark.parametrize("rtype,after", [
    ("aws_security_group", {"ingress": [{"from_port": DB, "to_port": DB, "protocol": "tcp", "cidr_blocks": OPEN}]}),
    ("aws_security_group", {"ingress": [{"from_port": 0, "to_port": 0, "protocol": "-1", "cidr_blocks": OPEN}]}),            # all protocols
    ("aws_security_group", {"ingress": [{"from_port": 3306, "to_port": 3306, "protocol": "tcp", "ipv6_cidr_blocks": ["::/0"]}]}),
    ("aws_security_group", {"ingress": [{"from_port": 0, "to_port": 65535, "protocol": "tcp", "cidr_blocks": OPEN}]}),
    ("aws_security_group_rule", {"type": "ingress", "from_port": 1433, "to_port": 1433, "protocol": "tcp", "cidr_blocks": OPEN}),
    ("aws_vpc_security_group_ingress_rule", {"cidr_ipv4": "0.0.0.0/0", "ip_protocol": "tcp", "from_port": DB, "to_port": DB}),
    ("aws_vpc_security_group_ingress_rule", {"cidr_ipv4": "0.0.0.0/0", "ip_protocol": "-1", "from_port": None, "to_port": None}),
    ("aws_vpc_security_group_ingress_rule", {"cidr_ipv6": "::/0", "ip_protocol": "tcp", "from_port": None, "to_port": None}),
])
def test_every_modern_way_of_exposing_a_database_port_is_blocked(rtype, after):
    after = {**after, "tags": {"env": "p", "owner": "o", "compliance": "c"}}
    assert IaCReviewer().review(_sg(rtype, after)).decision is Decision.BLOCK


@pytest.mark.parametrize("rtype,after", [
    ("aws_security_group", {"ingress": [{"from_port": 443, "to_port": 443, "protocol": "tcp", "cidr_blocks": OPEN}]}),
    ("aws_security_group", {"ingress": [{"from_port": DB, "to_port": DB, "protocol": "tcp", "cidr_blocks": ["10.0.0.0/8"]}]}),
    ("aws_security_group", {"ingress": [{"from_port": DB, "to_port": DB, "protocol": "udp", "cidr_blocks": OPEN}]}),
    ("aws_security_group_rule", {"type": "egress", "from_port": DB, "to_port": DB, "protocol": "tcp", "cidr_blocks": OPEN}),
    ("aws_vpc_security_group_ingress_rule", {"cidr_ipv4": "10.1.0.0/16", "ip_protocol": "-1"}),
])
def test_safe_rules_are_not_blocked(rtype, after):
    assert IaCReviewer().review(_sg(rtype, after)).decision is Decision.APPROVE
