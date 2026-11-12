import json, stat, pytest
from pathlib import Path
from sintetico.devops.compliance import EvidenceStore, StageResult, build_evidence_package, verify_evidence_package
from sintetico.devops.residency import RegulatoryProfile, ResidencyPolicy, load_profiles

STAGES = [StageResult("secret-detection", "PASS", "gitleaks"), StageResult("sast-analysis", "PASS", "sonarqube"),
          StageResult("dependency-scan", "PASS", "trivy", 0)]
AGENT = {"prompt_sha256": "ab" * 32, "model_version": "m1", "eval_pass_rate": 0.85, "trust_score": 0.81,
         "human_approval": {"approver": "p@x.com", "approved_at": "2026-01-01T00:00:00Z"}}


def test_package_maps_controls_and_signs():
    pkg = build_evidence_package("abc123", STAGES, "key")
    assert pkg["overall_result"] == "PASS" and "PCI-DSS-6.3" in pkg["controls_satisfied"]
    assert verify_evidence_package(pkg, "key")
    assert not verify_evidence_package(pkg, "other-key")
    pkg["overall_result"] = "PASS "          # any edit invalidates the signature
    assert not verify_evidence_package(pkg, "key")


def test_failed_stage_fails_package_and_withholds_its_controls():
    pkg = build_evidence_package("c", STAGES + [StageResult("container-scan", "FAIL", "trivy", 3)], "k")
    assert pkg["overall_result"] == "FAIL" and pkg["stages"][-1]["findings"] == 3
    assert "PCI-DSS-6.3.3" in pkg["controls_satisfied"]       # still satisfied by dependency-scan (PASS)
    assert build_evidence_package("c2", [StageResult("iac-scan", "FAIL", "x")], "k")["controls_satisfied"] == []


def test_agent_evidence_must_be_complete():
    assert build_evidence_package("c", STAGES, "k", agent=AGENT)["agent"]["model_version"] == "m1"
    with pytest.raises(ValueError):
        build_evidence_package("c", STAGES, "k", agent={k: v for k, v in AGENT.items() if k != "trust_score"})
    with pytest.raises(ValueError):
        build_evidence_package("c", STAGES, "k", agent={**AGENT, "human_approval": {"approver": "x"}})


def test_input_validation():
    with pytest.raises(ValueError):
        build_evidence_package("c", [], "k")
    with pytest.raises(ValueError):
        build_evidence_package("c", STAGES, "")
    with pytest.raises(ValueError):
        StageResult("s", "MAYBE", "t")


def test_store_is_write_once_readonly_and_queryable(tmp_path):
    store = EvidenceStore(tmp_path / "ev")
    p = store.write(build_evidence_package("sha1", STAGES, "k"))
    assert stat.S_IMODE(p.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        store.write(build_evidence_package("sha1", STAGES, "k"))
    store.write(build_evidence_package("sha2", STAGES + [StageResult("iac-scan", "FAIL", "t")], "k"))
    assert len(store.query()) == 2 and len(store.query(result="FAIL")) == 1 and len(store.query(control="PCI-DSS-6.4")) == 2
    assert store.verify_all("k") == {"sha1": True, "sha2": True}
    assert store.verify_all("wrong") == {"sha1": False, "sha2": False}


@pytest.mark.parametrize("bad", ["", "../x", "a/b"])
def test_store_rejects_path_traversal(tmp_path, bad):
    with pytest.raises(ValueError):
        EvidenceStore(tmp_path).get(bad)


PROFILES = {"BR": RegulatoryProfile("BR", "LGPD", allowed_regions=["sa-east-1"]),
            "AR": RegulatoryProfile("AR", "PDPA", allowed_regions=["sa-east-1"], cross_border_requires_consent=False)}


def test_residency_allowed_region_consent_and_fail_closed():
    pol = ResidencyPolicy(PROFILES)
    assert pol.evaluate_write("BR", "sa-east-1").allowed
    assert not pol.evaluate_write("BR", "us-east-1").allowed
    assert pol.evaluate_write("BR", "us-east-1", {"explicit": True, "country": "BR"}).allowed
    assert not pol.evaluate_write("BR", "us-east-1", {"explicit": True, "country": "MX"}).allowed     # consent of another country
    assert not pol.evaluate_write("BR", "us-east-1", {"explicit": False, "country": "BR"}).allowed
    assert pol.evaluate_write("AR", "us-east-1").allowed                                              # profile says no consent needed
    assert not pol.evaluate_write("ZZ", "us-east-1").allowed                                           # unknown => deny
    assert pol.evaluate_write("br", "sa-east-1").allowed                                               # case-insensitive


def test_deployment_reports_every_violation():
    rep = ResidencyPolicy(PROFILES).evaluate_deployment({"db": ("BR", "us-east-1"), "s3": ("BR", "sa-east-1"), "q": ("XX", "eu-west-1")})
    assert not rep["compliant"] and {v["resource"] for v in rep["violations"]} == {"db", "q"}


def test_profiles_load_from_json_and_policy_is_data(tmp_path):
    (tmp_path / "BR.json").write_text(json.dumps({"country": "BR", "regulation": "LGPD", "data_residency": {"allowed_regions": ["sa-east-1"]},
                                                  "breach_notification": {"deadline_hours": 72}, "retention": {"audit_logs_years": 5}}))
    profiles = load_profiles(tmp_path)
    assert profiles["BR"].breach_notification_hours == 72 and profiles["BR"].retention_years["audit_logs_years"] == 5
    assert not ResidencyPolicy(profiles).evaluate_write("BR", "us-east-1").allowed
    # a regulatory change is a data edit, no code change:
    (tmp_path / "BR.json").write_text(json.dumps({"country": "BR", "regulation": "LGPD", "data_residency": {"allowed_regions": ["sa-east-1", "us-east-1"]}}))
    assert ResidencyPolicy(load_profiles(tmp_path)).evaluate_write("BR", "us-east-1").allowed


def test_example_profiles_shipped_with_the_project_load():
    profiles = load_profiles(Path(__file__).resolve().parent.parent / "examples" / "profiles")
    assert {"BR", "CO", "MX"} <= set(profiles)
