"""The signature schemes verified by a SEPARATE implementation (jq + openssl), not by this library."""
import json, os, shutil, subprocess
from pathlib import Path
import pytest
from sintetico.audit import AgentAuditLogger
from sintetico.devops.compliance import EvidenceStore, StageResult, build_evidence_package

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
pytestmark = pytest.mark.skipif(not (shutil.which("jq") and shutil.which("openssl")), reason="needs jq and openssl")


def sh(script, *args, key_var, key="s3cret"):
    return subprocess.run(["sh", str(SCRIPTS / script), *args], capture_output=True, text=True, env={**os.environ, key_var: key})


def make_log(path):
    log = AgentAuditLogger("s3cret", path=path)
    log.log("gateway:ñandú", "llm_call", cost_usd=0.5, tokens=120, note="acentos y emoji 🚀")      # non-ASCII on purpose
    log.log("human", "human_approval", approver="ana@x.com", ok=True, extra=None, nested={"b": [1, 2.5, "x"], "a": 0})
    log.log("gateway:ops", "llm_call", cost_usd=0.25, tokens=60)
    return log


def test_openssl_agrees_with_python_on_the_audit_chain(tmp_path):
    f = tmp_path / "a.jsonl"; log = make_log(f)
    r = sh("verify_audit_with_openssl.sh", str(f), key_var="SINTETICO_AUDIT_KEY")
    assert r.returncode == 0 and "OK: 3 events" in r.stdout and log.head in r.stdout, r.stdout + r.stderr


def test_openssl_detects_the_same_attacks_python_detects(tmp_path):
    f = tmp_path / "a.jsonl"; log = make_log(f); anchor = log.head
    lines = f.read_text().splitlines()
    # 1) edited content
    bad = json.loads(lines[0]); bad["details"]["cost_usd"] = 0.0
    f.write_text("\n".join([json.dumps(bad), *lines[1:]]) + "\n")
    r = sh("verify_audit_with_openssl.sh", str(f), key_var="SINTETICO_AUDIT_KEY")
    assert r.returncode == 1 and "BAD SIGNATURE at event 0" in r.stdout
    # 2) deleted middle event
    f.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert "BROKEN LINK at event 1" in sh("verify_audit_with_openssl.sh", str(f), key_var="SINTETICO_AUDIT_KEY").stdout
    # 3) wrong key
    f.write_text("\n".join(lines) + "\n")
    assert sh("verify_audit_with_openssl.sh", str(f), key_var="SINTETICO_AUDIT_KEY", key="nope").returncode == 1
    # 4) truncated tail: only the stored anchor reveals it
    f.write_text("\n".join(lines[:2]) + "\n")
    assert sh("verify_audit_with_openssl.sh", str(f), key_var="SINTETICO_AUDIT_KEY").returncode == 0
    t = subprocess.run(["sh", str(SCRIPTS / "verify_audit_with_openssl.sh"), str(f), anchor], capture_output=True, text=True, env={**os.environ, "SINTETICO_AUDIT_KEY": "s3cret"})
    assert t.returncode == 1 and "TRUNCATED" in t.stdout


def test_openssl_agrees_with_python_on_evidence_packages(tmp_path):
    pkg = build_evidence_package("c0ffee", [StageResult("sast-analysis", "PASS", "sonar"), StageResult("iac-scan", "FAIL", "tf", 2)], "s3cret",
                                 agent={"prompt_sha256": "ab" * 32, "model_version": "m", "eval_pass_rate": 0.85, "trust_score": 0.81,
                                        "human_approval": {"approver": "ñ@x.com", "approved_at": "2026-01-01T00:00:00Z"}})
    p = EvidenceStore(tmp_path).write(pkg)
    assert sh("verify_evidence_with_openssl.sh", str(p), key_var="SINTETICO_EVIDENCE_KEY").returncode == 0
    tampered = json.loads(p.read_text()); tampered["overall_result"] = "PASS"
    q = tmp_path / "t.json"; q.write_text(json.dumps(tampered))
    r = sh("verify_evidence_with_openssl.sh", str(q), key_var="SINTETICO_EVIDENCE_KEY")
    assert r.returncode == 1 and "INVALID" in r.stdout
