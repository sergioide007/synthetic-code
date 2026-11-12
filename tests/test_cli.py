import json, subprocess, sys
from pathlib import Path
import pytest
from sintetico import cli
from sintetico.audit import AgentAuditLogger
from sintetico.devops.compliance import EvidenceStore, StageResult, build_evidence_package

FIX = Path(__file__).parent / "fixtures"


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else {}


def results(tmp_path, **kw):
    d = {"pass_rate": 0.9, "precision": 0.9, "human_approval": 1.0, "critical_bugs": 0, "consistency_score": 1.0}
    d.update(kw)
    p = tmp_path / "r.json"; p.write_text(json.dumps(d)); return str(p)


def test_gate_exit_codes(tmp_path, capsys):
    assert run(capsys, "gate", "--results", results(tmp_path))[0] == 0
    code, out = run(capsys, "gate", "--results", results(tmp_path, critical_bugs=1))
    assert code == 2 and out["exit_code"] == 2
    assert run(capsys, "gate", "--results", results(tmp_path, pass_rate=0.6, precision=0.6))[0] == 1
    assert run(capsys, "gate", "--results", results(tmp_path), "--min-pass-rate", "0.95")[0] == 1
    assert run(capsys, "gate", "--results", results(tmp_path), "--min-trust", "0.99")[0] == 1


def test_gate_detects_drift_only_when_otherwise_healthy(tmp_path, capsys):
    weak = results(tmp_path, human_approval=0.0)                         # score 0.745: passes the absolute gate
    code, out = run(capsys, "gate", "--results", weak, "--previous-score", "0.95")
    assert code == 3 and "dropped" in out["reasons"][0]                   # but fell 0.205 vs last release
    assert run(capsys, "gate", "--results", weak, "--previous-score", "0.80")[0] == 0     # a 0.055 dip is tolerated
    crit = results(tmp_path, critical_bugs=1)
    assert run(capsys, "gate", "--results", crit, "--previous-score", "0.99")[0] == 2     # critical bugs win over drift


def test_eval_command_runs_user_agent(monkeypatch, capsys):
    monkeypatch.chdir(FIX)
    code, out = run(capsys, "eval", "--evalset", "sample_evalset.json", "--agent", "demo_agent:answer")
    assert code == 0 and out["pass_rate"] == 1.0
    code, out = run(capsys, "eval", "--evalset", "sample_evalset.json", "--agent", "demo_agent:leaky")
    assert code == 2 and out["critical_failures"] == 1                    # a critical failure beats the pass-rate rule
    with pytest.raises(SystemExit):
        cli.main(["eval", "--evalset", "sample_evalset.json", "--agent", "no_colon"])


def test_audit_verify(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SINTETICO_AUDIT_KEY", "k")
    f = tmp_path / "a.jsonl"
    log = AgentAuditLogger("k", path=f); log.log("a", "x", cost_usd=1)
    code, out = run(capsys, "audit-verify", "--file", str(f))
    assert code == 0 and out["chain_valid"]
    f.write_text(f.read_text().replace('"cost_usd": 1', '"cost_usd": 0'))
    code, out = run(capsys, "audit-verify", "--file", str(f))
    assert code == 1 and out["chain_valid"] is False
    monkeypatch.delenv("SINTETICO_AUDIT_KEY")
    with pytest.raises(SystemExit):
        cli.main(["audit-verify", "--file", str(f)])


def test_evidence_verify(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SINTETICO_EVIDENCE_KEY", "k")
    EvidenceStore(tmp_path).write(build_evidence_package("c1", [StageResult("sast-analysis", "PASS", "sonar")], "k"))
    assert run(capsys, "evidence-verify", str(tmp_path))[0] == 0
    monkeypatch.setenv("SINTETICO_EVIDENCE_KEY", "wrong")
    assert run(capsys, "evidence-verify", str(tmp_path))[0] == 1
    empty = tmp_path / "empty"; empty.mkdir()
    assert run(capsys, "evidence-verify", str(empty))[0] == 1             # nothing to verify is not a pass


def test_iac_review_command(tmp_path, capsys):
    tags = {"env": "p", "owner": "o", "compliance": "c"}
    plan = {"resource_changes": [{"address": "aws_db_instance.l", "type": "aws_db_instance", "change": {"actions": ["update"],
            "before": {"identifier": "l", "instance_class": "db.r6g.2xlarge", "tags": tags},
            "after": {"identifier": "l", "instance_class": "db.r6g.large", "tags": tags}}}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "m.json").write_text(json.dumps({"l": {"p99_cpu_pct": 94, "min_freeable_memory_gb": 36}}))
    code, out = run(capsys, "iac-review", "--plan", str(tmp_path / "plan.json"), "--metrics", str(tmp_path / "m.json"))
    assert code == 1 and out["decision"] == "BLOCK"
    code, out = run(capsys, "iac-review", "--plan", str(tmp_path / "plan.json"))
    assert code == 2 and out["decision"] == "REVIEW"


def test_console_entry_point_end_to_end(tmp_path):
    """The real CLI as a CI job would call it: the exit code carries the verdict."""
    r = tmp_path / "r.json"; r.write_text(json.dumps({"pass_rate": 0.9, "precision": 0.9, "human_approval": 1, "critical_bugs": 1}))
    proc = subprocess.run([sys.executable, "-m", "sintetico.cli", "gate", "--results", str(r)], capture_output=True, text=True)
    assert proc.returncode == 2 and json.loads(proc.stdout)["exit_code"] == 2
    assert subprocess.run([sys.executable, "-m", "sintetico.cli", "--version"], capture_output=True, text=True).stdout.startswith("sintetico ")
