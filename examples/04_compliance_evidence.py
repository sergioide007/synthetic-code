"""Compliance as a pipeline output: signed, write-once evidence per commit + data-residency policy as data."""
import tempfile
from pathlib import Path

from sintetico.devops import (EvidenceStore, ResidencyPolicy, StageResult, build_evidence_package, load_profiles,
                              verify_evidence_package)

KEY = "evidence-signing-key"        # load from your secret manager in real life
stages = [StageResult("secret-detection", "PASS", "gitleaks"), StageResult("sast-analysis", "PASS", "sonarqube"),
          StageResult("dependency-scan", "PASS", "trivy")]
agent = {"prompt_sha256": "a3" * 32, "model_version": "claude-sonnet", "eval_pass_rate": 0.85, "trust_score": 0.81,
         "human_approval": {"approver": "lead@company.com", "approved_at": "2026-01-15T10:00:00Z"}}

with tempfile.TemporaryDirectory() as tmp:
    store = EvidenceStore(tmp)
    pkg = build_evidence_package("9f2c1ab", stages, KEY, agent=agent)
    store.write(pkg)
    print("controls satisfied:", pkg["controls_satisfied"])
    print("signature valid   :", verify_evidence_package(store.get("9f2c1ab"), KEY))
    try:
        store.write(pkg)
    except FileExistsError as exc:
        print("overwrite refused :", exc)

policy = ResidencyPolicy(load_profiles(Path(__file__).parent / "profiles"))      # EXAMPLE profiles, not legal advice
print(policy.evaluate_write("BR", "us-east-1"))
print(policy.evaluate_write("BR", "us-east-1", consent={"explicit": True, "country": "BR"}))
assert not policy.evaluate_write("BR", "us-east-1").allowed and policy.evaluate_write("BR", "sa-east-1").allowed
print("OK")
