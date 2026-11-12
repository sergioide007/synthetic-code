"""Compliance evidence as a pipeline output (book Chapter 21).

Every pipeline run produces a signed evidence package that maps stage results to
control identifiers. Packages are stored write-once (no overwrite, read-only file)
with an append-only index. This gives an assessor raw material; whether it
satisfies a given control is the assessor's decision, not the code's.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

__all__ = ["CONTROL_MAP", "StageResult", "build_evidence_package", "verify_evidence_package", "EvidenceStore", "REQUIRED_AGENT_FIELDS"]

# Illustrative mapping. Replace with the control IDs your auditor actually uses.
CONTROL_MAP: Dict[str, List[str]] = {
    "secret-detection": ["ISO27001-A.8.12", "PCI-DSS-6.4", "SOC2-CC6.1"],
    "sast-analysis": ["ISO27001-A.8.25", "PCI-DSS-6.3", "SOC2-CC6.1"],
    "dependency-scan": ["ISO27001-A.8.8", "PCI-DSS-6.3.3"],
    "container-scan": ["ISO27001-A.8.8", "PCI-DSS-6.3.3"],
    "iac-scan": ["ISO27001-A.8.9", "PCI-DSS-2.2"],
    "agent-eval": ["SOC2-CC7.1"],
}

REQUIRED_AGENT_FIELDS = ("prompt_sha256", "model_version", "eval_pass_rate", "trust_score", "human_approval")


@dataclass
class StageResult:
    stage: str
    status: str            # "PASS" | "FAIL"
    tool: str
    finding_count: int = 0
    artifact: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in ("PASS", "FAIL"):
            raise ValueError("status must be 'PASS' or 'FAIL'")


def _canonical(obj: Dict[str, Any]) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _key(k: Union[str, bytes]) -> bytes:
    if not k:
        raise ValueError("signing key must not be empty")
    return k.encode("utf-8") if isinstance(k, str) else k


def build_evidence_package(commit_sha: str, stages: Iterable[StageResult], signing_key: Union[str, bytes],
                           agent: Optional[Dict[str, Any]] = None, pipeline_run: Optional[str] = None,
                           control_map: Optional[Dict[str, List[str]]] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
    cmap = control_map or CONTROL_MAP
    entries: List[Dict[str, Any]] = []
    satisfied: set = set()
    overall = "PASS"
    for s in stages:
        controls = cmap.get(s.stage, [])
        entries.append({"stage": s.stage, "result": s.status, "tool": s.tool, "findings": s.finding_count,
                        "artifact": s.artifact, "controls": controls})
        if s.status == "FAIL":
            overall = "FAIL"
        else:
            satisfied.update(controls)
    if not entries:
        raise ValueError("an evidence package needs at least one stage")
    pkg: Dict[str, Any] = {
        "schema_version": "1.0", "commit_sha": commit_sha, "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "pipeline_run": pipeline_run, "stages": entries, "controls_satisfied": sorted(satisfied), "overall_result": overall,
    }
    if agent is not None:
        missing = [f for f in REQUIRED_AGENT_FIELDS if f not in agent or agent[f] in (None, "")]
        if missing:
            raise ValueError(f"incomplete agent evidence, missing: {missing}")
        ha = agent["human_approval"]
        if not isinstance(ha, dict) or not ha.get("approver") or not ha.get("approved_at"):
            raise ValueError("human_approval must include 'approver' and 'approved_at'")
        pkg["agent"] = agent
    pkg["hmac_sha256"] = hmac.new(_key(signing_key), _canonical(pkg), hashlib.sha256).hexdigest()
    return pkg


def verify_evidence_package(pkg: Dict[str, Any], signing_key: Union[str, bytes]) -> bool:
    claimed = pkg.get("hmac_sha256")
    if not claimed:
        return False
    body = {k: v for k, v in pkg.items() if k != "hmac_sha256"}
    expected = hmac.new(_key(signing_key), _canonical(body), hashlib.sha256).hexdigest()
    return hmac.compare_digest(claimed, expected)


class EvidenceStore:
    """Write-once local store: ``<root>/<commit>/evidence-package.json`` + append-only ``index.jsonl``.

    On a real cloud use object storage with a compliance-mode retention lock; this class
    gives you the same *semantics* locally and in tests."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.jsonl"

    def _path(self, commit_sha: str) -> Path:
        if not commit_sha or any(c in commit_sha for c in ("/", "\\", "\x00")) or ".." in commit_sha:
            raise ValueError("invalid commit_sha")
        return self.root / commit_sha / "evidence-package.json"

    def write(self, pkg: Dict[str, Any]) -> Path:
        path = self._path(pkg["commit_sha"])
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as fh:        # exclusive create: atomic, no check-then-write race
                fh.write(json.dumps(pkg, indent=2, sort_keys=True))
        except FileExistsError:
            raise FileExistsError(f"evidence for {pkg['commit_sha']} already exists and cannot be overwritten") from None
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        with self.index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"commit_sha": pkg["commit_sha"], "generated_at": pkg["generated_at"],
                                 "overall_result": pkg["overall_result"], "controls_satisfied": pkg["controls_satisfied"],
                                 "path": str(path.relative_to(self.root))}) + "\n")
        return path

    def get(self, commit_sha: str) -> Dict[str, Any]:
        return json.loads(self._path(commit_sha).read_text(encoding="utf-8"))

    def query(self, control: Optional[str] = None, result: Optional[str] = None) -> List[Dict[str, Any]]:
        rows = [json.loads(line) for line in self.index.read_text(encoding="utf-8").splitlines() if line.strip()] if self.index.exists() else []
        if control:
            rows = [r for r in rows if control in r["controls_satisfied"]]
        if result:
            rows = [r for r in rows if r["overall_result"] == result]
        return rows

    def verify_all(self, signing_key: Union[str, bytes]) -> Dict[str, bool]:
        return {r["commit_sha"]: verify_evidence_package(self.get(r["commit_sha"]), signing_key) for r in self.query()}
