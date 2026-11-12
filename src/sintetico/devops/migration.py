"""Supervised migrations: dual-write, field-level checksums, human-gated cutover (book Chapter 26).

The agent (or a script) *proposes* phase advancement with evidence; the irreversible
cutover is always a human decision.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

__all__ = ["row_hash", "BatchReport", "verify_batch", "DualWriteRepository", "Phase", "Recommendation", "MigrationAdvisor", "PHASE_CRITERIA"]


def row_hash(record: Mapping[str, Any], fields: Sequence[str]) -> str:
    """SHA-256 over the *values of the critical fields*, normalised to strings.

    Row counts only prove cardinality; this proves content."""
    canonical = json.dumps({f: str(record.get(f, "")) for f in fields}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class BatchReport:
    size: int
    mismatches: List[Dict[str, Any]] = field(default_factory=list)
    missing_in_secondary: List[str] = field(default_factory=list)

    @property
    def mismatch_rate(self) -> float:
        bad = len(self.mismatches) + len(self.missing_in_secondary)
        return bad / self.size if self.size else 0.0

    @property
    def clean(self) -> bool:
        return not self.mismatches and not self.missing_in_secondary


def verify_batch(ids: Sequence[str], primary: Mapping[str, Mapping[str, Any]], secondary: Mapping[str, Mapping[str, Any]],
                 fields: Sequence[str]) -> BatchReport:
    report = BatchReport(size=len(ids))
    for i in ids:
        if i not in secondary:
            report.missing_in_secondary.append(i)
            continue
        if row_hash(primary[i], fields) != row_hash(secondary[i], fields):
            diff = [f for f in fields if str(primary[i].get(f)) != str(secondary[i].get(f))]
            report.mismatches.append({"id": i, "fields": diff})
    return report


class DualWriteRepository:
    """Primary is authoritative. A secondary failure never fails the write: it goes to a DLQ."""

    def __init__(self, primary: MutableMapping[str, Dict[str, Any]], secondary_write: Callable[[str, Dict[str, Any]], None],
                 dlq: Optional[List[Dict[str, Any]]] = None) -> None:
        self.primary = primary
        self.secondary_write = secondary_write
        self.dlq: List[Dict[str, Any]] = dlq if dlq is not None else []

    def write(self, entity_id: str, record: Dict[str, Any]) -> None:
        self.primary[entity_id] = dict(record)
        try:
            self.secondary_write(entity_id, dict(record))
        except Exception as exc:  # noqa: BLE001
            self.dlq.append({"id": entity_id, "record": dict(record), "error": f"{type(exc).__name__}: {exc}"})

    def reprocess_dlq(self) -> Dict[str, int]:
        """Retry failed secondary writes (call after fixing the root cause)."""
        pending, self.dlq = self.dlq, []
        ok = 0
        for item in pending:
            try:
                self.secondary_write(item["id"], item["record"])
                ok += 1
            except Exception as exc:  # noqa: BLE001
                item["error"] = f"{type(exc).__name__}: {exc}"
                self.dlq.append(item)
        return {"recovered": ok, "still_failing": len(self.dlq)}


class Phase(enum.IntEnum):
    SHADOW = 1
    DUAL_WRITE = 2
    VERIFY = 3
    CUTOVER = 4


PHASE_CRITERIA = {
    Phase.SHADOW: {"min_days": 7, "max_dlq_depth": 100, "min_clean_batches": 0},
    Phase.DUAL_WRITE: {"min_days": 14, "max_dlq_depth": 10, "min_clean_batches": 0},
    Phase.VERIFY: {"min_days": 14, "max_dlq_depth": 0, "min_clean_batches": 3},
}


@dataclass
class Recommendation:
    action: str                      # ADVANCE | HOLD | PROPOSE_CUTOVER | COMPLETE
    requires_human: bool
    blockers: List[str] = field(default_factory=list)
    approvers: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class MigrationAdvisor:
    def __init__(self, cutover_approvers: Sequence[str] = ("principal-engineer", "cto")) -> None:
        self.cutover_approvers = list(cutover_approvers)

    def evaluate(self, phase: Phase, days_in_phase: float, dlq_depth: int, consecutive_clean_batches: int = 0,
                 mismatch_rate: float = 0.0, rollback_tested: bool = False) -> Recommendation:
        if phase is Phase.CUTOVER:
            return Recommendation("COMPLETE", False)
        c = PHASE_CRITERIA[phase]
        blockers: List[str] = []
        if days_in_phase < c["min_days"]:
            blockers.append(f"days_in_phase {days_in_phase} < {c['min_days']}")
        if dlq_depth > c["max_dlq_depth"]:
            blockers.append(f"dlq_depth {dlq_depth} > {c['max_dlq_depth']}")
        if consecutive_clean_batches < c["min_clean_batches"]:
            blockers.append(f"clean batches {consecutive_clean_batches} < {c['min_clean_batches']}")
        if phase is Phase.VERIFY:
            if mismatch_rate > 0:
                blockers.append(f"mismatch_rate {mismatch_rate:.6f} > 0")
            if not rollback_tested:
                blockers.append("rollback plan not tested")
        if blockers:
            return Recommendation("HOLD", False, blockers)
        if phase is Phase.VERIFY:       # the irreversible step: always a human
            return Recommendation("PROPOSE_CUTOVER", True, [], self.cutover_approvers, {
                "days_in_phase": days_in_phase, "consecutive_clean_batches": consecutive_clean_batches,
                "mismatch_rate": mismatch_rate, "dlq_depth": dlq_depth, "rollback_tested": rollback_tested})
        return Recommendation("ADVANCE", False)
