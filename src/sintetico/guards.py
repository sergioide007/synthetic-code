"""Guards for autonomous incident handling (book Chapters 13 and 16)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["IRREVERSIBLE_PATTERNS", "GuardDecision", "DetectionGuard"]

# The "decisions that always require a person" (Chapter 16), as detectable patterns.
IRREVERSIBLE_PATTERNS = {
    "database_schema": r"\b(ALTER\s+TABLE|DROP\s+(COLUMN|TABLE|DATABASE)|TRUNCATE)\b",
    "payment_processing": r"\b(payment|refund|chargeback|payout)s?\b",
    "data_deletion": r"\b(DELETE\s+FROM|purge|hard[-_ ]delete)\b",
    "regulatory_compliance": r"\b(GDPR|HIPAA|PCI[-_ ]?DSS|SOX)\b",
    "infrastructure_migration": r"\b(terraform\s+destroy|migrate\s+(region|cloud|provider))\b",
}


@dataclass
class GuardDecision:
    pause_for_human: bool
    blocked_irreversible: bool
    reasons: List[str] = field(default_factory=list)


class DetectionGuard:
    """Decide when automatic debugging must stop and a person must step in."""

    def __init__(self, max_users_affected_pct: float = 50.0, max_minutes_unresolved: float = 15.0,
                 max_minutes_unvalidated: float = 5.0) -> None:
        self.max_users = max_users_affected_pct
        self.max_unresolved = max_minutes_unresolved
        self.max_unvalidated = max_minutes_unvalidated
        self._patterns = {k: re.compile(v, re.I) for k, v in IRREVERSIBLE_PATTERNS.items()}

    def irreversible_matches(self, patch_text: str) -> List[str]:
        return [name for name, rx in self._patterns.items() if rx.search(patch_text)]

    def check(self, users_affected_pct: float = 0.0, minutes_unresolved: float = 0.0, hypothesis_validated: bool = True,
              minutes_since_hypothesis: float = 0.0, patch_text: Optional[str] = None) -> GuardDecision:
        reasons: List[str] = []
        blocked = False
        if patch_text:
            hits = self.irreversible_matches(patch_text)
            if hits:
                blocked = True
                reasons.append("irreversible change detected: " + ", ".join(hits))   # never overridden by agent confidence
        if users_affected_pct > self.max_users:
            reasons.append(f"{users_affected_pct:.0f}% of users affected (> {self.max_users:.0f}%)")
        if minutes_unresolved > self.max_unresolved:
            reasons.append(f"unresolved for {minutes_unresolved:.0f} min (> {self.max_unresolved:.0f})")
        if not hypothesis_validated and minutes_since_hypothesis > self.max_unvalidated:
            reasons.append(f"hypothesis unvalidated for {minutes_since_hypothesis:.0f} min (> {self.max_unvalidated:.0f})")
        return GuardDecision(pause_for_human=bool(reasons), blocked_irreversible=blocked, reasons=reasons)
