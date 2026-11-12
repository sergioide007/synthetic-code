"""Autonomy bounded by verifiability and cost of error (book Chapters 7 and 9)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from .governance import RACILevel

__all__ = ["Verifiability", "DomainRiskProfile", "compute_max_allowed_autonomy", "ActivitySpec", "GateDecision",
           "DynamicThresholdRACIGate", "threshold_for_error_cost"]


class Verifiability(Enum):
    NOT_VERIFIABLE = "not_verifiable"  # no ground truth
    SLOW_MANUAL = "slow_manual"        # verifying takes more than an hour
    QUICK_MANUAL = "quick_manual"      # a person verifies in minutes
    AUTOMATED = "automated"            # tests/checks verify in seconds


@dataclass(frozen=True)
class DomainRiskProfile:
    name: str
    verifiability: Verifiability
    trust_threshold: float = 0.6


def compute_max_allowed_autonomy(profile: DomainRiskProfile) -> float:
    """Maximum autonomy (0..1) an agent may hold in a domain, whatever its competence."""
    v = profile.verifiability
    if v is Verifiability.NOT_VERIFIABLE:
        return 0.0
    if v is Verifiability.SLOW_MANUAL:
        return min(0.1, profile.trust_threshold)
    if v is Verifiability.QUICK_MANUAL:
        return profile.trust_threshold
    return min(1.0, round(profile.trust_threshold * 1.2, 4))


def threshold_for_error_cost(error_cost: float) -> float:
    """Higher cost of error => higher confidence required before acting alone."""
    if error_cost <= 10:
        return 0.2
    if error_cost <= 30:
        return 0.4
    if error_cost <= 60:
        return 0.6
    if error_cost <= 100:
        return 0.8
    return 0.95


@dataclass
class ActivitySpec:
    agent_role: RACILevel
    human_owner: str
    error_cost: float


@dataclass
class GateDecision:
    allowed: bool
    requires_human_approval: bool
    threshold: float
    reason: str
    owner: Optional[str] = None


class DynamicThresholdRACIGate:
    """Authorize agent actions with a threshold that scales with the cost of error.

    Cardinal rule: an agent is never Accountable. Agents that are only Consulted/Informed
    never execute. A Responsible agent executes alone only when its trust score reaches
    the threshold for that activity."""

    def __init__(self, matrix: Dict[str, ActivitySpec]) -> None:
        self.matrix = matrix

    def authorize(self, activity: str, trust_score: float) -> GateDecision:
        if activity not in self.matrix:
            raise KeyError(f"Unknown activity '{activity}'")
        spec = self.matrix[activity]
        thr = threshold_for_error_cost(spec.error_cost)
        if spec.agent_role is RACILevel.ACCOUNTABLE:
            return GateDecision(False, True, thr, "cardinal rule: an agent can never be Accountable", spec.human_owner)
        if spec.agent_role in (RACILevel.CONSULTED, RACILevel.INFORMED):
            return GateDecision(False, False, thr, f"agent role {spec.agent_role.name} does not execute", spec.human_owner)
        if trust_score >= thr:
            return GateDecision(True, False, thr, f"trust {trust_score:.2f} >= threshold {thr:.2f}", spec.human_owner)
        return GateDecision(False, True, thr, f"trust {trust_score:.2f} < threshold {thr:.2f}: human approval required", spec.human_owner)
