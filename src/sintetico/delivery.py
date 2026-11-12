"""Progressive delivery, architecture review, shadow gate and trust-gated swarms (book Chapters 2, 6, 11)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import AbstractSet, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .quality import ShadowAgent, ShadowDecision, ShadowVerdict

__all__ = ["DeployPhase", "DeployMetrics", "AdvanceDecision", "DeployGate", "Answer", "Question", "ReviewStatus",
           "ArchitectureReview", "AgentArchitectureReviewer", "DEFAULT_QUESTIONS", "PipelineDecision", "PipelineResult",
           "ProductionPipeline", "SwarmReport", "SwarmOrchestrator"]


class DeployPhase(enum.Enum):
    CANARY = 10
    SLOW_ROLLOUT = 50
    FULL = 100


@dataclass
class DeployMetrics:
    pass_rate: float
    error_rate: float
    trust_score: float = 1.0
    p99_latency_s: float = 0.0


@dataclass
class AdvanceDecision:
    advance: bool
    current: DeployPhase
    next: Optional[DeployPhase]
    reasons: List[str] = field(default_factory=list)


class DeployGate:
    """canary 10% -> 50% -> 100%, advancing only on evidence; any single bad signal rolls back."""

    ROLLBACK = dict((("error_rate", 0.20), ("pass_rate", 0.70), ("trust_score", 0.40), ("p99_latency_s", 5.0)))

    def __init__(self) -> None:
        self.phase = DeployPhase.CANARY
        self.rolled_back = False

    @property
    def traffic_percent(self) -> int:
        return 0 if self.rolled_back else self.phase.value

    def evaluate(self, m: DeployMetrics) -> AdvanceDecision:
        if self.phase is DeployPhase.CANARY:
            problems = []
            if m.pass_rate < 0.80: problems.append(f"pass_rate {m.pass_rate:.2f} < 0.80")
            if m.error_rate > 0.10: problems.append(f"error_rate {m.error_rate:.2f} > 0.10")
            return AdvanceDecision(not problems, self.phase, DeployPhase.SLOW_ROLLOUT if not problems else None, problems)
        if self.phase is DeployPhase.SLOW_ROLLOUT:
            problems = []
            if m.pass_rate < 0.85: problems.append(f"pass_rate {m.pass_rate:.2f} < 0.85")
            if m.trust_score < 0.60: problems.append(f"trust_score {m.trust_score:.2f} < 0.60")
            return AdvanceDecision(not problems, self.phase, DeployPhase.FULL if not problems else None, problems)
        return AdvanceDecision(False, self.phase, None, ["already at 100%"])

    def advance(self, m: DeployMetrics) -> DeployPhase:
        d = self.evaluate(m)
        if d.advance and d.next:
            self.phase = d.next
        return self.phase

    def should_rollback(self, m: DeployMetrics) -> Tuple[bool, List[str]]:
        reasons = []
        if m.error_rate > self.ROLLBACK["error_rate"]: reasons.append(f"error_rate {m.error_rate:.2f} > 0.20")
        if m.pass_rate < self.ROLLBACK["pass_rate"]: reasons.append(f"pass_rate {m.pass_rate:.2f} < 0.70")
        if m.trust_score < self.ROLLBACK["trust_score"]: reasons.append(f"trust_score {m.trust_score:.2f} < 0.40")
        if m.p99_latency_s > self.ROLLBACK["p99_latency_s"]: reasons.append(f"p99 latency {m.p99_latency_s:.1f}s > 5s")
        return bool(reasons), reasons

    def rollback_if_needed(self, m: DeployMetrics) -> bool:
        bad, _ = self.should_rollback(m)
        if bad:
            self.rolled_back = True
        return bad


class Answer(enum.Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"


class ReviewStatus(enum.Enum):
    REJECTED = "rejected"
    APPROVED_WITH_CHANGES = "approved_with_changes"
    NEEDS_INFORMATION = "needs_information"
    APPROVED = "approved"


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str


DEFAULT_QUESTIONS: Tuple[Question, ...] = (
    Question("eval_set", "evaluation", "Is there a live eval-set with 50+ representative samples?"),
    Question("eval_ci", "evaluation", "Do evals run in CI and block the merge below the threshold?"),
    Question("immutable_audit", "security", "Is every agent action written to an immutable audit log?"),
    Question("least_privilege", "security", "Does each agent/tool have least-privilege access?"),
    Question("pii_filter", "security", "Is agent output scanned for PII/secrets before leaving the system?"),
    Question("raci", "autonomy", "Is a RACI defined with a human as Accountable for every activity?"),
    Question("autonomy_gate", "autonomy", "Is autonomy earned progressively (not full autonomy on day one)?"),
    Question("cost_tracking", "costs", "Is spend tracked per team with budgets and alerts?"),
    Question("structured_logs", "observability", "Are logs structured with a correlation_id per task?"),
    Question("srp_agents", "architecture", "Does each agent have a single responsibility?"),
)
CRITICAL_CATEGORIES = {"security", "evaluation"}


@dataclass
class ArchitectureReview:
    status: ReviewStatus
    non_compliant: List[Question]
    missing: List[Question]
    critical_failures: List[Question]


class AgentArchitectureReviewer:
    def __init__(self, questions: Sequence[Question] = DEFAULT_QUESTIONS, critical_categories: AbstractSet[str] = frozenset(CRITICAL_CATEGORIES)) -> None:
        self.questions = list(questions)
        self.critical = set(critical_categories)

    def review(self, answers: Mapping[str, Answer]) -> ArchitectureReview:
        nc = [q for q in self.questions if answers.get(q.id) is Answer.NON_COMPLIANT]
        missing = [q for q in self.questions if q.id not in answers]
        crit = [q for q in nc if q.category in self.critical]
        if crit:
            status = ReviewStatus.REJECTED
        elif nc:
            status = ReviewStatus.APPROVED_WITH_CHANGES
        elif missing:
            status = ReviewStatus.NEEDS_INFORMATION
        else:
            status = ReviewStatus.APPROVED
        return ArchitectureReview(status, nc, missing, crit)


class PipelineDecision(enum.Enum):
    AUTO_APPROVED = "auto_approved"
    REJECTED = "rejected"
    ESCALATED_TO_HUMAN = "escalated_to_human"


@dataclass
class PipelineResult:
    output: str
    decision: PipelineDecision
    shadow: Optional[ShadowDecision] = None
    note: str = ""


class ProductionPipeline:
    """Main agent -> Shadow validation -> {auto-approve | reject | escalate}.

    Auto-approval requires an APPROVE verdict *and* confidence above the threshold.
    Any shadow failure escalates to a human; it never approves silently."""

    def __init__(self, main_agent: Callable[[str], str], shadow: ShadowAgent, confidence_threshold: float = 0.8) -> None:
        self.main_agent = main_agent
        self.shadow = shadow
        self.threshold = confidence_threshold

    def run(self, task: str) -> PipelineResult:
        output = self.main_agent(task)
        try:
            d = self.shadow.validate(task, output)
        except Exception as exc:  # noqa: BLE001
            return PipelineResult(output, PipelineDecision.ESCALATED_TO_HUMAN, None, f"shadow failed: {exc}")
        if d.verdict is ShadowVerdict.REJECT:
            return PipelineResult(output, PipelineDecision.REJECTED, d, d.reasoning)
        if d.verdict is ShadowVerdict.APPROVE and d.confidence >= self.threshold:
            return PipelineResult(output, PipelineDecision.AUTO_APPROVED, d, d.reasoning)
        return PipelineResult(output, PipelineDecision.ESCALATED_TO_HUMAN, d, "needs human review")


@dataclass
class SwarmReport:
    results: Dict[str, str] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)

    def consolidated(self) -> str:
        return "\n".join(f"[{k}] {v}" for k, v in self.results.items())


class SwarmOrchestrator:
    """Decompose a task, check each agent's trust gate and approval requirement, delegate, consolidate."""

    def __init__(self, agents: Mapping[str, Callable[[str], str]], trust_scores: Mapping[str, float],
                 roles_requiring_approval: Optional[Set[str]] = None, min_trust: float = 0.65,
                 approver: Optional[Callable[[str, str], bool]] = None) -> None:
        self.agents = dict(agents)
        self.trust = dict(trust_scores)
        self.needs_approval = set(roles_requiring_approval or ())
        self.min_trust = min_trust
        self.approver = approver

    def run(self, task: str, decompose: Callable[[str], List[Tuple[str, str]]]) -> SwarmReport:
        report = SwarmReport()
        for i, (role, subtask) in enumerate(decompose(task)):
            key = f"{i}:{role}"
            if role not in self.agents:
                report.skipped[key] = "no such agent"
            elif self.trust.get(role, 0.0) < self.min_trust:
                report.skipped[key] = f"trust gate: {self.trust.get(role, 0.0):.2f} < {self.min_trust:.2f}"
            elif role in self.needs_approval and not (self.approver and self.approver(role, subtask)):
                report.skipped[key] = "awaiting human approval"
            else:
                report.results[key] = self.agents[role](subtask)
        return report
