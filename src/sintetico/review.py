"""Code-review agent and human-in-the-loop patterns (book Chapter 12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .adapters import OutputAdapter, OutputParseError
from .providers import LLMProvider, LLMRequest

__all__ = ["BugFinding", "CodeReviewAgent", "ReviewOutcome", "HumanInLoopReviewer", "ConfidenceEscalationReviewer"]

SYSTEM_PROMPT = """You are a senior code reviewer. Find LOGIC bugs only (not style, not naming).
For each bug give the line, a description and a minimal failing test.
Answer with JSON only: {"bugs": [{"line": int, "type": str, "severity": "critical|high|medium|low",
"description": str, "test": str, "confidence": number between 0 and 1}]}.
If there are no bugs answer {"bugs": []}."""

SEVERITIES = ("critical", "high", "medium", "low")


@dataclass
class BugFinding:
    line: int
    type: str
    severity: str
    description: str
    test: str = ""
    confidence: float = 0.5


class CodeReviewAgent:
    def __init__(self, provider: LLMProvider, model: Optional[str] = None) -> None:
        self.provider = provider
        self.model = model

    def review(self, diff: str, context: str = "") -> List[BugFinding]:
        """Never raises on a malformed answer: a parsing failure must not take the pipeline down."""
        user = f"Repository context:\n{context}\n\nDiff:\n{diff}" if context else f"Diff:\n{diff}"
        resp = self.provider.complete(LLMRequest(SYSTEM_PROMPT, [{"role": "user", "content": user}], model=self.model, temperature=0.0))
        try:
            data = OutputAdapter.extract_json(resp.content)
            raw = data.get("bugs", []) if isinstance(data, dict) else []
        except OutputParseError:
            return []
        findings: List[BugFinding] = []
        for b in raw:
            try:
                sev = str(b.get("severity", "medium")).lower()
                findings.append(BugFinding(int(b["line"]), str(b.get("type", "unknown")), sev if sev in SEVERITIES else "medium",
                                           str(b.get("description", "")), str(b.get("test", "")), float(b.get("confidence", 0.5))))
            except (KeyError, TypeError, ValueError):
                continue
        return findings


@dataclass
class ReviewOutcome:
    blocking: bool
    findings: List[BugFinding]
    reason: str = ""


class HumanInLoopReviewer:
    """Never auto-merge when a critical bug is found: notify a person and block."""

    def __init__(self, reviewer: CodeReviewAgent, notify: Callable[[List[BugFinding]], None]) -> None:
        self.reviewer = reviewer
        self.notify = notify

    def review(self, diff: str) -> ReviewOutcome:
        findings = self.reviewer.review(diff)
        critical = [f for f in findings if f.severity == "critical"]
        if critical:
            self.notify(critical)
            return ReviewOutcome(True, findings, f"{len(critical)} critical finding(s): human review required")
        return ReviewOutcome(False, findings, "no critical findings")


class ConfidenceEscalationReviewer:
    """Split findings by the agent's own declared confidence; low-confidence ones go to a person."""

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def split(self, findings: Sequence[BugFinding]) -> Tuple[List[BugFinding], List[BugFinding]]:
        return ([f for f in findings if f.confidence >= self.threshold], [f for f in findings if f.confidence < self.threshold])

    def escalation_rate(self, findings: Sequence[BugFinding]) -> float:
        return len(self.split(findings)[1]) / len(findings) if findings else 0.0
