"""Evaluation sets: TDD for probabilistic systems (book Chapters 2, 4, 11, 18).

An :class:`EvalSet` is a list of cases; each case decides pass/fail with a plain
Python callable or declaratively (substring / regex), so the same set can live in
a JSON file and run in CI without any API key (use a deterministic agent or a
recorded one) or against a real model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

__all__ = ["EvalCase", "EvalResult", "EvalReport", "EvalSet", "OptimizationDecision", "CostOptimizerWithEvals", "sha256_text"]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class EvalCase:
    input: str
    check: Optional[Callable[[str], bool]] = None
    expected_contains: Sequence[str] = ()
    forbidden_contains: Sequence[str] = ()
    expected_regex: Optional[str] = None
    critical: bool = False  # a failing critical case blocks CI regardless of pass rate
    tags: Sequence[str] = ()

    def passes(self, output: str) -> bool:
        if self.check is not None and not self.check(output):
            return False
        low = output.lower()
        if any(s.lower() not in low for s in self.expected_contains):
            return False
        if any(s.lower() in low for s in self.forbidden_contains):
            return False
        if self.expected_regex and not re.search(self.expected_regex, output, flags=re.S):
            return False
        return True


@dataclass
class EvalResult:
    input: str
    output: str
    passed: bool
    critical: bool = False
    error: Optional[str] = None


@dataclass
class EvalReport:
    results: List[EvalResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def critical_failures(self) -> int:
        return sum(1 for r in self.results if r.critical and not r.passed)

    def to_dict(self) -> Dict:
        return {"total": self.total, "passed": self.passed, "pass_rate": round(self.pass_rate, 4),
                "critical_failures": self.critical_failures}


class EvalSet:
    def __init__(self, cases: Sequence[EvalCase], name: str = "eval-set", version: str = "1") -> None:
        if not cases:
            raise ValueError("An EvalSet needs at least one case")
        self.cases = list(cases)
        self.name = name
        self.version = version

    def run(self, agent: Callable[[str], str]) -> EvalReport:
        """Run every case. An exception inside ``agent`` counts as a failed case, never a crash."""
        results: List[EvalResult] = []
        for case in self.cases:
            try:
                out = agent(case.input)
                results.append(EvalResult(case.input, out, case.passes(out), case.critical))
            except Exception as exc:  # noqa: BLE001 - evals must survive misbehaving agents
                results.append(EvalResult(case.input, "", False, case.critical, error=f"{type(exc).__name__}: {exc}"))
        return EvalReport(results)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EvalSet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = [EvalCase(input=c["input"], expected_contains=c.get("expected_contains", []),
                          forbidden_contains=c.get("forbidden_contains", []), expected_regex=c.get("expected_regex"),
                          critical=bool(c.get("critical", False)), tags=c.get("tags", [])) for c in data["cases"]]
        return cls(cases, name=data.get("name", Path(path).stem), version=str(data.get("version", "1")))


@dataclass
class OptimizationDecision:
    approved: bool
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_drop: float
    cost_saving_pct: float
    reason: str


class CostOptimizerWithEvals:
    """Approve a cheaper configuration only if measurable quality does not regress.

    Strict on purpose: an accuracy drop above ``max_accuracy_drop`` blocks the change
    no matter how much it saves."""

    def __init__(self, eval_set: EvalSet, max_accuracy_drop: float = 0.02) -> None:
        self.eval_set = eval_set
        self.max_accuracy_drop = max_accuracy_drop
        self._baseline: Optional[tuple] = None

    def measure_baseline(self, agent: Callable[[str], str], cost_per_call: float) -> float:
        report = self.eval_set.run(agent)
        self._baseline = (report.pass_rate, cost_per_call)
        return report.pass_rate

    def validate_candidate(self, agent: Callable[[str], str], cost_per_call: float) -> OptimizationDecision:
        if self._baseline is None:
            raise RuntimeError("Call measure_baseline() first")
        base_acc, base_cost = self._baseline
        cand_acc = self.eval_set.run(agent).pass_rate
        drop = base_acc - cand_acc
        saving = (base_cost - cost_per_call) / base_cost * 100 if base_cost > 0 else 0.0
        if drop > self.max_accuracy_drop + 1e-12:
            return OptimizationDecision(False, base_acc, cand_acc, drop, saving, f"accuracy dropped {drop:.1%} (> {self.max_accuracy_drop:.1%})")
        if saving <= 0:
            return OptimizationDecision(False, base_acc, cand_acc, drop, saving, "no cost saving")
        return OptimizationDecision(True, base_acc, cand_acc, drop, saving, "quality preserved and cost reduced")
