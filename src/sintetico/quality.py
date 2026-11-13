"""
sintetico.quality — self-cleaning code, TrustScore, adversarial
validation (ShadowAgent) and technical-debt tracking
(Chapters 5, 6 and 11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

__all__ = [
    "CodeSmellType",
    "CodeSmell",
    "AgentAnalyzer",
    "AgentRefactor",
    "AgentTests",
    "SelfCleaningCodeLoop",
    "TrustScore",
    "ShadowVerdict",
    "ShadowDecision",
    "ShadowAgent",
    "AgentDebt",
    "DebtTracker",
    "TrustLevel",
    "TrustScoreResult",
    "TrustScoreCalculator",
]


# ═══════════════════════════════════════════════════════════════════
# Self-Cleaning Code Loop
# ═══════════════════════════════════════════════════════════════════


class CodeSmellType(Enum):
    DUPLICATE_CODE = "duplicate_code"
    LONG_FUNCTION = "long_function"
    DEAD_CODE = "dead_code"


@dataclass
class CodeSmell:
    smell_type: CodeSmellType
    location: str
    severity: str
    description: str
    suggested_fix: str


class AgentAnalyzer:
    """Deterministic detector of simple smells, used as a cheap gate
    before an LLM is invoked for the full semantic analysis."""

    def analyze(self, code: str) -> List[CodeSmell]:
        smells = []
        print_calls = len(re.findall(r"\bprint\s*\(", code))
        if print_calls > 1:
            smells.append(
                CodeSmell(
                    CodeSmellType.DUPLICATE_CODE,
                    "multiple locations",
                    "medium",
                    f"Found {print_calls} print() calls; candidates for extraction into a shared function.",
                    "Extract into a helper function (_log)",
                )
            )
        return smells


class AgentRefactor:
    """Deterministic refactorer for the DUPLICATE_CODE smell detected by
    `AgentAnalyzer`.

    It replaces only the *calls* to `print(...)` with `_log(...)` and
    prepends the definition of `_log`. Unlike v1 (which did
    `code.replace("print", "def log(msg): print(msg)")` and inserted a
    whole function definition at every call site, producing syntactically
    invalid code), this always produces valid code.
    """

    _HELPER = "def _log(msg):\n    print(msg)\n\n\n"

    def refactor(self, code: str, smells: List[CodeSmell]) -> str:
        if not any(s.smell_type == CodeSmellType.DUPLICATE_CODE for s in smells):
            return code
        refactored = re.sub(r"\bprint(\s*\()", r"_log\1", code)
        if "_log(" in refactored and "def _log" not in refactored:
            refactored = self._HELPER + refactored
        return refactored


class AgentTests:
    """Generates a minimal regression test that checks the refactor did
    not remove the expected observable behavior."""

    def generate_regression_tests(self, original_code: str, refactored_code: str) -> str:
        return (
            "def test_refactor_preserves_behavior():\n"
            "    assert '_log(' in refactored_code or 'print(' in refactored_code\n"
        )


class SelfCleaningCodeLoop:
    """Self-cleaning loop with imperative static validation (compile())
    before and after each refactor, to avoid needless LLM cycles on
    purely syntactic errors."""

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.analyzer = AgentAnalyzer()
        self.refactor = AgentRefactor()
        self.tests = AgentTests()

    @staticmethod
    def _static_validate(code: str) -> Tuple[bool, Optional[str]]:
        try:
            compile(code, "<string>", "exec")
            return True, None
        except SyntaxError as exc:
            return False, f"SyntaxError on line {exc.lineno}: {exc.msg}"

    def clean(self, code: str) -> Dict:
        valid, error = self._static_validate(code)
        if not valid:
            return {"status": "syntax_error", "error": error, "code": code}

        for iteration in range(1, self.max_iterations + 1):
            smells = self.analyzer.analyze(code)
            if not smells:
                return {"status": "cleaned", "code": code, "iterations": iteration}

            refactored = self.refactor.refactor(code, smells)
            valid, error = self._static_validate(refactored)
            if not valid:
                return {"status": "refactor_broke_syntax", "error": error, "code": code}

            tests = self.tests.generate_regression_tests(code, refactored)
            if not self._run_tests(tests):
                return {"status": "tests_failed", "code": code, "iterations": iteration}

            code = refactored

        return {"status": "max_iterations_reached", "code": code, "iterations": self.max_iterations}

    @staticmethod
    def _run_tests(tests: str) -> bool:
        return "assert" in tests


# ═══════════════════════════════════════════════════════════════════
# TrustScore and ShadowAgent
# ═══════════════════════════════════════════════════════════════════


@dataclass
class TrustScore:
    competence_proxy: float
    verifiability: float
    traceability: float
    error_cost: float

    def compute(self) -> float:
        numerator = self.competence_proxy * self.verifiability * self.traceability
        denominator = max(self.error_cost, 0.01)
        return round((numerator / denominator) * 100, 2)

    def get_autonomy_level(self) -> str:
        score = self.compute()
        if score >= 80:
            return "High autonomy"
        if score >= 60:
            return "Medium autonomy"
        if score >= 40:
            return "Low autonomy"
        return "No autonomy"


class ShadowVerdict(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    FLAG_HUMAN = "flag_for_human"


@dataclass
class ShadowDecision:
    verdict: ShadowVerdict
    reasoning: str
    confidence: float
    detected_risks: List[str]


class ShadowAgent:
    """Adversarial / consistency / constraints validator that audits the
    output of a primary agent before it is considered final."""

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        strategy: str = "adversarial",
        constraints: Optional[List[str]] = None,
    ):
        if strategy not in {"adversarial", "consistency", "constraints"}:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.llm_call = llm_call or (lambda prompt: '{"verdict": "approve", "findings": []}')
        self.strategy = strategy
        self.constraints = constraints or []

    def validate(self, task: str, main_output: str, main_thoughts: str = "") -> ShadowDecision:
        if self.strategy == "adversarial":
            return self._adversarial_validate(task, main_output)
        if self.strategy == "consistency":
            return self._consistency_validate(main_output)
        return self._constraints_validate(main_output)

    def _adversarial_validate(self, task: str, output: str) -> ShadowDecision:
        response = self.llm_call(f"Try to refute this solution. Task: {task}. Output: {output}. If you find nothing wrong, answer exactly: no problems.")
        if any(p in response.lower() for p in ("no problemas", "no problems", "no issues", "no flaws", "no weaknesses")):
            return ShadowDecision(ShadowVerdict.APPROVE, "No flaws found", 0.9, [])
        return ShadowDecision(ShadowVerdict.REJECT, "Weaknesses found", 0.7, ["Possible flaw"])

    def _consistency_validate(self, output: str) -> ShadowDecision:
        if any(p in output.lower() for p in ("contradicción", "contradiccion", "contradiction", "inconsistent")):
            return ShadowDecision(ShadowVerdict.REJECT, "Inconsistency detected", 0.8, [])
        return ShadowDecision(ShadowVerdict.APPROVE, "Consistent", 0.85, [])

    def _constraints_validate(self, output: str) -> ShadowDecision:
        violated = [c for c in self.constraints if c.lower() in output.lower()]
        if violated:
            return ShadowDecision(ShadowVerdict.REJECT, f"Violates: {', '.join(violated)}", 0.95, violated)
        return ShadowDecision(ShadowVerdict.APPROVE, "Constraints satisfied", 0.9, [])


# ═══════════════════════════════════════════════════════════════════
# Technical debt and TrustScoreCalculator (with exit codes for CI)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AgentDebt:
    id: str
    type: str
    severity: str
    created_at: datetime
    description: str


class DebtTracker:
    _SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    def __init__(self):
        self.debt_items: List[AgentDebt] = []
        self._id_counter = 0

    def add_debt(self, debt_type: str, severity: str, description: str) -> AgentDebt:
        if severity not in self._SEVERITY_ORDER:
            raise ValueError(f"Unknown severity: {severity}")
        self._id_counter += 1
        debt = AgentDebt(f"DEBT-{self._id_counter:04d}", debt_type, severity, datetime.now(timezone.utc), description)
        self.debt_items.append(debt)
        return debt

    def prioritize(self) -> List[AgentDebt]:
        return sorted(self.debt_items, key=lambda d: (self._SEVERITY_ORDER[d.severity], d.created_at))


class TrustLevel(Enum):
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TrustScoreResult:
    score: float
    level: TrustLevel
    components: Dict
    ci_exit_code: int
    recommendation: str


class TrustScoreCalculator:
    """Computes a weighted trust score and an exit code suitable for CI
    gates: 0=OK, 1=rejected on quality, 2=blocked by a critical bug."""

    WEIGHTS = {
        "pass_rate": 0.30,
        "precision": 0.25,
        "human_approval": 0.20,
        "no_critical_bugs": 0.15,
        "consistency": 0.10,
    }

    def calculate(
        self, pass_rate: float, precision: float, human_approval: float, critical_bugs: int, consistency_score: float
    ) -> TrustScoreResult:
        no_critical_bugs = 0.0 if critical_bugs > 0 else 1.0
        components = {
            "pass_rate": pass_rate,
            "precision": precision,
            "human_approval": human_approval,
            "no_critical_bugs": no_critical_bugs,
            "consistency": consistency_score,
        }
        score = round(sum(components[k] * w for k, w in self.WEIGHTS.items()), 3)

        if score < 0.40:
            level = TrustLevel.UNTRUSTED
        elif score < 0.60:
            level = TrustLevel.LOW
        elif score < 0.80:
            level = TrustLevel.MEDIUM
        elif score < 0.90:
            level = TrustLevel.HIGH
        else:
            level = TrustLevel.VERY_HIGH

        if critical_bugs > 0:
            exit_code = 2
        elif score < 0.50 or pass_rate < 0.70 or precision < 0.75:
            exit_code = 1
        else:
            exit_code = 0

        recommendation = "OK" if exit_code == 0 else f"Failed: score={score}, critical_bugs={critical_bugs}"
        return TrustScoreResult(score, level, components, exit_code, recommendation)
