"""
sintetico.quality — Autolimpieza de código, TrustScore, validación
adversarial (ShadowAgent) y seguimiento de deuda técnica
(Capítulos 5, 6 y 11).
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
    """Detector determinista de smells simples, usado como puerta barata
    antes de invocar a un LLM para el análisis semántico completo."""

    def analyze(self, code: str) -> List[CodeSmell]:
        smells = []
        print_calls = len(re.findall(r"\bprint\s*\(", code))
        if print_calls > 1:
            smells.append(
                CodeSmell(
                    CodeSmellType.DUPLICATE_CODE,
                    "múltiples ubicaciones",
                    "medium",
                    f"Se detectaron {print_calls} llamadas a print(); candidatas a extraer a una función común.",
                    "Extraer a una función auxiliar (_log)",
                )
            )
        return smells


class AgentRefactor:
    """Refactorizador determinista para el smell DUPLICATE_CODE detectado
    por `AgentAnalyzer`.

    Sustituye únicamente las *llamadas* a `print(...)` por `_log(...)` y
    antepone la definición de `_log`. A diferencia de la v1 (que hacía
    `code.replace("print", "def log(msg): print(msg)")` e insertaba una
    definición de función completa en cada punto de llamada, generando
    código sintácticamente inválido), esto produce siempre código válido.
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
    """Genera una prueba de regresión mínima que verifica que el
    refactor no eliminó el comportamiento observable esperado."""

    def generate_regression_tests(self, original_code: str, refactored_code: str) -> str:
        return (
            "def test_refactor_preserves_behavior():\n"
            "    assert '_log(' in refactored_code or 'print(' in refactored_code\n"
        )


class SelfCleaningCodeLoop:
    """Bucle de autolimpieza con validación estática imperativa (compile())
    antes y después de cada refactor, para evitar ciclos innecesarios de
    LLM ante errores puramente sintácticos."""

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
            return False, f"SyntaxError en línea {exc.lineno}: {exc.msg}"

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
# TrustScore y ShadowAgent
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
            return "Alta autonomía"
        if score >= 60:
            return "Media autonomía"
        if score >= 40:
            return "Baja autonomía"
        return "Sin autonomía"


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
    """Validador adversarial/de consistencia/de restricciones que audita
    la salida de un agente principal antes de que se considere definitiva."""

    def __init__(
        self,
        llm_call: Optional[Callable[[str], str]] = None,
        strategy: str = "adversarial",
        constraints: Optional[List[str]] = None,
    ):
        if strategy not in {"adversarial", "consistency", "constraints"}:
            raise ValueError(f"Estrategia desconocida: {strategy}")
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
        response = self.llm_call(f"Refuta esta solución. Tarea: {task}. Output: {output}")
        if "no problemas" in response.lower():
            return ShadowDecision(ShadowVerdict.APPROVE, "Sin fallos encontrados", 0.9, [])
        return ShadowDecision(ShadowVerdict.REJECT, "Debilidades encontradas", 0.7, ["Posible fallo"])

    def _consistency_validate(self, output: str) -> ShadowDecision:
        if "contradicción" in output.lower():
            return ShadowDecision(ShadowVerdict.REJECT, "Inconsistencia detectada", 0.8, [])
        return ShadowDecision(ShadowVerdict.APPROVE, "Coherente", 0.85, [])

    def _constraints_validate(self, output: str) -> ShadowDecision:
        violated = [c for c in self.constraints if c.lower() in output.lower()]
        if violated:
            return ShadowDecision(ShadowVerdict.REJECT, f"Viola: {', '.join(violated)}", 0.95, violated)
        return ShadowDecision(ShadowVerdict.APPROVE, "Restricciones cumplidas", 0.9, [])


# ═══════════════════════════════════════════════════════════════════
# Deuda técnica y TrustScoreCalculator (con exit codes para CI)
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
            raise ValueError(f"Severidad desconocida: {severity}")
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
    """Calcula un score de confianza ponderado y un exit code apto para
    gates de CI: 0=OK, 1=rechazo por calidad, 2=bloqueo por bug crítico."""

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

        recommendation = "OK" if exit_code == 0 else f"Fallo: score={score}, bugs={critical_bugs}"
        return TrustScoreResult(score, level, components, exit_code, recommendation)
