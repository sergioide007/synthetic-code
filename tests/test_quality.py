from sintetico import (
    AgentAnalyzer,
    AgentRefactor,
    DebtTracker,
    SelfCleaningCodeLoop,
    ShadowAgent,
    ShadowVerdict,
    TrustScoreCalculator,
)


class TestSelfCleaningCodeLoop:
    def test_clean_code_returns_immediately(self):
        code = "def f():\n    return 1\n"
        result = SelfCleaningCodeLoop().clean(code)
        assert result["status"] == "cleaned"
        assert result["iterations"] == 1
        assert result["code"] == code

    def test_dirty_code_is_refactored_into_valid_python(self):
        dirty = 'def f():\n    print("a")\n    print("b")\n'
        result = SelfCleaningCodeLoop().clean(dirty)
        assert result["status"] == "cleaned"
        compile(result["code"], "<test>", "exec")  # no debe lanzar SyntaxError
        assert "_log(" in result["code"]

    def test_syntax_error_input_is_rejected_immediately(self):
        result = SelfCleaningCodeLoop().clean("def f(:\n  pass")
        assert result["status"] == "syntax_error"

    def test_refactor_never_produces_invalid_syntax(self):
        # Caso que rompía la v1: reemplazar "print" por una definición de
        # función completa en el propio punto de llamada generaba código
        # sintácticamente inválido.
        dirty = 'print("x")\nprint("y")\nprint("z")\n'
        smells = AgentAnalyzer().analyze(dirty)
        refactored = AgentRefactor().refactor(dirty, smells)
        compile(refactored, "<test>", "exec")


class TestTrustScoreCalculator:
    def test_excellent_pr_passes_ci(self):
        result = TrustScoreCalculator().calculate(0.95, 0.92, 0.98, 0, 0.95)
        assert result.ci_exit_code == 0

    def test_critical_bug_blocks_regardless_of_score(self):
        result = TrustScoreCalculator().calculate(0.99, 0.99, 0.99, 1, 0.99)
        assert result.ci_exit_code == 2

    def test_low_precision_rejects_without_blocking(self):
        result = TrustScoreCalculator().calculate(0.72, 0.65, 0.80, 0, 0.75)
        assert result.ci_exit_code == 1


class TestShadowAgent:
    def test_constraints_strategy_rejects_violation(self):
        agent = ShadowAgent(strategy="constraints", constraints=["borrar toda la base de datos"])
        decision = agent.validate("tarea", "El plan implica borrar toda la base de datos")
        assert decision.verdict == ShadowVerdict.REJECT

    def test_constraints_strategy_approves_clean_output(self):
        agent = ShadowAgent(strategy="constraints", constraints=["borrar toda la base de datos"])
        decision = agent.validate("tarea", "El plan implica un backup incremental")
        assert decision.verdict == ShadowVerdict.APPROVE


class TestDebtTracker:
    def test_prioritizes_by_severity_then_age(self):
        tracker = DebtTracker()
        tracker.add_debt("smell", "low", "menor")
        tracker.add_debt("smell", "critical", "urgente")
        tracker.add_debt("smell", "medium", "media")
        prioritized = tracker.prioritize()
        assert [d.severity for d in prioritized] == ["critical", "medium", "low"]
