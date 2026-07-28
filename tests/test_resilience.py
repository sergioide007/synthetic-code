import time

from sintetico import (
    AgentCircuitBreaker,
    AgentHarness,
    AuditLogger,
    CircuitBreaker,
    TokenBudget,
)


class TestCircuitBreaker:
    def test_opens_after_threshold_failures(self):
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state == "open"
        assert breaker.can_execute() is False

    def test_half_open_after_recovery_timeout(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0)
        breaker.record_failure()
        time.sleep(0.01)
        assert breaker.can_execute() is True
        assert breaker.state == "half-open"

    def test_success_resets_breaker(self):
        breaker = CircuitBreaker(failure_threshold=2)
        breaker.record_failure()
        breaker.record_success()
        assert breaker.state == "closed"
        assert breaker.failures == 0


class TestAgentCircuitBreaker:
    def test_opens_on_identical_thought_repetition(self):
        breaker = AgentCircuitBreaker(max_cycles=3)
        for _ in range(3):
            breaker.record_cycle(thought="mismo pensamiento", tools=["search"], observation="obs")
        assert breaker.is_open()

    def test_opens_on_tool_flooding(self):
        breaker = AgentCircuitBreaker(same_tool_threshold=4, window_size=10)
        for i in range(10):
            breaker.record_cycle(thought=f"pensamiento {i}", tools=["search"], observation=f"obs{i}")
        assert breaker.is_open()

    def test_reset_closes_breaker(self):
        breaker = AgentCircuitBreaker(max_cycles=2)
        for _ in range(2):
            breaker.record_cycle(thought="x", tools=[], observation="y")
        assert breaker.is_open()
        breaker.reset()
        assert not breaker.is_open()


class TestAgentHarness:
    def test_run_uses_actual_result_values_not_hardcoded_ones(self):
        """Regresión: la v1 ignoraba el resultado real de agent_fn y
        registraba siempre tokens=100, cost=0.01, model="test-model"."""
        budget = TokenBudget(monthly_budget=10.0)
        breaker = AgentCircuitBreaker()
        audit = AuditLogger("team-x")
        harness = AgentHarness(budget, breaker, audit, model_name="sonnet")

        def agent_fn(task: str) -> dict:
            return {"content": "respuesta real", "tokens_input": 123, "tokens_output": 45, "cost_usd": 0.0042}

        result = harness.run(agent_fn, "tarea de prueba", correlation_id="c1")

        assert result.content == "respuesta real"
        assert result.tokens_used == 123 + 45
        assert audit.logs[-1]["tokens_input"] == 123
        assert audit.logs[-1]["tokens_output"] == 45
        assert audit.logs[-1]["cost_usd"] == 0.0042
        assert audit.logs[-1]["model"] == "sonnet"

    def test_open_breaker_short_circuits_before_spending_budget(self):
        budget = TokenBudget(monthly_budget=10.0)
        breaker = AgentCircuitBreaker(max_cycles=1)
        breaker.record_cycle(thought="x", tools=[], observation="y")
        breaker.state = "open"
        breaker.failure_reason = "forzado para el test"
        audit = AuditLogger("team-x")
        harness = AgentHarness(budget, breaker, audit)

        result = harness.run(lambda task: {}, "tarea", correlation_id="c2")
        assert result.circuit_breaker_triggered == "forzado para el test"
        assert budget.spent == 0.0
