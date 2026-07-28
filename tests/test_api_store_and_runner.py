"""Tests de sintetico_api.store y sintetico_api.runner.

Deliberadamente NO requieren FastAPI/Pydantic instalados: store.py y
runner.py son lógica de dominio pura. Los tests de los endpoints HTTP
(que sí requieren FastAPI) viven en test_api_http.py.
"""

import os
import tempfile

import pytest

from sintetico_api.runner import (
    run_orchestration_case,
    run_quality_gate_case,
    run_react_agent,
    run_resilient_swarm_case,
    run_self_cleaning_case,
)
from sintetico_api.store import TraceStore


@pytest.fixture
def store():
    path = tempfile.mktemp(suffix=".db")
    s = TraceStore(path)
    yield s
    s.close()
    if os.path.exists(path):
        os.remove(path)


class TestTraceStore:
    def test_insert_and_get_trace_preserves_order(self, store):
        store.insert_event(
            {"correlation_id": "c1", "event_type": "a", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        store.insert_event(
            {"correlation_id": "c1", "event_type": "b", "level": "INFO", "timestamp": "2026-01-01T00:00:01+00:00"}
        )
        trace = store.get_trace("c1")
        assert [e["event_type"] for e in trace] == ["a", "b"]

    def test_run_lifecycle(self, store):
        run_id = store.create_run(pillar="orchestration", team_id="t1")
        assert store.get_run(run_id)["status"] == "running"
        store.finish_run(run_id, status="completed", summary={"x": 1})
        run = store.get_run(run_id)
        assert run["status"] == "completed"
        assert run["summary"] == {"x": 1}

    def test_live_tail_receives_published_events(self, store):
        sub = store.subscribe()
        store.insert_event(
            {"correlation_id": "c1", "event_type": "a", "level": "INFO", "timestamp": "2026-01-01T00:00:00+00:00"}
        )
        assert sub.get_nowait()["event_type"] == "a"
        store.unsubscribe(sub)

    def test_metrics_summary_aggregates_cost_and_latency(self, store):
        store.insert_event(
            {
                "correlation_id": "c1",
                "event_type": "tool_result",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "cost_usd": 0.01,
                "model_used": "haiku",
                "latency_ms": 100,
            }
        )
        store.insert_event(
            {
                "correlation_id": "c1",
                "event_type": "tool_result",
                "level": "INFO",
                "timestamp": "2026-01-01T00:00:01+00:00",
                "cost_usd": 0.02,
                "model_used": "haiku",
                "latency_ms": 200,
            }
        )
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["total_cost_usd"] == 0.03
        assert metrics["cost_by_model"] == {"haiku": 0.03}
        assert metrics["avg_latency_ms"] == 150.0


class TestRunner:
    def test_orchestration_case_produces_trace_and_summary(self, store):
        result = run_orchestration_case(store)
        assert result["summary"]["queries"] > 0
        assert result["summary"]["total_cost_usd"] > 0
        trace = store.get_trace(result["correlation_id"])
        assert len(trace) > 0
        assert store.get_run(result["run_id"])["status"] == "completed"

    def test_orchestration_case_populates_cost_by_model(self, store):
        """Regresión: `model_used`/`cost_usd` no llegaban nunca al log
        estructurado (campo `model_used` inalcanzable en la práctica), así
        que el desglose de coste por modelo del dashboard estaba siempre
        vacío pese a que la información sí se calculaba internamente."""
        run_orchestration_case(store)
        metrics = store.metrics_summary(window_minutes=10_000_000)
        assert metrics["cost_by_model"], "cost_by_model no debería estar vacío"

    def test_quality_gate_case_records_override_for_false_positive(self, store):
        result = run_quality_gate_case(store)
        assert result["summary"]["overrides_issued"] == 1
        assert result["summary"]["blocked_critical"] == 1

    def test_self_cleaning_case_produces_valid_python(self, store):
        result = run_self_cleaning_case(store)
        assert result["summary"]["status"] == "cleaned"
        compile(result["code_after"], "<test>", "exec")

    def test_resilient_swarm_case_reports_circuit_breaker_state(self, store):
        result = run_resilient_swarm_case(store)
        assert "circuit_breaker_open" in result["summary"]

    def test_react_agent_mock_mode_completes_and_traces(self, store):
        result = run_react_agent(store, "Hola, ¿qué puedes hacer?", provider_type="mock")
        assert result["summary"]["provider"] == "mock"
        assert result["summary"]["status"] == "success"
        trace = store.get_trace(result["correlation_id"])
        assert any(e["event_type"] == "agent_start" for e in trace)
        assert any(e["event_type"] == "agent_finish" for e in trace)
