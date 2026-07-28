"""Tests de los endpoints HTTP de sintetico_api usando FastAPI TestClient.

Requieren `fastapi` y `httpx` instalados (`pip install -e ".[api,dev]"`).
La lógica de negocio que estos endpoints envuelven (store.py, runner.py)
ya está cubierta sin dependencias web en test_api_store_and_runner.py;
aquí sólo se valida el cableado HTTP: rutas, status codes, y que los
`response_model` de Pydantic serializan lo que el dominio produce.
"""

import os
import tempfile

import pytest

pytest.importorskip("fastapi", reason="fastapi no instalado en este entorno")
pytest.importorskip("httpx", reason="httpx no instalado (requerido por TestClient)")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    db_path = tempfile.mktemp(suffix=".db")
    monkeypatch.setenv("SINTETICO_API_DB_PATH", db_path)

    from sintetico_api import deps

    deps.reset_store_for_testing()

    from sintetico_api.main import app

    with TestClient(app) as c:
        yield c

    deps.reset_store_for_testing()
    if os.path.exists(db_path):
        os.remove(db_path)


class TestHealth:
    def test_health_reports_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestRuns:
    @pytest.mark.parametrize("pillar", ["orchestration", "quality_gate", "self_cleaning", "resilient_swarm"])
    def test_run_each_pillar_returns_summary(self, client, pillar):
        resp = client.post(f"/api/v1/runs/{pillar}", json={"team_id": "qa"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"].startswith(pillar)
        assert body["correlation_id"]
        assert isinstance(body["summary"], dict) and body["summary"]

    def test_unknown_pillar_returns_422_or_404(self, client):
        resp = client.post("/api/v1/runs/not-a-real-pillar", json={})
        assert resp.status_code in (404, 422)

    def test_list_and_get_run(self, client):
        created = client.post("/api/v1/runs/self_cleaning", json={}).json()
        listed = client.get("/api/v1/runs").json()
        assert any(r["run_id"] == created["run_id"] for r in listed)

        detail = client.get(f"/api/v1/runs/{created['run_id']}")
        assert detail.status_code == 200
        assert detail.json()["status"] == "completed"

    def test_get_unknown_run_returns_404(self, client):
        resp = client.get("/api/v1/runs/does-not-exist")
        assert resp.status_code == 404


class TestTraces:
    def test_get_full_trace_for_a_run(self, client):
        created = client.post("/api/v1/runs/orchestration", json={}).json()
        resp = client.get(f"/api/v1/traces/{created['correlation_id']}")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) > 0
        assert events == sorted(events, key=lambda e: e["id"])  # orden cronológico

    def test_unknown_correlation_id_returns_404(self, client):
        resp = client.get("/api/v1/traces/does-not-exist")
        assert resp.status_code == 404

    def test_list_events_supports_pillar_filter(self, client):
        client.post("/api/v1/runs/quality_gate", json={})
        resp = client.get("/api/v1/traces", params={"pillar": "quality_gate", "limit": 50})
        assert resp.status_code == 200
        events = resp.json()
        assert all(e["pillar"] == "quality_gate" for e in events)


class TestAgent:
    def test_invoke_agent_in_mock_mode(self, client):
        resp = client.post(
            "/api/v1/agent/invoke",
            json={
                "query": "¿Qué archivos hay disponibles?",
                "team_id": "qa",
                "provider": "mock",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["summary"]["provider"] == "mock"
        assert body["summary"]["status"] == "success"

    def test_invoke_agent_rejects_empty_query(self, client):
        resp = client.post("/api/v1/agent/invoke", json={"query": "", "team_id": "qa"})
        assert resp.status_code == 422  # min_length=1 en el schema


class TestMetrics:
    def test_summary_reflects_executed_runs(self, client):
        client.post("/api/v1/runs/orchestration", json={})
        resp = client.get("/api/v1/metrics/summary", params={"window_minutes": 1440})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_events"] > 0
        assert body["total_cost_usd"] > 0
        assert body["cost_by_model"]
