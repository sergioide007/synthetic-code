"""
sintetico_api.main: entry point of the "Synthetic Code" observability API
and dashboard.

Run it locally:

    pip install -e ".[api]"
    uvicorn sintetico_api.main:app --reload

    Dashboard:  http://localhost:8000/
    Swagger UI: http://localhost:8000/docs
    ReDoc:      http://localhost:8000/redoc
    OpenAPI:    http://localhost:8000/openapi.json
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sintetico
from sintetico_api.schemas import HealthOut

from .auth import is_auth_enabled
from .routers import agent, metrics, runs, traces

_STATIC_DIR = Path(__file__).parent / "static"

DESCRIPTION = """
Observability API for the **Synthetic Code** patterns: it triggers
instrumented runs of each of the book's four pillars (efficient
orchestration, autonomous quality in CI/CD, self-cleaning code, and
resilient architecture), invokes a live ReAct agent, and queries their
traces and metrics. It is the same kind of API that would feed a
Datadog/CloudWatch-style panel for a multi-agent system in production.

### Key concepts

- **Run**: one execution of a pillar or of the ReAct agent. It groups one
  or more traces and produces a business summary (`summary`).
- **Trace / correlation_id**: the ordered sequence of events
  (`agent_start`, `reasoning_step`, `tool_result`, `decision`,
  `security_event`, `agent_finish`, ...) of one agent execution. It is the
  "waterfall" shown in the dashboard.
- **Live tail**: `/api/v1/traces/stream` emits every new event in real
  time through Server-Sent Events.

With no API key configured on the server, every endpoint still works with
simulated providers (`MockProvider` and a deterministic ReAct simulator):
the demo is fully functional at no cost. With `ANTHROPIC_API_KEY` or
`OPENAI_API_KEY` set, `/api/v1/agent/invoke` uses a real model.

### Authentication

By default this API **requires no authentication** (local demo mode). If
the server has the `SINTETICO_API_KEY` environment variable set, every
endpoint under `/api/v1/` (except `/api/v1/health`) requires the
`X-API-Key` header with that value. Use the **Authorize** button on this
Swagger page to try it.
"""

app = FastAPI(
    title="Synthetic Code: Agent Observability API",
    description=DESCRIPTION,
    version=sintetico.__version__,
    contact={"name": "Synthetic Code"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Book pillars", "description": "Run and query the book's four architecture pillars."},
        {"name": "ReAct agent", "description": "Invoke a live agent with full traceability."},
        {"name": "Traces", "description": "Historical queries and live tail (SSE) of trace events."},
        {"name": "Metrics", "description": "Aggregated KPIs for observability dashboards."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("SINTETICO_API_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(traces.router)
app.include_router(agent.router)
app.include_router(metrics.router)


@app.get("/api/v1/health", response_model=HealthOut, tags=["Health"], summary="Service status")
def health() -> HealthOut:
    providers = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        providers.append("openai")
    return HealthOut(
        status="ok",
        version="2.0.0",
        sintetico_version=sintetico.__version__,
        auth_enabled=is_auth_enabled(),
        providers_available=providers,
    )


if _STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR)), name="assets")

    @app.get("/", include_in_schema=False)
    def dashboard_index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))
