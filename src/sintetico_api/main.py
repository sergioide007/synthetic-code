"""
sintetico_api.main — Punto de entrada de la API + dashboard de
observabilidad de "Código Sintético".

Arranque local:

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
API de observabilidad para los patrones de **Código Sintético**: dispara
ejecuciones instrumentadas de cada uno de los 4 pilares del libro
(orquestación eficiente, calidad autónoma en CI/CD, código autolimpiable
y arquitectura resiliente), invoca un agente ReAct en vivo, y consulta
sus trazas y métricas — el mismo tipo de API que alimentaría un panel
estilo Datadog/CloudWatch para un sistema multiagente en producción.

### Conceptos clave

- **Run**: una ejecución de un pilar o del agente ReAct. Agrupa una o
  varias trazas y produce un resumen de negocio (`summary`).
- **Trace / correlation_id**: la secuencia ordenada de eventos
  (`agent_start`, `reasoning_step`, `tool_result`, `decision`,
  `security_event`, `agent_finish`, ...) de una ejecución del agente. Es
  la "waterfall" que se visualiza en el dashboard.
- **Live tail**: `/api/v1/traces/stream` emite cada evento nuevo en
  tiempo real vía Server-Sent Events.

Sin ninguna API key configurada en el servidor, todos los endpoints
funcionan igualmente con proveedores simulados (`MockProvider` / un
simulador ReAct determinista): la demo es 100% funcional sin coste. Con
`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` configuradas, `/api/v1/agent/invoke`
usa un modelo real.

### Autenticación

Por defecto esta API **no exige autenticación** (modo demo local). Si el
servidor tiene configurada la variable de entorno `SINTETICO_API_KEY`,
todos los endpoints bajo `/api/v1/` (excepto `/api/v1/health`) exigen la
cabecera `X-API-Key` con ese valor. Usa el botón **Authorize** de esta
misma página de Swagger para probarlo.
"""

app = FastAPI(
    title="Código Sintético — Agent Observability API",
    description=DESCRIPTION,
    version=sintetico.__version__,
    contact={"name": "Código Sintético"},
    license_info={"name": "MIT"},
    openapi_tags=[
        {"name": "Pilares del libro", "description": "Ejecuta y consulta los 4 pilares de arquitectura del libro."},
        {"name": "Agente ReAct", "description": "Invoca en vivo un agente con trazabilidad completa."},
        {"name": "Trazas", "description": "Consulta histórica y live-tail (SSE) de eventos de traza."},
        {"name": "Métricas", "description": "KPIs agregados para dashboards de observabilidad."},
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


@app.get("/api/v1/health", response_model=HealthOut, tags=["Salud"], summary="Estado del servicio")
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
