"""Endpoint para invocar en vivo al agente ReAct de `trazabilidad`, con
proveedor real (Anthropic/OpenAI) si hay API key en el servidor, o
simulado en caso contrario. Es el endpoint pensado para la demo
"pregúntale algo al agente" del dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from sintetico_api import runner
from sintetico_api.schemas import ReactAgentRequest, RunSummary
from sintetico_api.store import TraceStore

from ..auth import require_api_key
from ..deps import get_store

router = APIRouter(prefix="/api/v1/agent", tags=["Agente ReAct"], dependencies=[Depends(require_api_key)])


@router.post(
    "/invoke",
    response_model=RunSummary,
    summary="Invoca al agente ReAct con trazabilidad completa",
    description=(
        "Ejecuta el agente ReAct de `trazabilidad` sobre la consulta dada. "
        "Si el servidor tiene `ANTHROPIC_API_KEY` u `OPENAI_API_KEY` en su "
        "entorno, la respuesta proviene de un modelo real; si no, de un "
        "simulador determinista que respeta el mismo protocolo ReAct, para "
        "que la demo funcione igualmente sin coste. El campo "
        "`summary.provider` de la respuesta indica cuál se usó."
    ),
)
def invoke_agent(body: ReactAgentRequest, store: TraceStore = Depends(get_store)) -> RunSummary:
    result = runner.run_react_agent(store, query=body.query, team_id=body.team_id, provider_type=body.provider.value)
    return RunSummary(**result)
