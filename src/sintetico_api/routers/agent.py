"""Endpoint that invokes the `trazabilidad` ReAct agent live, with a real
provider (Anthropic/OpenAI) when the server has an API key, or a simulated
one otherwise. It backs the dashboard's "ask the agent something" demo."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from sintetico_api import runner
from sintetico_api.schemas import ReactAgentRequest, RunSummary
from sintetico_api.store import TraceStore

from ..auth import require_api_key
from ..deps import get_store

router = APIRouter(prefix="/api/v1/agent", tags=["ReAct agent"], dependencies=[Depends(require_api_key)])


@router.post(
    "/invoke",
    response_model=RunSummary,
    summary="Invoke the ReAct agent with full traceability",
    description=(
        "Runs the `trazabilidad` ReAct agent on the given query. If the "
        "server has `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in its "
        "environment, the answer comes from a real model; otherwise it comes "
        "from a deterministic simulator that follows the same ReAct protocol, "
        "so the demo still works at no cost. The response's "
        "`summary.provider` field says which one was used."
    ),
)
def invoke_agent(body: ReactAgentRequest, store: TraceStore = Depends(get_store)) -> RunSummary:
    result = runner.run_react_agent(store, query=body.query, team_id=body.team_id, provider_type=body.provider.value)
    return RunSummary(**result)
