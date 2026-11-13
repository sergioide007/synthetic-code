"""Endpoints that trigger a run of each of the book's four pillars and query
the run history. Each execution is recorded as a `run` and produces one or
more traces you can query at `/api/v1/traces`."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from sintetico_api import runner
from sintetico_api.schemas import Pillar, RunOut, RunRequest, RunSummary
from sintetico_api.store import TraceStore

from ..auth import require_api_key
from ..deps import get_store

router = APIRouter(prefix="/api/v1/runs", tags=["Book pillars"], dependencies=[Depends(require_api_key)])

_PILLAR_FUNCS = {
    Pillar.orchestration: runner.run_orchestration_case,
    Pillar.quality_gate: runner.run_quality_gate_case,
    Pillar.self_cleaning: runner.run_self_cleaning_case,
    Pillar.resilient_swarm: runner.run_resilient_swarm_case,
}


@router.post(
    "/{pillar}",
    response_model=RunSummary,
    summary="Run one of the book's four pillars",
    description=(
        "Triggers an instrumented run of the requested pillar "
        "(orchestration, autonomous quality in CI/CD, self-cleaning code, or "
        "resilient architecture) and returns a summary with its business "
        "metrics. Every step is recorded as a trace you can query at "
        "`/api/v1/traces/{correlation_id}`."
    ),
)
def run_pillar(pillar: Pillar, body: RunRequest = RunRequest(), store: TraceStore = Depends(get_store)) -> RunSummary:
    fn = _PILLAR_FUNCS.get(pillar)
    if fn is None:
        raise HTTPException(status_code=404, detail=f"Unknown pillar: {pillar}")
    result = fn(store, team_id=body.team_id)
    return RunSummary(**result)


@router.get("", response_model=List[RunOut], summary="List recent runs")
def list_runs(
    pillar: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    store: TraceStore = Depends(get_store),
) -> List[RunOut]:
    return [RunOut(**r) for r in store.list_runs(pillar=pillar, limit=limit)]


@router.get("/{run_id}", response_model=RunOut, summary="Run details")
def get_run(run_id: str, store: TraceStore = Depends(get_store)) -> RunOut:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return RunOut(**run)
