"""Endpoints para disparar la ejecución de cada uno de los 4 pilares del
libro y consultar el histórico de ejecuciones (`runs`). Cada ejecución
queda registrada como un `run` y produce una o más trazas consultables en
`/api/v1/traces`."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from sintetico_api import runner
from sintetico_api.schemas import Pillar, RunOut, RunRequest, RunSummary
from sintetico_api.store import TraceStore

from ..auth import require_api_key
from ..deps import get_store

router = APIRouter(prefix="/api/v1/runs", tags=["Pilares del libro"], dependencies=[Depends(require_api_key)])

_PILLAR_FUNCS = {
    Pillar.orchestration: runner.run_orchestration_case,
    Pillar.quality_gate: runner.run_quality_gate_case,
    Pillar.self_cleaning: runner.run_self_cleaning_case,
    Pillar.resilient_swarm: runner.run_resilient_swarm_case,
}


@router.post(
    "/{pillar}",
    response_model=RunSummary,
    summary="Ejecuta uno de los 4 pilares del libro",
    description=(
        "Dispara una ejecución instrumentada del pilar solicitado "
        "(orquestación, calidad autónoma en CI/CD, código autolimpiable o "
        "arquitectura resiliente) y devuelve un resumen con sus métricas "
        "de negocio. Cada paso queda registrado como una traza consultable "
        "en `/api/v1/traces/{correlation_id}`."
    ),
)
def run_pillar(pillar: Pillar, body: RunRequest = RunRequest(), store: TraceStore = Depends(get_store)) -> RunSummary:
    fn = _PILLAR_FUNCS.get(pillar)
    if fn is None:
        raise HTTPException(status_code=404, detail=f"Pilar desconocido: {pillar}")
    result = fn(store, team_id=body.team_id)
    return RunSummary(**result)


@router.get("", response_model=List[RunOut], summary="Lista ejecuciones recientes")
def list_runs(
    pillar: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    store: TraceStore = Depends(get_store),
) -> List[RunOut]:
    return [RunOut(**r) for r in store.list_runs(pillar=pillar, limit=limit)]


@router.get("/{run_id}", response_model=RunOut, summary="Detalle de una ejecución")
def get_run(run_id: str, store: TraceStore = Depends(get_store)) -> RunOut:
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run no encontrado: {run_id}")
    return RunOut(**run)
