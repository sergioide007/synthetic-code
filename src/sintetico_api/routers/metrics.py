"""Endpoint de métricas agregadas (KPIs) para las tarjetas del dashboard:
coste total, latencia p95, distribución de eventos por nivel/tipo/pilar y
coste por modelo — el mismo tipo de panel que un "Summary" de Datadog."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from sintetico_api.schemas import MetricsSummaryOut
from sintetico_api.store import TraceStore

from ..auth import require_api_key
from ..deps import get_store

router = APIRouter(prefix="/api/v1/metrics", tags=["Métricas"], dependencies=[Depends(require_api_key)])


@router.get(
    "/summary",
    response_model=MetricsSummaryOut,
    summary="Resumen agregado de métricas en una ventana de tiempo",
)
def metrics_summary(
    window_minutes: int = Query(default=60, ge=1, le=10_080, description="Ventana en minutos (máx. 7 días)."),
    store: TraceStore = Depends(get_store),
) -> MetricsSummaryOut:
    return MetricsSummaryOut(**store.metrics_summary(window_minutes=window_minutes))
