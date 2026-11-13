"""Aggregated metrics (KPIs) for the dashboard cards: total cost, p95
latency, event distribution by level/type/pillar, and cost per model; the
same kind of panel as a Datadog "Summary"."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from sintetico_api.schemas import MetricsSummaryOut
from sintetico_api.store import TraceStore

from ..auth import require_api_key
from ..deps import get_store

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"], dependencies=[Depends(require_api_key)])


@router.get(
    "/summary",
    response_model=MetricsSummaryOut,
    summary="Aggregated metrics over a time window",
)
def metrics_summary(
    window_minutes: int = Query(default=60, ge=1, le=10_080, description="Window in minutes (max. 7 days)."),
    store: TraceStore = Depends(get_store),
) -> MetricsSummaryOut:
    return MetricsSummaryOut(**store.metrics_summary(window_minutes=window_minutes))
