"""Endpoints de consulta de trazas: histórico paginado, una traza completa
(la "waterfall" de un correlation_id) y live-tail vía Server-Sent Events,
al estilo de Datadog Live Tail / CloudWatch Logs Insights."""

from __future__ import annotations

import asyncio
import json
import queue
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from sintetico_api.schemas import TraceEventOut
from sintetico_api.store import TraceStore

from ..auth import require_api_key
from ..deps import get_store

router = APIRouter(prefix="/api/v1/traces", tags=["Trazas"], dependencies=[Depends(require_api_key)])


@router.get(
    "",
    response_model=List[TraceEventOut],
    summary="Lista eventos de traza recientes (paginado por id)",
)
def list_events(
    run_id: Optional[str] = Query(default=None),
    pillar: Optional[str] = Query(
        default=None,
        description="Filtra por pilar: orchestration, quality_gate, self_cleaning, resilient_swarm, react_agent",
    ),
    level: Optional[str] = Query(default=None, description="INFO | WARNING | ERROR | CRITICAL"),
    after_id: Optional[int] = Query(
        default=None, description="Sólo eventos con id > after_id (para polling incremental)."
    ),
    limit: int = Query(default=200, ge=1, le=2000),
    store: TraceStore = Depends(get_store),
) -> List[TraceEventOut]:
    events = store.list_events(run_id=run_id, pillar=pillar, level=level, limit=limit, after_id=after_id)
    return [TraceEventOut(**e) for e in events]


@router.get(
    "/stream",
    summary="Live-tail de trazas vía Server-Sent Events",
    description=(
        "Abre un stream `text/event-stream` que emite cada nuevo evento de "
        "traza en tiempo real, al estilo del 'Live Tail' de Datadog. "
        "Consúmelo desde el navegador con `new EventSource('/api/v1/traces/stream')`."
    ),
)
async def stream_events(store: TraceStore = Depends(get_store)) -> StreamingResponse:
    q = store.subscribe()

    async def event_generator():
        try:
            yield "event: connected\ndata: {}\n\n"
            while True:
                try:
                    event = await asyncio.get_event_loop().run_in_executor(None, q.get, True, 15.0)
                    yield f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"  # comentario SSE, mantiene viva la conexión
        finally:
            store.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/{correlation_id}",
    response_model=List[TraceEventOut],
    summary="Devuelve la traza completa (waterfall) de un correlation_id",
)
def get_trace(correlation_id: str, store: TraceStore = Depends(get_store)) -> List[TraceEventOut]:
    events = store.get_trace(correlation_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"No hay eventos para correlation_id={correlation_id}")
    return [TraceEventOut(**e) for e in events]
