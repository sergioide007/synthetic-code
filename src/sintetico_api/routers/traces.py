"""Trace query endpoints: paginated history, one complete trace (the
"waterfall" of a correlation_id), and a live tail over Server-Sent Events, in
the style of Datadog Live Tail or CloudWatch Logs Insights."""

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

router = APIRouter(prefix="/api/v1/traces", tags=["Traces"], dependencies=[Depends(require_api_key)])


@router.get(
    "",
    response_model=List[TraceEventOut],
    summary="List recent trace events (paginated by id)",
)
def list_events(
    run_id: Optional[str] = Query(default=None),
    pillar: Optional[str] = Query(
        default=None,
        description="Filter by pillar: orchestration, quality_gate, self_cleaning, resilient_swarm, react_agent",
    ),
    level: Optional[str] = Query(default=None, description="INFO | WARNING | ERROR | CRITICAL"),
    after_id: Optional[int] = Query(
        default=None, description="Only events with id > after_id (for incremental polling)."
    ),
    limit: int = Query(default=200, ge=1, le=2000),
    store: TraceStore = Depends(get_store),
) -> List[TraceEventOut]:
    events = store.list_events(run_id=run_id, pillar=pillar, level=level, limit=limit, after_id=after_id)
    return [TraceEventOut(**e) for e in events]


@router.get(
    "/stream",
    summary="Live tail of traces over Server-Sent Events",
    description=(
        "Opens a `text/event-stream` that emits every new trace event in "
        "real time, like Datadog's 'Live Tail'. Consume it from the browser "
        "with `new EventSource('/api/v1/traces/stream')`."
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
                    yield ": keep-alive\n\n"  # SSE comment: keeps the connection alive
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
    summary="Return the complete trace (waterfall) of a correlation_id",
)
def get_trace(correlation_id: str, store: TraceStore = Depends(get_store)) -> List[TraceEventOut]:
    events = store.get_trace(correlation_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"No events for correlation_id={correlation_id}")
    return [TraceEventOut(**e) for e in events]
