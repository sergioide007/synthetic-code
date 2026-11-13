"""
sintetico_api.schemas: Pydantic request/response models.

This is the only layer of the service that knows about Pydantic; it exists to
produce rich, typed OpenAPI/Swagger documentation (`/docs`, `/redoc`,
`/openapi.json`). The real business logic lives in `runner.py`/`store.py` and
does not depend on these models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Pillar(str, Enum):
    orchestration = "orchestration"
    quality_gate = "quality_gate"
    self_cleaning = "self_cleaning"
    resilient_swarm = "resilient_swarm"


class ProviderType(str, Enum):
    auto = "auto"
    anthropic = "anthropic"
    openai = "openai"
    mock = "mock"


class RunRequest(BaseModel):
    team_id: str = Field(default="demo", description="Team/tenant the run and its cost are attributed to.")


class RunSummary(BaseModel):
    run_id: str = Field(description="Run identifier (groups every trace it produced).")
    correlation_id: str = Field(description="Main trace identifier; use it in /traces/{correlation_id}.")
    summary: Dict[str, Any] = Field(description="Business metrics specific to the pillar that ran.")


class ReactAgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Natural-language query for the ReAct agent.")
    team_id: str = Field(default="demo")
    provider: ProviderType = Field(
        default=ProviderType.auto,
        description=(
            "Provider to use. 'auto' uses a real provider when the server environment has an API "
            "key (ANTHROPIC_API_KEY/OPENAI_API_KEY); otherwise it falls back to 'mock' automatically."
        ),
    )


class TraceEventOut(BaseModel):
    id: int
    correlation_id: str
    run_id: Optional[str] = None
    pillar: Optional[str] = None
    agent_id: Optional[str] = None
    team_id: Optional[str] = None
    event_type: str
    level: str
    timestamp: str
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    model_used: Optional[str] = None
    retry_count: Optional[int] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class RunOut(BaseModel):
    run_id: str
    pillar: str
    team_id: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)


class MetricsSummaryOut(BaseModel):
    window_minutes: int
    total_events: int
    total_cost_usd: float
    avg_latency_ms: float
    p95_latency_ms: float
    events_by_level: Dict[str, int]
    events_by_type: Dict[str, int]
    events_by_pillar: Dict[str, int]
    cost_by_model: Dict[str, float]
    runs_by_status: Dict[str, int]


class HealthOut(BaseModel):
    status: str = "ok"
    version: str
    sintetico_version: str
    auth_enabled: bool = Field(
        description="When true, every /api/v1/* endpoint (except /health) requires the X-API-Key header."
    )
    providers_available: List[str] = Field(
        description="Real providers the server can use, based on the environment variables it found."
    )
