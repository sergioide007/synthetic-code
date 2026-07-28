"""
sintetico_api.schemas — Modelos Pydantic de request/response.

Esta es la única capa del servicio que conoce Pydantic; existe
específicamente para producir una documentación OpenAPI/Swagger rica y
tipada (`/docs`, `/redoc`, `/openapi.json`). La lógica de negocio real
vive en `runner.py`/`store.py` y no depende de estos modelos.
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
    team_id: str = Field(default="demo", description="Equipo/tenant al que se atribuye la ejecución y su coste.")


class RunSummary(BaseModel):
    run_id: str = Field(description="Identificador de la ejecución (agrupa todas las trazas que produjo).")
    correlation_id: str = Field(description="Identificador de la traza principal; úsalo en /traces/{correlation_id}.")
    summary: Dict[str, Any] = Field(description="Métricas de negocio específicas del pilar ejecutado.")


class ReactAgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Consulta en lenguaje natural para el agente ReAct.")
    team_id: str = Field(default="demo")
    provider: ProviderType = Field(
        default=ProviderType.auto,
        description=(
            "Proveedor a usar. 'auto' usa un proveedor real si hay una API key en el entorno del "
            "servidor (ANTHROPIC_API_KEY/OPENAI_API_KEY); si no, cae a 'mock' automáticamente."
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
        description="Si es true, todos los endpoints /api/v1/* (salvo /health) exigen la cabecera X-API-Key."
    )
    providers_available: List[str] = Field(
        description="Proveedores reales que el servidor puede usar según las variables de entorno detectadas."
    )
