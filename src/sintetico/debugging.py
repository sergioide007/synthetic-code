"""
sintetico.debugging — Agente de debugging en producción con guardas de
timeout (Capítulo 13).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["DebugPhase", "DebugProgress", "ProductionDebugAgent", "DebugTimeoutGuard"]


class DebugPhase(Enum):
    DETECTION = "detection"
    ISOLATION = "isolation"
    DIAGNOSIS = "diagnosis"
    PATCH = "patch"
    VERIFICATION = "verification"


@dataclass
class DebugProgress:
    phase: DebugPhase
    affected_users: int
    affected_services: List[str]
    elapsed_minutes: int
    hypothesis: Optional[str] = None
    root_cause: Optional[str] = None
    patch_generated: bool = False


class ProductionDebugAgent:
    """Sesión de debugging asistida por LLM con fases explícitas. Sólo
    escala más allá de DETECTION si el impacto supera un umbral mínimo,
    para no gastar presupuesto en incidentes triviales."""

    MIN_AFFECTED_USERS_TO_ESCALATE = 10

    def __init__(self, llm_call: Optional[Callable[[str], str]] = None):
        self.llm_call = llm_call or (lambda prompt: '{"root_cause": "Error de conexión", "confidence": 0.8}')
        self.progress = DebugProgress(DebugPhase.DETECTION, 0, [], 0)

    def start_debug_session(self, alert: dict) -> DebugProgress:
        self.progress.affected_users = alert.get("affected_users", 0)
        self.progress.affected_services = alert.get("services", [])

        if self.progress.affected_users < self.MIN_AFFECTED_USERS_TO_ESCALATE:
            return self.progress

        self.progress.phase = DebugPhase.ISOLATION
        self.progress.phase = DebugPhase.DIAGNOSIS
        response = self.llm_call("Analiza logs y determina la causa raíz probable.")
        try:
            data = json.loads(response)
            self.progress.root_cause = data.get("root_cause", "Desconocida")
        except json.JSONDecodeError:
            logger.warning("Respuesta del LLM no es JSON válido; se usa causa raíz por defecto")
            self.progress.root_cause = "No determinada automáticamente; requiere análisis manual"

        self.progress.phase = DebugPhase.PATCH
        self.progress.patch_generated = True
        self.progress.phase = DebugPhase.VERIFICATION
        return self.progress


class DebugTimeoutGuard:
    """Corta una sesión de debugging que no progresa, evitando que un
    agente quede iterando indefinidamente sobre un incidente sin resolver."""

    ABSOLUTE_TIMEOUT_SECONDS = 900

    def __init__(self, max_idle_minutes: int = 10):
        self.max_idle_minutes = max_idle_minutes
        self.started_at: Optional[datetime] = None
        self.last_progress_at: Optional[datetime] = None
        self.progress_markers: List[str] = []

    def start(self) -> None:
        now = datetime.now(timezone.utc)
        self.started_at = now
        self.last_progress_at = now
        self.progress_markers = []

    def record_progress(self, marker: str) -> None:
        self.progress_markers.append(marker)
        self.last_progress_at = datetime.now(timezone.utc)

    def should_timeout(self) -> Tuple[bool, str]:
        if not self.started_at:
            return False, "No iniciado"
        now = datetime.now(timezone.utc)
        elapsed = (now - self.started_at).total_seconds()
        if elapsed > self.ABSOLUTE_TIMEOUT_SECONDS:
            return True, "Timeout absoluto"
        if self.last_progress_at:
            idle = (now - self.last_progress_at).total_seconds()
            if idle > self.max_idle_minutes * 60:
                return True, f"Inactividad > {self.max_idle_minutes} min"
        return False, "OK"
