"""
sintetico_api.base — Interfaz común para backends de almacenamiento de
trazas. `TraceStore` (SQLite) y `PostgresTraceStore` implementan este
mismo contrato, así que la capa de API (`deps.get_store`) puede elegir
uno u otro según el entorno sin que ningún router/servicio tenga que
saberlo.

Por qué dos backends:
- **SQLite** (`sintetico_api.store.TraceStore`): cero configuración, un
  fichero local. Perfecto para la demo del libro y para desarrollo, pero
  no compartido entre procesos/instancias.
- **Postgres** (`sintetico_api.postgres_store.PostgresTraceStore`): para
  un despliegue real con más de una instancia del servicio detrás de un
  balanceador. Usa `LISTEN`/`NOTIFY` nativo de Postgres para que el
  live-tail (Server-Sent Events) funcione correctamente entre instancias
  — un evento insertado por la instancia A se ve en tiempo real en un
  cliente conectado a la instancia B.

`sintetico.budget.TokenBudget` ya resuelve el mismo problema (local vs.
distribuido) con `BudgetBackend`/`RedisBudgetBackend`; esta es la misma
idea aplicada al almacén de trazas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

__all__ = ["BaseTraceStore"]


class BaseTraceStore(ABC):
    """Contrato que debe cumplir cualquier backend de trazas/ejecuciones."""

    # ─── Eventos ────────────────────────────────────────────────────
    @abstractmethod
    def insert_event(self, entry: Dict[str, Any], run_id: Optional[str] = None, pillar: Optional[str] = None) -> int:
        """Inserta un evento de traza y lo publica a los suscriptores de
        live-tail. Devuelve el id asignado."""

    @abstractmethod
    def get_trace(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Todos los eventos de una traza, en orden cronológico."""

    @abstractmethod
    def list_events(
        self,
        run_id: Optional[str] = None,
        pillar: Optional[str] = None,
        level: Optional[str] = None,
        limit: int = 200,
        after_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Eventos recientes, opcionalmente filtrados."""

    # ─── Ejecuciones (runs) ─────────────────────────────────────────
    @abstractmethod
    def create_run(self, pillar: str, team_id: str = "demo") -> str: ...

    @abstractmethod
    def finish_run(self, run_id: str, status: str, summary: Optional[Dict[str, Any]] = None) -> None: ...

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def list_runs(self, pillar: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]: ...

    # ─── Métricas ───────────────────────────────────────────────────
    @abstractmethod
    def metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]: ...

    # ─── Live tail (pub-sub) ────────────────────────────────────────
    @abstractmethod
    def subscribe(self) -> "Any":
        """Registra un suscriptor para live-tail; devuelve un objeto tipo
        `queue.Queue` del que se puede hacer `.get()`."""

    @abstractmethod
    def unsubscribe(self, subscriber: "Any") -> None: ...

    @abstractmethod
    def close(self) -> None: ...
