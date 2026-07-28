"""
sintetico.budget — Control de presupuesto de tokens (Capítulo 3).

Provee una interfaz de backend intercambiable para llevar la cuenta del
gasto acumulado por equipo, con una implementación en memoria (para tests
y demos de un solo proceso) y una implementación distribuida sobre Redis
que garantiza atomicidad "check-and-increment" vía un script Lua.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)

__all__ = [
    "BudgetBackend",
    "LocalMemoryBudgetBackend",
    "RedisBudgetBackend",
    "RedisUnavailableError",
    "TokenBudget",
]


class RedisUnavailableError(RuntimeError):
    """Se lanza cuando se intenta operar con un backend Redis sin conexión."""


class BudgetBackend(ABC):
    """Interfaz para almacenamiento atómico y distribuido del presupuesto."""

    @abstractmethod
    def get_spent(self, key: str) -> float:
        """Devuelve el gasto acumulado para `key`."""

    @abstractmethod
    def add_cost(self, key: str, amount: float, limit: float) -> bool:
        """Intenta añadir `amount` a `key` sin superar `limit`.

        Debe ser atómico: si dos llamadas concurrentes verifican el límite
        a la vez, sólo una de ellas puede tener éxito si juntas lo superan.
        """


class LocalMemoryBudgetBackend(BudgetBackend):
    """Backend en memoria local con lock para atomicidad en un solo proceso.

    Adecuado para tests, demos y despliegues de un único nodo. No es
    apto para múltiples procesos/instancias porque el estado no se
    comparte entre ellos.
    """

    def __init__(self) -> None:
        self._store: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get_spent(self, key: str) -> float:
        with self._lock:
            return self._store.get(key, 0.0)

    def add_cost(self, key: str, amount: float, limit: float) -> bool:
        with self._lock:
            current = self._store.get(key, 0.0)
            if current + amount > limit:
                return False
            self._store[key] = current + amount
            return True


class RedisBudgetBackend(BudgetBackend):
    """Backend distribuido con Redis para control de presupuesto multi-nodo.

    Usa un script Lua para la operación atómica check-and-increment, de
    forma que múltiples procesos/instancias compartan un único contador
    consistente sin condiciones de carrera.

    Nota: la clave ya llega formateada desde `TokenBudget` (p. ej.
    ``"budget:team_id"``), por lo que el prefijo por defecto es vacío
    para evitar duplicación.
    """

    _INCR_SCRIPT = """
    local current = tonumber(redis.call('GET', KEYS[1]) or '0')
    local amount = tonumber(ARGV[1])
    local limit = tonumber(ARGV[2])
    if current + amount > limit then
        return 0
    end
    return redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
    """

    def __init__(
        self, host: str = "localhost", port: int = 6379, db: int = 0, prefix: str = "", socket_timeout: float = 2.0
    ) -> None:
        try:
            import redis  # import local para no forzar la dependencia
        except ImportError as exc:
            raise ImportError("Instala redis-py: pip install redis") from exc

        self.prefix = prefix
        self.redis = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=socket_timeout,
        )
        try:
            self.redis.ping()
            self._available = True
            self._connection_error: str | None = None
            self._incr_script = self.redis.register_script(self._INCR_SCRIPT)
        except redis.exceptions.RedisError as exc:
            self._available = False
            self._connection_error = str(exc)
            self._incr_script = None
            logger.warning("No se pudo conectar a Redis (%s:%s): %s", host, port, exc)

    def _full_key(self, key: str) -> str:
        return f"{self.prefix}{key}" if self.prefix else key

    def get_spent(self, key: str) -> float:
        if not self._available:
            raise RedisUnavailableError(self._connection_error or "Redis no disponible")
        value = self.redis.get(self._full_key(key))
        return float(value) if value else 0.0

    def add_cost(self, key: str, amount: float, limit: float) -> bool:
        if not self._available:
            raise RedisUnavailableError(self._connection_error or "Redis no disponible")

        full_key = self._full_key(key)
        try:
            result = self._incr_script(keys=[full_key], args=[amount, limit])
            return result != 0
        except Exception:
            logger.exception("Fallo el script Lua, aplicando fallback con WATCH/MULTI")
            return self._add_cost_with_optimistic_lock(full_key, amount, limit)

    def _add_cost_with_optimistic_lock(self, full_key: str, amount: float, limit: float) -> bool:
        """Fallback con bloqueo optimista (WATCH/MULTI) si el script Lua falla."""
        import redis as redis_module

        with self.redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(full_key)
                    current = float(pipe.get(full_key) or 0.0)
                    if current + amount > limit:
                        pipe.unwatch()
                        return False
                    pipe.multi()
                    pipe.incrbyfloat(full_key, amount)
                    pipe.execute()
                    return True
                except redis_module.WatchError:
                    continue


@dataclass
class TokenBudget:
    """Presupuesto mensual de tokens con backend intercambiable (local/Redis).

    Las alertas progresivas (`alerts_triggered`) se almacenan en memoria
    local por instancia: en un despliegue multi-nodo cada nodo emitirá su
    propia alerta la primera vez que la observe, lo cual es aceptable para
    logging pero no debe usarse como fuente de verdad para decisiones de
    negocio (para eso está el propio backend, que sí es consistente).
    """

    monthly_budget: float
    backend: BudgetBackend = field(default_factory=LocalMemoryBudgetBackend)
    team_id: str = "default"
    alerts_triggered: Dict[float, bool] = field(default_factory=dict)
    on_alert: "callable | None" = None

    def __post_init__(self) -> None:
        if self.monthly_budget <= 0:
            raise ValueError("monthly_budget debe ser positivo")

    @property
    def _key(self) -> str:
        return f"budget:{self.team_id}"

    @property
    def spent(self) -> float:
        """Gasto acumulado para este equipo."""
        return self.backend.get_spent(self._key)

    @property
    def remaining(self) -> float:
        return max(self.monthly_budget - self.spent, 0.0)

    def record_cost(self, cost: float) -> bool:
        """Registra `cost` si no excede el presupuesto. Devuelve si tuvo éxito."""
        if cost < 0:
            raise ValueError("cost no puede ser negativo")
        if not self.backend.add_cost(self._key, cost, self.monthly_budget):
            return False
        self._check_alerts()
        return True

    def _check_alerts(self) -> None:
        spent = self.backend.get_spent(self._key)
        pct = spent / self.monthly_budget
        for threshold in (0.5, 0.75, 0.9, 1.0):
            if pct >= threshold and not self.alerts_triggered.get(threshold):
                self.alerts_triggered[threshold] = True
                message = f"Budget {int(threshold * 100)}% alcanzado para {self.team_id}"
                logger.warning(message)
                if self.on_alert:
                    self.on_alert(threshold, self.team_id, spent)
