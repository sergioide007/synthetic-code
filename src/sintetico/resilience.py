"""
sintetico.resilience — Circuit breakers, harness de ejecución y logging
de auditoría (Capítulo 5).
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from .budget import TokenBudget

logger = logging.getLogger(__name__)

__all__ = [
    "CircuitBreaker",
    "AgentCircuitBreaker",
    "AuditLogger",
    "AgentResult",
    "AgentHarness",
]


class CircuitBreaker:
    """Circuit breaker clásico (cerrado/abierto/semi-abierto) para llamadas
    a servicios externos inestables."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "closed"
        self.last_failure_time: Optional[datetime] = None

    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open" and self._can_recover():
            self.state = "half-open"
            return True
        return self.state == "half-open"

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = datetime.now(timezone.utc)
        if self.failures >= self.failure_threshold:
            self.state = "open"

    def _can_recover(self) -> bool:
        if not self.last_failure_time:
            return False
        return (datetime.now(timezone.utc) - self.last_failure_time).total_seconds() >= self.recovery_timeout


class AgentCircuitBreaker:
    """Circuit breaker especializado en agentes: detecta ciclos de
    pensamiento idénticos y "flooding" de una misma herramienta."""

    def __init__(self, max_cycles: int = 5, same_tool_threshold: int = 4, window_size: int = 30):
        self.max_cycles = max_cycles
        self.same_tool_threshold = same_tool_threshold
        self.window_size = window_size
        self.history: List[Dict] = []
        self.state = "closed"
        self.failure_reason: Optional[str] = None
        self.tool_invocation_window: List[str] = []

    def record_cycle(self, thought: str, tools: List[str], observation: str) -> None:
        thought_hash = hashlib.sha256(thought.encode()).hexdigest()[:8]
        obs_hash = hashlib.sha256(observation.encode()).hexdigest()[:8]
        self.history.append({"thought_hash": thought_hash, "tools": tools, "obs_hash": obs_hash})
        self.tool_invocation_window.extend(tools)
        if len(self.tool_invocation_window) > self.window_size:
            self.tool_invocation_window = self.tool_invocation_window[-self.window_size :]
        self._check_repetition()
        self._check_tool_flooding()

    def _check_repetition(self) -> None:
        if len(self.history) < self.max_cycles:
            return
        recent = self.history[-self.max_cycles :]
        if len({r["thought_hash"] for r in recent}) == 1:
            self.state = "open"
            self.failure_reason = f"{self.max_cycles} ciclos de pensamiento idénticos"

    def _check_tool_flooding(self) -> None:
        if len(self.tool_invocation_window) < self.window_size:
            return
        counts = Counter(self.tool_invocation_window)
        for tool, count in counts.items():
            if count >= self.same_tool_threshold and count / self.window_size > 0.3:
                self.state = "open"
                self.failure_reason = f"Flooding: {tool} {count}/{self.window_size}"

    def is_open(self) -> bool:
        return self.state == "open"

    def reset(self) -> None:
        self.state = "closed"
        self.failure_reason = None


class AuditLogger:
    """Registro de auditoría por equipo. En producción se emitiría a un
    sumidero append-only (S3, BigQuery, etc.) en vez de acumularse en
    memoria; `logs` se expone para poder inspeccionarlo en tests/demos."""

    def __init__(self, team_id: str):
        self.team_id = team_id
        self.logs: List[Dict] = []

    def log(
        self,
        model: str,
        tokens_input: int,
        tokens_output: int,
        cost_usd: float,
        prompt: str,
        response: str,
        tools_used: Optional[List[str]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "team_id": self.team_id,
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cost_usd": round(cost_usd, 4),
            # Nunca se persiste el prompt/respuesta en claro: sólo su hash,
            # suficiente para deduplicar y correlacionar sin filtrar datos.
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "response_hash": hashlib.sha256(response.encode()).hexdigest()[:16],
            "tools_used": tools_used or [],
        }
        self.logs.append(entry)
        logger.info("audit team=%s model=%s cost=%.4f", self.team_id, model, cost_usd)


@dataclass
class AgentResult:
    content: str
    tokens_used: int
    cost_usd: float
    budget_remaining: float
    circuit_breaker_triggered: Optional[str] = None
    events_logged: int = 0


class AgentHarness:
    """Orquesta la ejecución de un agente aplicando presupuesto, circuit
    breaker y auditoría de forma consistente (Capítulo 5).

    `agent_fn` debe devolver un dict con, al menos, las claves
    `content`, `tokens_input`, `tokens_output` y `cost_usd`; opcionalmente
    `thought`, `tools` y `observation` para alimentar el circuit breaker.
    """

    def __init__(
        self, budget: TokenBudget, breaker: AgentCircuitBreaker, audit: AuditLogger, model_name: str = "unknown"
    ):
        self.budget = budget
        self.breaker = breaker
        self.audit = audit
        self.model_name = model_name

    def run(
        self, agent_fn: Callable[[str], dict], task: str, correlation_id: str, estimated_cost: Optional[float] = None
    ) -> AgentResult:
        if self.breaker.is_open():
            return AgentResult("", 0, 0.0, self.budget.remaining, circuit_breaker_triggered=self.breaker.failure_reason)

        # Reserva optimista de presupuesto antes de ejecutar, para no gastar
        # tokens en una llamada que de todas formas se descartaría.
        reserved = estimated_cost if estimated_cost is not None else len(task) * 0.00001
        if not self.budget.record_cost(reserved):
            return AgentResult("", 0, 0.0, self.budget.remaining, circuit_breaker_triggered="budget_exceeded")

        try:
            result = agent_fn(task)
            tokens_in = result.get("tokens_input", 0)
            tokens_out = result.get("tokens_output", 0)
            cost = result.get("cost_usd", 0.0)
            content = result.get("content", "")

            self.breaker.record_cycle(
                thought=result.get("thought", ""),
                tools=result.get("tools", []),
                observation=result.get("observation", ""),
            )
            self.audit.log(
                self.model_name, tokens_in, tokens_out, cost, task, content, tools_used=result.get("tools", [])
            )

            return AgentResult(content, tokens_in + tokens_out, cost, self.budget.remaining)
        except Exception as exc:
            self.breaker.record_cycle(thought="error", tools=[], observation=str(exc))
            logger.exception("Fallo ejecutando agent_fn para correlation_id=%s", correlation_id)
            raise
