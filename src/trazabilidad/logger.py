# ============================================================
# logger.py: structured logger for agentic systems
# ============================================================
# Version: 1.0.0
# Author: Sergio Perez Ruiz
# ============================================================
"""
Structured logger for production agentic systems.

Features:
- JSON Lines for easy ingestion (ELK, Datadog, Splunk)
- Automatic or caller-supplied correlation ID
- Automatic redaction of sensitive data
- Per-session aggregated metrics
- Configurable log levels
"""

from __future__ import annotations

import json
import time
import logging
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, Callable
from enum import Enum


class EventType(Enum):
    """Standard event types for agentic logs."""

    # Lifecycle
    AGENT_START = "agent_start"
    AGENT_FINISH = "agent_finish"
    AGENT_ERROR = "agent_error"

    # Reasoning
    REASONING_STEP = "reasoning_step"
    DECISION = "decision"

    # Tools
    TOOL_INVOKED = "tool_invoked"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"

    # Context
    CONTEXT_INJECTED = "context_injected"
    RAG_QUERY = "rag_query"

    # Security
    INPUT_SANITIZED = "input_sanitized"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    HUMAN_APPROVAL_REQUESTED = "human_approval_requested"
    HUMAN_APPROVAL_RESULT = "human_approval_result"

    # Metrics
    METRICS_SNAPSHOT = "metrics_snapshot"
    BUDGET_CHECK = "budget_check"


@dataclass
class StructuredLogEntry:
    """Structured log entry for agentic systems."""

    # Required
    timestamp: str
    level: str
    correlation_id: str
    event_type: str
    agent_id: str

    # Optional
    session_id: Optional[str] = None
    team_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_version: Optional[str] = None
    model_used: Optional[str] = None

    # Metrics
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    retry_count: Optional[int] = 0

    # Payload and links
    payload: Dict[str, Any] = field(default_factory=dict)
    parent_correlation_id: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON for the logs."""
        return json.dumps(asdict(self), default=str)


class StructuredAgentLogger:
    """Structured logger for production agentic systems.

    `sink`, if given, receives every entry as a `dict` (via
    `dataclasses.asdict`) in addition to the usual output to `output_stream`
    and the native logger. It is the extension point for connecting this
    logger to your own observability sink (a database, an OpenTelemetry
    collector, the `TraceStore` of `sintetico_api`, and so on) without
    coupling this class to any particular backend.
    """

    def __init__(
        self,
        agent_id: str,
        agent_version: str = "1.0.0",
        team_id: str = "default",
        environment: str = "production",
        log_level: str = "INFO",
        output_stream=sys.stdout,
        enable_sanitization: bool = True,
        sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.team_id = team_id
        self.environment = environment
        self.log_level = log_level.upper()
        self.output = output_stream
        self.enable_sanitization = enable_sanitization
        self.sink = sink

        # Configure native logging
        self._setup_native_logging()

        # Per-session metrics
        self._session_metrics = {}

        # Keys treated as sensitive by exact name (after lower-casing) or by
        # suffix. The original implementation flagged any key that *contained*
        # "token" anywhere, which silently masked legitimate metrics such as
        # `input_tokens`, `output_tokens`, or `total_tokens` (they contain
        # "token" as a substring but are not secrets), breaking the very
        # observability this logger exists to provide.
        self._sensitive_exact_keys = {
            "api_key",
            "password",
            "secret",
            "token",
            "ssn",
            "credit_card",
            "email",
            "phone",
            "auth",
            "credential",
        }
        self._sensitive_key_suffixes = (
            "_key",
            "_token",
            "_secret",
            "_password",
            "_credential",
            "_ssn",
            "_auth",
        )
        # Known business metrics that must never be masked, as defense in
        # depth on top of the whole-word/suffix matching above.
        self._metric_key_allowlist = {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "tokens_input",
            "tokens_output",
            "retry_count",
            "step_count",
            "tool_invocations",
            "errors",
        }

    def _setup_native_logging(self):
        """Configure native logging for compatibility.

        `logging.getLogger(name)` always returns the same instance for the
        same name: if several `StructuredAgentLogger`s are created with the
        same `agent_id` (common in tests or in workers that get re-created),
        adding a new handler each time would duplicate every log line. Our own
        handlers are removed before adding the new one, so the operation is
        idempotent.
        """
        self._native_logger = logging.getLogger(f"agent.{self.agent_id}")
        self._native_logger.setLevel(getattr(logging, self.log_level, logging.INFO))
        for existing in list(self._native_logger.handlers):
            if getattr(existing, "_sintetico_managed", False):
                self._native_logger.removeHandler(existing)

        if self.output is None:
            # No output stream (e.g. used as a library with a `sink`):
            # don't write anything to stderr by default.
            handler = logging.NullHandler()
        else:
            handler = logging.StreamHandler(self.output)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        handler._sintetico_managed = True
        self._native_logger.addHandler(handler)
        self._native_logger.propagate = False

    def _get_level_priority(self, level: str) -> int:
        levels = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        return levels.get(level.upper(), 20)

    def _should_log(self, level: str) -> bool:
        return self._get_level_priority(level) >= self._get_level_priority(self.log_level)

    def _is_sensitive_key(self, key_lower: str) -> bool:
        if key_lower in self._metric_key_allowlist:
            return False
        if key_lower in self._sensitive_exact_keys:
            return True
        return key_lower.endswith(self._sensitive_key_suffixes)

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively redact sensitive data."""
        if not self.enable_sanitization:
            return data

        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                key_lower = key.lower()
                needs_sanitization = self._is_sensitive_key(key_lower)
                if needs_sanitization:
                    sanitized[key] = "***"
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize(value)
                elif isinstance(value, list):
                    sanitized[key] = [self._sanitize(item) if isinstance(item, dict) else item for item in value]
                else:
                    sanitized[key] = value
            return sanitized
        return data

    def _create_entry(
        self,
        level: str,
        correlation_id: str,
        event_type: str,
        message: str,
        payload: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        latency_ms: Optional[float] = None,
        cost_usd: Optional[float] = None,
        model_used: Optional[str] = None,
        retry_count: Optional[int] = None,
        parent_correlation_id: Optional[str] = None,
    ) -> StructuredLogEntry:
        """Create a structured log entry."""
        if payload:
            payload = self._sanitize(payload)

        return StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.upper(),
            correlation_id=correlation_id,
            event_type=event_type,
            agent_id=self.agent_id,
            session_id=session_id,
            team_id=self.team_id,
            user_id=user_id,
            agent_version=self.agent_version,
            model_used=model_used,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            retry_count=retry_count,
            payload=payload or {"message": message},
            parent_correlation_id=parent_correlation_id,
        )

    def _emit(self, entry: StructuredLogEntry):
        """Emit the log entry to the configured stream, the native logger,
        and the registered `sink`, if any."""
        if not self._should_log(entry.level):
            return

        if self.output is not None:
            self.output.write(entry.to_json() + "\n")
            self.output.flush()

            log_method = getattr(self._native_logger, entry.level.lower(), self._native_logger.info)
            log_method(f"[{entry.correlation_id}] {entry.event_type}: {entry.payload.get('message', '')}")

        if self.sink is not None:
            self.sink(asdict(entry))

    # ============================================
    # LOG METHODS PER EVENT
    # ============================================

    def start_session(
        self,
        correlation_id: str,
        session_id: str,
        user_id: str,
        task_complexity: str = "moderate",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Start a new agent session."""
        self._session_metrics[correlation_id] = {
            "start_time": time.time(),
            "step_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
            "cost_total": 0.0,
            "tool_invocations": 0,
            "errors": 0,
        }

        entry = self._create_entry(
            level="INFO",
            correlation_id=correlation_id,
            event_type=EventType.AGENT_START.value,
            message="Agent session started",
            payload={
                "message": f"Session started for user {user_id}",
                "task_complexity": task_complexity,
                "metadata": metadata or {},
            },
            session_id=session_id,
            user_id=user_id,
        )
        self._emit(entry)

    def reasoning_step(
        self,
        correlation_id: str,
        step_number: int,
        thought: str,
        tool_invoked: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ):
        """Record a reasoning step."""
        if correlation_id in self._session_metrics:
            self._session_metrics[correlation_id]["step_count"] += 1
            if tool_invoked:
                self._session_metrics[correlation_id]["tool_invocations"] += 1

        entry = self._create_entry(
            level="INFO",
            correlation_id=correlation_id,
            event_type=EventType.REASONING_STEP.value,
            message=f"Reasoning step {step_number}",
            payload={
                "step_number": step_number,
                "thought": thought,
                "tool_invoked": tool_invoked,
                "tool_args": tool_args,
                "confidence": confidence,
            },
        )
        self._emit(entry)

    def tool_invocation(
        self,
        correlation_id: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        permission_level: str = "read",
    ):
        """Record a tool invocation."""
        entry = self._create_entry(
            level="INFO",
            correlation_id=correlation_id,
            event_type=EventType.TOOL_INVOKED.value,
            message=f"Tool invoked: {tool_name}",
            payload={
                "tool_name": tool_name,
                "tool_args": tool_args,
                "permission_level": permission_level,
            },
        )
        self._emit(entry)

    def tool_result(
        self,
        correlation_id: str,
        tool_name: str,
        success: bool,
        result: Any,
        error: Optional[str] = None,
        latency_ms: Optional[float] = None,
        cost_usd: Optional[float] = None,
        model_used: Optional[str] = None,
    ):
        """Record a tool result (or an LLM call, modelled as one more tool:
        passing `cost_usd`/`model_used` in that case lets later aggregations
        attribute cost per model)."""
        if correlation_id in self._session_metrics:
            if not success:
                self._session_metrics[correlation_id]["errors"] += 1
            if cost_usd:
                self._session_metrics[correlation_id]["cost_total"] += cost_usd

        entry = self._create_entry(
            level="INFO" if success else "ERROR",
            correlation_id=correlation_id,
            event_type=EventType.TOOL_RESULT.value,
            message=f"Tool result {tool_name}: {'OK' if success else 'FAIL'}",
            payload={
                "tool_name": tool_name,
                "success": success,
                "result": result if success else None,
                "error": error if not success else None,
                "latency_ms": latency_ms,
            },
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model_used=model_used,
        )
        self._emit(entry)

    def decision(
        self,
        correlation_id: str,
        decision: str,
        rationale: str,
        confidence: float,
        trust_score: float,
        human_approval_required: bool = False,
        cost_usd: Optional[float] = None,
    ):
        """Record an agent decision."""
        entry = self._create_entry(
            level="INFO",
            correlation_id=correlation_id,
            event_type=EventType.DECISION.value,
            message=f"Decision: {decision}",
            payload={
                "decision": decision,
                "rationale": rationale,
                "confidence": confidence,
                "trust_score": trust_score,
                "human_approval_required": human_approval_required,
            },
            cost_usd=cost_usd,
        )
        self._emit(entry)

    def error(
        self,
        correlation_id: str,
        error_type: str,
        error_message: str,
        error_stack: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "ERROR",
    ):
        """Record an agent error."""
        entry = self._create_entry(
            level=severity,
            correlation_id=correlation_id,
            event_type=EventType.AGENT_ERROR.value,
            message=f"Agent error: {error_type}",
            payload={
                "error_type": error_type,
                "error_message": error_message,
                "error_stack": error_stack,
                "context": context or {},
            },
        )
        self._emit(entry)

    def finish_session(
        self,
        correlation_id: str,
        status: str,
        message: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: Optional[float] = None,
        latency_ms: Optional[float] = None,
        retry_count: int = 0,
    ):
        """Finish an agent session with its metrics."""
        metrics = self._session_metrics.get(correlation_id, {})

        entry = self._create_entry(
            level="INFO" if status == "success" else "ERROR",
            correlation_id=correlation_id,
            event_type=EventType.AGENT_FINISH.value,
            message=f"Agent finished: {status}",
            payload={
                "status": status,
                "message": message,
                "metrics": {
                    "input_tokens": tokens_input,
                    "output_tokens": tokens_output,
                    "total_tokens": tokens_input + tokens_output,
                    "cost_usd": cost_usd,
                    "latency_ms": latency_ms,
                    "retry_count": retry_count,
                    "step_count": metrics.get("step_count", 0),
                    "tool_invocations": metrics.get("tool_invocations", 0),
                    "errors": metrics.get("errors", 0),
                },
            },
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            retry_count=retry_count,
        )
        self._emit(entry)

        if correlation_id in self._session_metrics:
            del self._session_metrics[correlation_id]

    def security_event(
        self,
        correlation_id: str,
        event_type: str,
        details: Dict[str, Any],
        severity: str = "WARNING",
    ):
        """Record security events."""
        entry = self._create_entry(
            level=severity,
            correlation_id=correlation_id,
            event_type=event_type,
            message=f"Security event: {event_type}",
            payload={"security_event_type": event_type, "details": details},
        )
        self._emit(entry)

    def get_session_metrics(self, correlation_id: str) -> Dict[str, Any]:
        """Return the metrics of a session."""
        return self._session_metrics.get(correlation_id, {})
