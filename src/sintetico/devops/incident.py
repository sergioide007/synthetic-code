"""Incident-response toolbox: every tool call is budgeted, loop-checked, gated and audited
(book Chapter 25). The agent proposes; tools tagged ``financial_impact`` need a human."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from ..audit import AgentAuditLogger

__all__ = ["ToolPolicy", "IncidentSession", "ToolOutcome", "GuardedToolbox", "InMemoryQueue", "inspect_dlq", "replay_dlq", "ApprovalRequired"]


class ApprovalRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    permission: str = "READ"                      # READ | WRITE
    financial_impact: bool = False
    requires_approval: bool = False
    approver_roles: Tuple[str, ...] = ()


@dataclass
class IncidentSession:
    correlation_id: str
    service: str
    audit: AgentAuditLogger
    max_seconds: float = 900.0
    max_tool_calls: int = 60
    loop_window: int = 6
    loop_repeats: int = 3
    clock: Callable[[], float] = time.monotonic
    started: float = 0.0
    calls: List[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def elapsed(self) -> float:
        return self.clock() - self.started

    def looping(self) -> bool:
        recent = self.calls[-self.loop_window:]
        return any(recent.count(s) >= self.loop_repeats for s in set(recent))


@dataclass
class ToolOutcome:
    ok: bool
    result: Any = None
    error: Optional[str] = None
    escalate: bool = False


ApprovalFn = Callable[[ToolPolicy, Mapping[str, Any]], bool]


class GuardedToolbox:
    def __init__(self, session: IncidentSession, policies: Sequence[ToolPolicy], executors: Mapping[str, Callable[..., Any]],
                 approval: Optional[ApprovalFn] = None) -> None:
        self.session = session
        self.policies = {p.name: p for p in policies}
        self.executors = dict(executors)
        self.approval = approval

    def _deny(self, tool: str, why: str, escalate: bool = True) -> ToolOutcome:
        self.session.audit.log(f"incident:{self.session.correlation_id}", "tool_denied", tool=tool, reason=why)
        return ToolOutcome(False, error=why, escalate=escalate)

    def invoke(self, tool: str, **params: Any) -> ToolOutcome:
        s = self.session
        policy = self.policies.get(tool)
        if policy is None or tool not in self.executors:
            return self._deny(tool, "unknown tool", escalate=False)
        if s.elapsed() > s.max_seconds:
            return self._deny(tool, f"session time budget exhausted ({s.max_seconds:.0f}s): escalate to on-call")
        if len(s.calls) >= s.max_tool_calls:
            return self._deny(tool, "tool-call budget exhausted: escalate to on-call")
        sig = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]
        s.calls.append((tool, sig))
        if s.looping():
            return self._deny(tool, "loop detected (same tool + same input repeated): escalate to on-call")
        if policy.financial_impact and policy.requires_approval:
            if self.approval is None or not self.approval(policy, params):
                return self._deny(tool, f"approval required from {list(policy.approver_roles)}")
            s.audit.log("human", "human_approval", tool=tool, roles=list(policy.approver_roles))
        t0 = time.monotonic()
        try:
            result = self.executors[tool](**params)
        except Exception as exc:  # noqa: BLE001 - a failing tool must not crash the incident loop
            s.audit.log(f"incident:{s.correlation_id}", "tool_error", tool=tool, input_hash=sig, error=f"{type(exc).__name__}: {exc}")
            return ToolOutcome(False, error=f"{type(exc).__name__}: {exc}")
        s.audit.log(f"incident:{s.correlation_id}", "tool_call", tool=tool, input_hash=sig, permission=policy.permission,
                    duration_ms=round((time.monotonic() - t0) * 1000), policy_decision="ALLOW")
        return ToolOutcome(True, result)


class InMemoryQueue:
    """Tiny FIFO standing in for SQS/Kafka in tests and demos."""

    def __init__(self, items: Optional[Sequence[Any]] = None) -> None:
        self._q: Deque[Any] = deque(items or [])

    def send(self, item: Any) -> None:
        self._q.append(item)

    def peek(self, n: int = 10) -> List[Any]:
        return list(self._q)[:n]

    def receive(self) -> Optional[Any]:
        return self._q.popleft() if self._q else None

    def __len__(self) -> int:
        return len(self._q)


def inspect_dlq(dlq: InMemoryQueue, n: int = 3) -> Dict[str, Any]:
    """Read-only: look at a dead-letter queue without consuming it."""
    return {"depth": len(dlq), "sample": dlq.peek(n)}


def replay_dlq(dlq: InMemoryQueue, target: InMemoryQueue, approver: Optional[str], notify: Callable[[Dict[str, Any]], None],
               max_messages: int = 1000) -> Dict[str, Any]:
    """Financial-impact write: needs a named approver and *always* notifies Finance. Preserves order."""
    if not approver:
        raise ApprovalRequired("DLQ replay requires a named human approver")
    replayed = 0
    while replayed < max_messages and len(dlq):
        target.send(dlq.receive())
        replayed += 1
    summary = {"replayed": replayed, "remaining": len(dlq), "approver": approver}
    notify(summary)
    return summary
