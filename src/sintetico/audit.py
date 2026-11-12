"""Tamper-evident audit log (book Chapter 5).

Every event stores the HMAC-SHA256 of its own content *and* the hash of the
previous event, forming a verifiable chain. Editing, deleting or reordering any
past event breaks verification from that point on.

The chain proves *integrity* (nothing was altered after the fact) as long as the
secret key stays secret. It does not prove the events themselves were truthful.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

__all__ = ["AuditEvent", "AgentAuditLogger", "AuditKeyMissingError", "AuditCorruptionError"]

try:  # POSIX only; on Windows cross-process safety is not provided (documented)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

GENESIS = "genesis"


class AuditCorruptionError(RuntimeError):
    """The log file ends with a partially written line (e.g. a crash mid-write). Appending would corrupt it further."""


class AuditKeyMissingError(ValueError):
    """Raised when no signing key is provided (explicitly or via SINTETICO_AUDIT_KEY)."""


@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    actor: str
    action: str
    details: Dict[str, Any]
    prev_hash: str
    own_hash: str = ""

    def payload(self) -> bytes:
        body = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "actor": self.actor,
            "action": self.action,
            "details": self.details,
            "prev_hash": self.prev_hash,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


class AgentAuditLogger:
    """Append-only, HMAC-chained audit log with optional JSONL persistence."""

    def __init__(
        self,
        secret_key: Union[str, bytes, None] = None,
        path: Union[str, Path, None] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        key = secret_key if secret_key is not None else os.environ.get("SINTETICO_AUDIT_KEY")
        if not key:
            raise AuditKeyMissingError("Provide secret_key or set the SINTETICO_AUDIT_KEY environment variable")
        self._key = key.encode("utf-8") if isinstance(key, str) else key
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        self.events: List[AuditEvent] = []
        self.path = Path(path) if path else None
        self._offset = 0
        self._partial_tail = False
        if self.path and self.path.exists():
            self._load()

    def _sign(self, event: AuditEvent) -> str:
        return hmac.new(self._key, event.payload(), hashlib.sha256).hexdigest()

    @contextlib.contextmanager
    def _file_lock(self) -> Iterator[None]:
        """Exclusive inter-process lock (POSIX flock on a sidecar file). Without it, two processes appending to
        the same log would each extend their own in-memory tail and fork the chain."""
        if self.path is None or fcntl is None:
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.path) + ".lock", "a") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)

    def log(self, actor: str, action: str, **details: Any) -> AuditEvent:
        with self._lock, self._file_lock():
            if self.path:
                self._read_new_events()                      # adopt events appended by other processes
                if self._partial_tail:
                    raise AuditCorruptionError(f"{self.path} ends with a partial line; repair or rotate the file before appending")
            prev = self.events[-1].own_hash if self.events else GENESIS
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                timestamp=self._clock().isoformat(),
                actor=actor,
                action=action,
                details=details,
                prev_hash=prev,
            )
            event.own_hash = self._sign(event)
            self.events.append(event)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                line = (json.dumps(asdict(event), sort_keys=True, default=str) + "\n").encode("utf-8")
                with self.path.open("ab") as fh:
                    fh.write(line)
                    fh.flush()
                    os.fsync(fh.fileno())                    # durable before we report success
                self._offset += len(line)
            return event

    @property
    def head(self) -> str:
        """Hash of the newest event. Store it *outside* the log (ticket, other system, printed): an HMAC chain
        cannot detect deletion of the last events on its own, but a remembered head can."""
        return self.events[-1].own_hash if self.events else GENESIS

    def verify(self, expected_head: Optional[str] = None) -> Tuple[bool, Optional[int]]:
        """Return ``(True, None)`` if the chain is intact, else ``(False, index_of_first_bad_event)``.

        With ``expected_head`` also detects truncation: a chain that ends earlier than the anchor fails
        with index ``len(events)``."""
        prev = GENESIS
        for i, ev in enumerate(self.events):
            if ev.prev_hash != prev:
                return False, i
            if not hmac.compare_digest(ev.own_hash, self._sign(ev)):
                return False, i
            prev = ev.own_hash
        if expected_head is not None and not hmac.compare_digest(prev, expected_head):
            return False, len(self.events)
        return True, None

    def _load(self) -> None:
        if self.path is None:
            return
        self._offset = 0
        self._read_new_events()
        ok, bad = self.verify()
        if not ok:
            raise ValueError(f"Audit log {self.path} failed verification at event index {bad}")

    def _read_new_events(self) -> None:
        """Append to memory any events other processes wrote after our last read (tracked by byte offset)."""
        if self.path is None or not self.path.exists():
            return
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            data = fh.read()
        if not data:
            return
        # only consume complete lines; a partially written tail is picked up next time
        complete = data[: data.rfind(b"\n") + 1]
        self._partial_tail = len(data) > len(complete)
        for line in complete.decode("utf-8").splitlines():
            if line.strip():
                self.events.append(AuditEvent(**json.loads(line)))
        self._offset += len(complete)

    def export_report(self) -> Dict[str, Any]:
        """Aggregate cost, tokens and human approvals (used for external auditors)."""
        cost = sum(float(e.details.get("cost_usd", 0) or 0) for e in self.events)
        tokens = sum(int(e.details.get("tokens", 0) or 0) for e in self.events)
        approvals = sum(1 for e in self.events if e.action == "human_approval")
        ok, bad = self.verify()
        return {"events": len(self.events), "total_cost_usd": round(cost, 6), "total_tokens": tokens,
                "human_approvals": approvals, "chain_valid": ok, "first_invalid_index": bad}
