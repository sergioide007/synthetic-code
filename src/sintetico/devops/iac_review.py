"""Deterministic infrastructure-change review (book Chapter 22).

The rules that matter for safety are *code*, not a prompt: P99 (not average) CPU, the
memory working set, downstream consumers and compliance flags. An LLM may explain
or enrich a decision, but it never overrides a BLOCK.

Input is the JSON from ``terraform show -json plan.tfplan``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["Decision", "InstanceMetrics", "ReviewResult", "IaCReviewer", "RDS_MEMORY_GB", "DB_PORTS"]

# Memory per RDS instance class (GiB). Extend or pass ``memory_overrides``.
RDS_MEMORY_GB: Dict[str, float] = {
    "db.r6g.large": 16, "db.r6g.xlarge": 32, "db.r6g.2xlarge": 64, "db.r6g.4xlarge": 128, "db.r6g.8xlarge": 256,
    "db.r5.large": 16, "db.r5.xlarge": 32, "db.r5.2xlarge": 64, "db.r5.4xlarge": 128,
    "db.t3.medium": 4, "db.t3.large": 8,
}
DB_PORTS = {1433, 1521, 3306, 5432, 6379, 27017}
REQUIRED_TAGS = ("env", "owner", "compliance")
TAGGABLE = {"aws_db_instance", "aws_instance", "aws_s3_bucket", "aws_lambda_function", "aws_ecs_service"}


class Decision(enum.Enum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


@dataclass
class InstanceMetrics:
    p99_cpu_pct: float
    min_freeable_memory_gb: float


@dataclass
class ReviewResult:
    decision: Decision
    reasons: List[str] = field(default_factory=list)
    required_approvers: List[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        """CI convention: 0 approved, 2 pause for a human (not a red build), 1 blocked."""
        return {Decision.APPROVE: 0, Decision.REVIEW: 2, Decision.BLOCK: 1}[self.decision]

    def to_dict(self) -> Dict[str, Any]:
        return {"decision": self.decision.value, "reasons": self.reasons, "required_approvers": self.required_approvers, "exit_code": self.exit_code}


def _ingress_rules(rtype: str, after: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Normalise the three ways AWS ingress rules appear in a plan (names verified against the provider docs):
    inline ``ingress`` blocks, ``aws_security_group_rule`` (type=ingress) and ``aws_vpc_security_group_ingress_rule``."""
    if rtype == "aws_security_group":
        return [r for r in (after.get("ingress") or []) if isinstance(r, dict)]
    if rtype == "aws_security_group_rule" and after.get("type") == "ingress":
        return [after]
    if rtype == "aws_vpc_security_group_ingress_rule":
        return [{"cidr_blocks": [after.get("cidr_ipv4")] if after.get("cidr_ipv4") else [],
                 "ipv6_cidr_blocks": [after.get("cidr_ipv6")] if after.get("cidr_ipv6") else [],
                 "protocol": after.get("ip_protocol"), "from_port": after.get("from_port"), "to_port": after.get("to_port")}]
    return []


def _opens_db_port_to_world(rule: Mapping[str, Any]) -> bool:
    world = "0.0.0.0/0" in (rule.get("cidr_blocks") or []) or "::/0" in (rule.get("ipv6_cidr_blocks") or [])
    if not world:
        return False
    proto = str(rule.get("protocol") or "tcp").lower()          # unknown/missing protocol: assume the worst (fail closed)
    if proto in ("-1", "all"):                      # "-1" = every protocol and port (ports are ignored by AWS)
        return True
    if proto not in ("tcp", "6"):
        return False                                # udp/icmp rules do not expose a TCP database port
    lo, hi = rule.get("from_port"), rule.get("to_port")
    lo = 0 if lo is None else int(lo)
    hi = 65535 if hi is None else int(hi)           # unspecified bound: assume the worst
    return any(lo <= p <= hi for p in DB_PORTS)


def _parse_window(window: str) -> Optional[tuple]:
    """``"23:00-02:30"`` -> (start_minute, end_minute) of the day; the window may wrap midnight."""
    try:
        a, b = window.split("-")
        (ah, am), (bh, bm) = (a.strip().split(":"), b.strip().split(":"))
        return int(ah) * 60 + int(am), int(bh) * 60 + int(bm)
    except (ValueError, AttributeError):
        return None


def _hour_overlaps_window(window: tuple, hour: int) -> bool:
    """True if ANY minute of the given UTC hour falls inside the window (conservative: the exact minute of the
    change is usually unknown). The window is half-open: [start, end)."""
    s, e = window
    lo, hi = hour * 60, hour * 60 + 59
    if s <= e:
        return lo <= e - 1 and hi >= s
    return hi >= s or lo <= e - 1                        # wraps midnight


class IaCReviewer:
    def __init__(self, memory_overrides: Optional[Mapping[str, float]] = None) -> None:
        self.memory = {**RDS_MEMORY_GB, **(memory_overrides or {})}

    def review(self, plan: Mapping[str, Any], dependency_map: Optional[Mapping[str, Any]] = None,
               metrics: Optional[Mapping[str, InstanceMetrics]] = None, change_hour_utc: Optional[int] = None,
               monthly_cost_delta_usd: float = 0.0) -> ReviewResult:
        """Never raises on garbage input: an unreadable plan is a BLOCK (fail closed), not a stack trace."""
        try:
            return self._review(plan, dependency_map, metrics, change_hour_utc, monthly_cost_delta_usd)
        except (AttributeError, TypeError, KeyError, ValueError) as exc:
            return ReviewResult(Decision.BLOCK, [f"plan could not be analysed ({type(exc).__name__}); refusing to approve"], [])

    def _review(self, plan: Mapping[str, Any], dependency_map: Optional[Mapping[str, Any]] = None,
                metrics: Optional[Mapping[str, InstanceMetrics]] = None, change_hour_utc: Optional[int] = None,
                monthly_cost_delta_usd: float = 0.0) -> ReviewResult:
        dep = dependency_map or {}
        metrics = metrics or {}
        block: List[str] = []
        review: List[str] = []
        approvers: List[str] = []
        frameworks = [str(f).upper() for f in dep.get("compliance_frameworks", [])]
        pci_scoped = any("PCI" in f for f in frameworks)
        consumers = dep.get("downstream_consumers", [])
        worst_breach = max((float(c.get("breach_cost_usd", 0)) for c in consumers), default=0.0)
        burst = _parse_window(dep.get("infrastructure", {}).get("burst_window", ""))

        changes = [rc for rc in plan.get("resource_changes", []) if set(rc.get("change", {}).get("actions", [])) - {"no-op", "read"}]
        if changes and str(dep.get("regulatory_sensitivity", "")).upper() == "STRICT":
            review.append("service is regulatory_sensitivity=STRICT: two approvers required")
            approvers += ["principal-engineer", "security-lead"]

        for rc in changes:
            addr, rtype = rc.get("address", rc.get("type", "?")), rc.get("type", "")
            ch = rc.get("change", {})
            before, after = ch.get("before") or {}, ch.get("after") or {}

            if rtype == "aws_db_instance" and "update" in ch.get("actions", []):
                b_cls, a_cls = before.get("instance_class"), after.get("instance_class")
                if b_cls != a_cls:
                    b_mem = self.memory.get(b_cls) if b_cls else None
                    a_mem = self.memory.get(a_cls) if a_cls else None
                    ident = after.get("identifier") or before.get("identifier") or addr
                    if b_mem is None or a_mem is None:
                        review.append(f"{addr}: unknown instance class ({b_cls} -> {a_cls}); cannot assess")
                    else:
                        downsizing = a_mem < b_mem
                        m = metrics.get(ident) or metrics.get(addr)
                        if downsizing and m is None:
                            review.append(f"{addr}: downsize {b_cls} -> {a_cls} without P99/memory metrics")
                        elif downsizing and m is not None:
                            working_set = b_mem - m.min_freeable_memory_gb
                            if working_set > 0.80 * a_mem:
                                block.append(f"{addr}: peak working set ~{working_set:.0f} GiB does not fit 80% of {a_mem:.0f} GiB ({a_cls})")
                            if m.p99_cpu_pct >= 80:
                                block.append(f"{addr}: P99 CPU {m.p99_cpu_pct:.0f}% too high to downsize")
                            elif m.p99_cpu_pct > 40:
                                review.append(f"{addr}: P99 CPU {m.p99_cpu_pct:.0f}% (> 40%) on a downsize")
                        if worst_breach > 5000:
                            review.append(f"{addr}: downstream breach cost ${worst_breach:,.0f} per occurrence")
                        if burst and change_hour_utc is not None:
                            if _hour_overlaps_window(burst, change_hour_utc):
                                review.append(f"{addr}: change scheduled inside the downstream burst window")
                if before.get("multi_az") is True and after.get("multi_az") is False and (pci_scoped or "pci" in str(after.get("tags", {})).lower()):
                    block.append(f"{addr}: removing multi-AZ on a PCI-scoped resource")

            for rule in _ingress_rules(rtype, after):
                if _opens_db_port_to_world(rule):
                    block.append(f"{addr}: opens a database port to the whole internet (0.0.0.0/0 or ::/0)")
                    break

            if rtype in TAGGABLE and set(ch.get("actions", [])) & {"create", "update"}:
                tags = after.get("tags") or {}
                missing = [t for t in REQUIRED_TAGS if t not in tags]
                if missing:
                    review.append(f"{addr}: missing required tags {missing}")

        if monthly_cost_delta_usd > 500:
            review.append(f"monthly cost delta ${monthly_cost_delta_usd:,.0f} > $500")

        if block:
            return ReviewResult(Decision.BLOCK, block + review, sorted(set(approvers)))
        if review:
            return ReviewResult(Decision.REVIEW, review, sorted(set(approvers)) or ["principal-engineer"])
        return ReviewResult(Decision.APPROVE, ["no risk rule triggered"], [])
