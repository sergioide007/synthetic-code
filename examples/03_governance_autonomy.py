"""Who may act alone? Autonomy is limited by verifiability and by the cost of being wrong."""
from sintetico import (ActivitySpec, DetectionGuard, DomainRiskProfile, DynamicThresholdRACIGate, Verifiability,
                       compute_max_allowed_autonomy)
from sintetico.governance import RACILevel

for name, v in [("financial reports", Verifiability.SLOW_MANUAL), ("code review with tests", Verifiability.AUTOMATED)]:
    print(f"max autonomy for {name:<24}: {compute_max_allowed_autonomy(DomainRiskProfile(name, v, 0.6)):.2f}")

gate = DynamicThresholdRACIGate({
    "scan_logs": ActivitySpec(RACILevel.RESPONSIBLE, "sre-lead", error_cost=5),
    "notify_breach": ActivitySpec(RACILevel.RESPONSIBLE, "dpo", error_cost=95),
})
for activity in ("scan_logs", "notify_breach"):
    d = gate.authorize(activity, trust_score=0.5)
    print(f"{activity:<14} allowed={d.allowed!s:<5} needs_human={d.requires_human_approval!s:<5} (threshold {d.threshold})")

d = DetectionGuard().check(patch_text="ALTER TABLE accounts DROP COLUMN balance")
print("irreversible patch ->", d.reasons)
assert gate.authorize("scan_logs", 0.5).allowed and not gate.authorize("notify_breach", 0.5).allowed and d.blocked_irreversible
print("OK")
