"""Review a Terraform plan before `apply`: the rules are code, so they cannot be talked out of a BLOCK."""
from sintetico.devops import Decision, IaCReviewer, InstanceMetrics

TAGS = {"env": "prod", "owner": "payments", "compliance": "pci"}
plan = {"resource_changes": [{"address": "aws_db_instance.ledger", "type": "aws_db_instance", "change": {
    "actions": ["update"],
    "before": {"identifier": "ledger", "instance_class": "db.r6g.2xlarge", "tags": TAGS},
    "after": {"identifier": "ledger", "instance_class": "db.r6g.large", "tags": TAGS}}}]}

# Average CPU was low, so a naive "optimizer" proposes a downsize. P99 and the memory working set say otherwise.
metrics = {"ledger": InstanceMetrics(p99_cpu_pct=94, min_freeable_memory_gb=36)}
result = IaCReviewer().review(plan, metrics=metrics)
print(result.decision.value, "->", result.reasons)
print("CI exit code:", result.exit_code, "(1 = blocked)")
assert result.decision is Decision.BLOCK
print("OK")
