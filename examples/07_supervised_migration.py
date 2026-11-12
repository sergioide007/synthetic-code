"""Dual-write migration: the primary stays authoritative, checksums prove content, a human flips the switch."""
from sintetico.devops import DualWriteRepository, MigrationAdvisor, Phase, verify_batch

FIELDS = ["id", "balance", "currency"]
primary, secondary, sql_up = {}, {}, [False]


def write_secondary(i, rec):
    if not sql_up[0]:
        raise ConnectionError("target DB unavailable")
    secondary[i] = rec


repo = DualWriteRepository(primary, write_secondary)
for i in range(5):
    repo.write(str(i), {"id": str(i), "balance": 100 * i, "currency": "USD"})
print("primary rows:", len(primary), "| secondary rows:", len(secondary), "| DLQ:", len(repo.dlq), "(writes were NOT lost)")

sql_up[0] = True
print("reprocess DLQ:", repo.reprocess_dlq())
secondary["3"]["balance"] += 1                                       # silent corruption: same row count, different content
report = verify_batch(list(primary), primary, secondary, FIELDS)
print("checksum mismatches:", report.mismatches)

advisor = MigrationAdvisor()
print("blockers:", advisor.evaluate(Phase.VERIFY, 14, 0, 3, report.mismatch_rate, True).blockers)
secondary["3"]["balance"] -= 1
rec = advisor.evaluate(Phase.VERIFY, 14, 0, 3, verify_batch(list(primary), primary, secondary, FIELDS).mismatch_rate, True)
print(rec.action, "requires_human =", rec.requires_human, "approvers =", rec.approvers)
assert report.mismatches and rec.action == "PROPOSE_CUTOVER" and rec.requires_human
print("OK")
