"""One gateway in front of every LLM call: routing by complexity, per-team budgets, tamper-evident audit.

The first thing to adopt in a real project: wrap the calls you already make.
"""
from _provider import get_provider

from sintetico import AgentAuditLogger, AIGateway, BudgetExceededError

audit = AgentAuditLogger("change-me-in-production", path="audit.jsonl")     # use SINTETICO_AUDIT_KEY in real life
gateway = AIGateway(get_provider(), audit=audit)
gateway.add_team("support", monthly_budget_usd=50.0)
gateway.add_team("research", monthly_budget_usd=0.0001)                       # deliberately tiny

cheap = gateway.complete("support", "Classify: 'where is my order?'", complexity="simple")
deep = gateway.complete("support", "Analyse this architecture trade-off ...", complexity="complex")
print(f"simple  -> {cheap.model:<22} ${cheap.cost_usd:.6f}")
print(f"complex -> {deep.model:<22} ${deep.cost_usd:.6f}")

try:
    gateway.complete("research", "x" * 4000, complexity="complex", max_tokens=4000)
except BudgetExceededError as exc:
    print("blocked before spending anything:", exc)

report = audit.export_report()
print("audit:", report)
assert report["chain_valid"] and report["events"] == 2 and gateway.budgets["research"].spent == 0
print("OK")
