"""Give an incident agent tools it cannot misuse: budgets, loop detection, human gate for financial impact."""
from sintetico import AgentAuditLogger
from sintetico.devops import GuardedToolbox, InMemoryQueue, IncidentSession, ToolPolicy, inspect_dlq, replay_dlq

audit = AgentAuditLogger("incident-audit-key")
session = IncidentSession("inc-42", "payments", audit)
dlq, main_queue = InMemoryQueue([{"event": i} for i in range(4)]), InMemoryQueue()
tools = GuardedToolbox(
    session,
    policies=[ToolPolicy("search_logs"), ToolPolicy("replay_dlq", "WRITE", financial_impact=True, requires_approval=True,
                                                  approver_roles=("finance-lead", "eng-lead"))],
    executors={"search_logs": lambda q: f"3 matches for {q!r}",
               "replay_dlq": lambda approver: replay_dlq(dlq, main_queue, approver, notify=print)},
    approval=None,                       # nobody approved: the agent can read, it cannot replay
)
print(tools.invoke("search_logs", q="HikariPool timeout").result)
print(inspect_dlq(dlq, 2))
denied = tools.invoke("replay_dlq", approver="agent")
print("replay without approval ->", denied.error)

tools.approval = lambda policy, params: True          # a human clicked "approve" in your ticketing/chat tool
tools.invoke("replay_dlq", approver="finance-lead")
print("audit chain intact:", audit.verify()[0], "| events:", [e.action for e in audit.events])
assert not denied.ok and len(main_queue) == 4 and audit.verify()[0]
print("OK")
