"""A second agent that only validates (Shadow) and a swarm whose members are gated by trust and approval."""
from sintetico import PipelineDecision, ProductionPipeline, SwarmOrchestrator
from sintetico.quality import ShadowAgent

rules = dict(strategy="constraints", constraints=["password", "iban"])
clean = ProductionPipeline(lambda task: "Total: 42 EUR", ShadowAgent(**rules))
leaky = ProductionPipeline(lambda task: "Customer IBAN is DE89...", ShadowAgent(**rules))
print("clean answer ->", clean.run("summarise invoice").decision.value)
print("leaky answer ->", leaky.run("summarise invoice").decision.value)

agents = {"analyst": lambda t: f"analysis of {t}", "writer": lambda t: f"draft of {t}", "deployer": lambda t: f"deployed {t}"}
swarm = SwarmOrchestrator(agents, trust_scores={"analyst": 0.9, "writer": 0.5, "deployer": 0.9}, roles_requiring_approval={"deployer"})
report = swarm.run("release 1.4", lambda task: [("analyst", task), ("writer", task), ("deployer", task)])
print("ran    :", list(report.results))
print("skipped:", report.skipped)
assert clean.run("x").decision is PipelineDecision.AUTO_APPROVED and leaky.run("x").decision is PipelineDecision.REJECTED
assert list(report.results) == ["0:analyst"]
print("OK")
