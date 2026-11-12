# Synthetic Code — reference implementation

[![CI](https://github.com/sergioide007/synthetic-code/actions/workflows/ci.yml/badge.svg)](https://github.com/sergioide007/synthetic-code/actions)

Production patterns for systems built with **AI agents**: budgets and routing, tamper-evident audit logs,
evals as a CI gate, bounded autonomy, supervised migrations, compliance evidence. It is the companion code of the
book *Synthetic Code: The Art of Orchestrating AI Agents*, but it is a normal Python library you can use on its own.

* **Offline by default.** Every pattern and every example runs without an API key (deterministic mock provider).
* **No mandatory dependencies** for the core patterns; real providers, Redis, Postgres and the dashboard are optional extras.
* **Tested, and the tests were tested.** 565 tests (real Redis + PostgreSQL in CI; a real-Kafka job too), Python 3.10 – 3.13 (CI matrix: 3.10-3.12; 3.13 verified locally), property-based tests, 91 % mutation score across 15 security- and concurrency-relevant modules, third-party checks (`jq`+`openssl`, OpenTofu, Kubernetes types). See [docs/VERIFICATION.md](docs/VERIFICATION.md) — including what was **not** verified.
* 🇪🇸 Documentación en español: [README.es.md](README.es.md)

> **Reference code, not a certified product.** Read it, test it in your environment, and have your security, legal and
> compliance people review it before using it in regulated systems. Nothing here is legal advice.

## Install

```bash
git clone https://github.com/sergioide007/synthetic-code.git && cd synthetic-code
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .                    # core patterns only
pip install -e ".[all]"             # + real providers, Redis, MCP, dashboard, pydantic, numpy, faiss, confluent-kafka
# or pick only what you need:  .[structured] (pydantic)  .[faiss] (numpy + faiss-cpu)  .[kafka] (confluent-kafka)
sintetico --version
```

## Five-minute tour (no API key needed)

```python
from sintetico import AgentAuditLogger, AIGateway, BudgetExceededError
from sintetico.providers import MockProvider        # swap for a real provider later

audit = AgentAuditLogger("a-long-random-secret")    # HMAC-chained: editing history is detectable
gw = AIGateway(MockProvider(), audit=audit)
gw.add_team("support", monthly_budget_usd=50)

r = gw.complete("support", "Classify this ticket: 'where is my order?'", complexity="simple")
print(r.model, f"${r.cost_usd:.6f}")                # cheap tier chosen automatically
print(audit.export_report())                        # cost, tokens, approvals, chain_valid
```

Run the eleven guided examples; each one ends with `OK`:

```bash
for f in examples/[01]*.py; do python "$f" || break; done          # Linux / macOS / Git Bash
```
```powershell
Get-ChildItem examples\[01]*.py | ForEach-Object { python $_.FullName }   # Windows PowerShell
```

| Example | What it shows | Book chapter |
|---------|---------------|--------------|
| `01_gateway_budget_audit` | One gateway in front of every LLM call: routing, per-team budgets, tamper-evident audit | 3, 5 |
| `02_ci_trust_gate` | Eval set -> trust score -> CI exit code | 6, 11 |
| `03_governance_autonomy` | Who may act alone: autonomy bounded by verifiability and cost of error | 7, 9 |
| `04_compliance_evidence` | Signed, write-once compliance evidence; data residency by country | 21, 27 |
| `05_iac_review` | Rule-based review of a Terraform/OpenTofu plan before `apply` | 22 |
| `06_incident_toolbox` | Incident tools with human approval and a dead-letter replay | 25 |
| `07_supervised_migration` | Dual-write migration; the cutover is proposed, a human decides | 26 |
| `08_shadow_and_swarm` | A validating second agent and a swarm gated by trust and approval | 6, 8 |
| `09_resilient_react` | Schema-validated ReAct with three independent stop conditions | 4 |
| `10_semantic_cache` | Semantic cache and its main risk, the false hit (needs NumPy; FAISS optional) | 18 |
| `11_distributed_agents` | Agents as workers: submit, retry, dead-letter (in-memory or real Kafka) | 3, 10 |

## Adopting it in a real project (step by step)

Do these in order; each step is useful on its own and none requires a rewrite.

| # | Step | What you do | Where |
|---|------|-------------|-------|
| 1 | **See your spend** | Route your existing LLM calls through `AIGateway`; give each team a budget | `examples/01` |
| 2 | **Make history tamper-evident** | Pass an `AgentAuditLogger` (key from `SINTETICO_AUDIT_KEY`); verify with `sintetico audit-verify` | `examples/01` |
| 3 | **Write an eval set** | 20–50 real cases in JSON (`expected_contains`, `forbidden_contains`, `critical`) | `tests/fixtures/sample_evalset.json` |
| 4 | **Gate your CI** | `sintetico eval` then `sintetico gate`; exit codes decide the build | below |
| 5 | **Bound autonomy** | `DynamicThresholdRACIGate`, `compute_max_allowed_autonomy`, `DetectionGuard` | `examples/03` |
| 6 | **Validate with a second agent** | `ProductionPipeline` + `ShadowAgent`; escalate on doubt | `examples/08` |
| 7 | **Production infrastructure** | `sintetico.devops`: evidence, IaC review, incident tools, migrations | `examples/04`–`07` |

### Using a real model

```bash
pip install -e ".[all]"
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY, or LOCALLLM_BASE_URL
SINTETICO_PROVIDER=anthropic python examples/01_gateway_budget_audit.py
```

In your code replace `MockProvider()` with `RealAnthropicProvider()` / `RealOpenAIProvider()` (`sintetico.real_providers`).
Model ids and prices live in `sintetico/model_registry.py` — they are a dated snapshot; **verify them against your
provider's current pricing** and override as needed.

### CI gate (GitHub Actions)

```yaml
- run: pip install -e .
- run: sintetico eval --evalset evals/support.json --agent myapp.agent:answer --min-pass-rate 0.85
- run: sintetico gate --results results.json --previous-score "${{ vars.LAST_TRUST_SCORE }}"
```

| Command | Exit code |
|---------|-----------|
| `sintetico gate` | 0 ok · 1 quality below threshold · 2 critical bug · 3 trust drift |
| `sintetico eval` | 0 ok · 1 below pass-rate · 2 critical case failed |
| `sintetico iac-review` | 0 approve · **2 human review (pauses, not red)** · 1 blocked |
| `sintetico audit-verify` / `evidence-verify` | 0 intact/valid · 1 tampered/invalid |
| `sintetico admission-serve` | Kubernetes webhook that refuses agents with a low trust annotation |

`--agent` is any `package.module:function` that takes a `str` and returns a `str`.

## What is where (book → code)

| Book chapters | Module(s) |
|---------------|-----------|
| 3 Gateway, budgets · 9 context budget | `budget`, `gateway`, `model_registry`, `providers` |
| 4 ReAct, ToT, Reflexion | `reasoning` |
| **4 Resilient ReAct / ToT (schema-validated steps)** | `structured`, `reasoning_resilient` |
| **3, 10 Asynchronous architecture: gateway + workers + Kafka** | `distributed` (`InMemoryBus`, `KafkaBus`, `AgentWorker`, `DistributedAIGateway`) |
| **18 Semantic cache (linear baseline and FAISS)** | `semantic_cache` |
| 5 Harness, circuit breaker, audit | `resilience`, `audit`, `trazabilidad` |
| 6 Trust · 11 TrustScore CI · deploy gate | `quality`, `delivery` |
| 7 RACI · 9 invariants · autonomy | `governance`, `autonomy` |
| 8–10, 14–16 swarm, debt, docs, escalation, irreversible decisions | `orchestration`, `delivery`, `docs`, `guards`, `governance` |
| 12 Code review | `review` |
| 13 Debugging | `debugging`, `guards` |
| 17 Security, MCP | `security` |
| 18 Cost optimization | `economics`, `evals` |
| 19–20 Backend-agnostic | `providers`, `adapters` |
| **21 Compliance evidence** | `devops.compliance` |
| **22 IaC review** | `devops.iac_review` |
| **23 GitOps trust gate** | `devops.gitops` |
| **25 Incident toolbox, DLQ** | `devops.incident` |
| **26 Supervised migration** | `devops.migration` |
| **27 Multi-country residency** | `devops.residency` (+ `examples/profiles`) |

### Nothing from the book's class list is missing any more

`AsyncAIGateway` is the *in-process* stand-in; the Kafka-capable one is `DistributedAIGateway`. `ReflexionAgent` is the book's
*Reflexion* pattern. Part VI listings that talk to cloud services (S3 object lock, Step Functions, Cosmos/SQL) are implemented
against in-memory or local stand-ins with the same semantics; connect your own adapters. Where the book overstates or
misdescribes a class, see [docs/BOOK_ERRATA.md](docs/BOOK_ERRATA.md) (measured numbers included).

## Known limits (read before production)

* The audit chain proves **integrity, not truth**, and **cannot see the deletion of the newest events** unless you store `log.head` somewhere else and pass it to `verify(expected_head=...)`.
* Cross-process audit safety uses POSIX `flock` (Linux/macOS, same host). Not safe over network filesystems; not provided on Windows.
* Live provider adapters were not exercised against real APIs here; model ids/prices are a dated snapshot — run `scripts/check_model_registry.py`.
* Regulatory profiles are **examples**, not legal advice.
* `KafkaBus` was verified against the real librdkafka client **without a broker**, against contract doubles, and (when you set `SINTETICO_TEST_KAFKA`) against a real broker; the last was **not run** by the author. Processing is **at-least-once**: make handlers idempotent.
* The Anthropic/OpenAI native structured-output hooks are **not live-tested**; without them the schema is validated and invalid answers are re-asked.
* `HashingEmbedder` is lexical, not semantic ("change" and "reset" my password score alike): bring a real embedder and tune the threshold on your own data. FAISS only beats an exact NumPy search from ~10^4 entries (see the errata for measurements).
* A Python thread cannot be killed: a worker abandons (does not stop) a handler that exceeds `task_timeout`.

Verify it yourself: `sh scripts/verify.sh` (tests, lint, types, security scan, dependency audit) and `docs/VERIFICATION.md`.

## Repository layout and language

| Path | Content |
|------|---------|
| `src/sintetico/` | The pattern library (core, no mandatory dependencies) and `sintetico.devops` |
| `src/sintetico_api/` | Optional observability API and dashboard (FastAPI) |
| `src/trazabilidad/` | Structured logging and a traced ReAct agent (*trazabilidad* = traceability) |
| `examples/` | Eleven short, offline, tested examples: **start here** |
| `tests/`, `scripts/`, `reports/` | Test suite, verification scripts and their saved reports |
| `docs/` | What was and was not verified; book errata found while implementing |
| `demos/`, `notebooks/`, `README.es.md` | Material of the Spanish edition of the book (prose in Spanish, same code) |

Documentation, examples, public docstrings and the command-line tool are in English. The package is named `sintetico`
after the Spanish title of the book (*Código sintético*). Some user-facing messages of the older modules, the
dashboard and the demos are still in Spanish; `ShadowAgent` and `ModelRouter` understand both languages.

## Design principles

1. **Rules that protect are code, not prompts.** An LLM may explain a decision; it never overrides a `BLOCK`.
2. **Fail closed.** Unknown country, missing metrics, missing consent or a crashing validator → deny or escalate.
3. **The agent proposes, a human accountable for the outcome decides** anything irreversible or with financial impact.
4. **Evidence over narrative.** Decisions leave signed, verifiable records.

## Tests

```bash
pip install -e ".[dev,api]"
pytest -q                      # Redis/Postgres tests skip unless SINTETICO_TEST_REDIS_URL / _POSTGRES_URL are set
```

CI (`.github/workflows/ci.yml`) starts real Redis and PostgreSQL services and runs everything on Python 3.10–3.12.

## Contributing

Issues and pull requests are welcome. Please add a test with every change and keep examples runnable offline.
Licence: see [LICENSE.txt](LICENSE.txt).
