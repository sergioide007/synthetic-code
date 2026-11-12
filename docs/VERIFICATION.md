# How this code was verified — and how you can check it without trusting the author

This document separates **what was checked**, **by which tool**, **how to reproduce it**, and **what was NOT checked**.
Every check below uses a third-party tool or a primary source; none of it depends on a statement by the author or by the AI
that helped write the code. Re-run everything with `sh scripts/verify.sh` (add `--full` for mutation testing).

## 1. Evidence at a glance

| Question | Evidence | Tool / source | Reproduce |
|----------|----------|---------------|-----------|
| Does it work? | **565 tests pass** with real Redis + PostgreSQL (3 real-Kafka-broker tests skipped, see section 4); **468 pass, 83 skipped (service tests)** on **Python 3.10, 3.11, 3.12 and 3.13** from a clean install of the `dev` extras | pytest, uv | `pytest -q` |
| Are the tests any good? | Mutation testing: **91 %** of deliberately broken variants were caught (467 of 514 sampled mutants, 15 security- and concurrency-relevant modules; per-module table below) | `scripts/mutation_check.py` (AST mutations) | `sh scripts/verify.sh --full` |
| Does it hold for inputs nobody thought of? | 11 property-based tests, hundreds of generated cases each (tamper detection, fail-closed policy, no-crash fuzzing) | Hypothesis | `pytest tests/test_properties.py` |
| Is the signature scheme what the docs say? | HMAC chain and evidence packages **verified by a separate implementation** (`jq` + `openssl`, no project code), including non-ASCII, nested values, and 4 attack types | `scripts/verify_*_with_openssl.sh` | `pytest tests/test_independent_verification.py` |
| Does it match real Terraform output? | Parser exercised on a **real `tofu show -json` plan** (create / update / delete), not a hand-written dict | OpenTofu v1.13.2 | `tests/fixtures/PROVENANCE.md` |
| Does the webhook speak Kubernetes? | Response fields and required fields checked against the **official Go types** (`kubernetes/api`, commit in the fixture) | raw source | `tests/test_primary_sources.py` |
| Are AWS attribute names right? | Checked against the provider's own documentation sources (`db_instance`, `security_group`, `security_group_rule`, `vpc_security_group_ingress_rule`) | `hashicorp/terraform-provider-aws` docs | `tests/fixtures/PROVENANCE.md` |
| Are model ids/prices current? | Updated from the vendor docs (2026-11-01); `scripts/check_model_registry.py` checks ids against **your** account's Models API | platform.claude.com docs | `ANTHROPIC_API_KEY=... python scripts/check_model_registry.py` |
| Static quality / security | ruff (bug + security rules), mypy, bandit: **0 findings** on the new code | ruff, mypy, bandit | `sh scripts/verify.sh` |
| Vulnerable dependencies? | **0 known vulnerabilities in 45 packages** (the full `[all]` tree) (report date in the file) | pip-audit | `reports/pip_audit.json` |
| What is inside? | CycloneDX 1.6 SBOM, 45 components | cyclonedx-py | `reports/sbom.cdx.json` |
| Is the CI itself valid? | `.github/workflows/ci.yml` passes **actionlint** | actionlint 1.7.12 | `actionlint` |
| Runs on someone else's machine? | The `quality` and `test` CI jobs run on GitHub's runners the first time you push. **Not yet run** — see section 4 | GitHub Actions | push, then open the Actions tab |

### Added in 2.2.0 (resilient agents, FAISS cache, distributed execution)

| Question | Evidence | Reproduce |
|----------|----------|-----------|
| Does a malformed model answer break the agent? | Prose, fenced JSON, wrong types, `"high"` scores, extra fields: each is re-asked with the exact validation errors, bounded, then reported (`invalid_output`) instead of raised | `tests/test_structured.py`, `tests/test_reasoning_resilient.py` |
| Does loop protection really work? | Each of the three layers is triggered in isolation, including the circuit breaker (30 distinct calls to one tool) | `tests/test_mutation_gaps5.py` |
| Is the FAISS cache right? | Property test: exact mode agrees with the linear baseline on hit and score; IVF recall >= 0.9 vs exact search; evicted ids never returned; 8-thread stress; a stale-neighbour false miss found and fixed | `tests/test_semantic_cache.py` |
| Is it fast? | **Measured**, not assumed: `reports/semantic_cache_benchmark.json`. FAISS beats an exact NumPy search only from ~10^4 entries; the book's absolute numbers were overstated (`docs/BOOK_ERRATA.md`) | `python scripts/bench_semantic_cache.py` |
| Does at-least-once hold? | Crash before publishing (task re-runs), crash after publishing but before commit (duplicate result ignored by the gateway), failed result/DLQ publication (nothing lost), competing workers (each task once), poison messages, deadlines, timeouts | `tests/test_distributed*.py` |
| Does the Kafka adapter work? | (1) **real librdkafka without a broker**: config validation, fail-fast on an unreachable broker; (2) contract doubles built with the real `confluent_kafka` types for every error path; (3) real-broker tests in `tests/test_kafka_bus.py::TestRealBroker` — **skipped here, run by the CI job `kafka-integration`** | `SINTETICO_TEST_KAFKA=host:port pytest tests/test_kafka_bus.py` |
| Works without the extras? | A bare install imports, star-imports and runs the distributed core; each extra fails with an actionable message (subprocess tests that simulate the missing library) | `tests/test_optional_dependencies.py` |

`mockafka` (an in-memory Kafka simulator) was evaluated and **rejected**: it returns `offset=None`, never invokes delivery callbacks and does not model consumer groups, so passing its tests would prove little.

### Mutation score per module (fixed seed, up to 60 mutants each)

| Module | Score | | Module | Score |
|--------|-------|-|--------|-------|
| `devops/residency.py` | 100 % | | `delivery.py` | 95 % |
| `autonomy.py` | 97 % | | `gateway.py` | 95 % |
| `devops/gitops.py` | 97 % | | `devops/migration.py` | 94 % |
| `audit.py` | 92 % | | `evals.py` | 89 % |
| `devops/iac_review.py` | 87 % | | `guards.py` | 80 % |
| `devops/compliance.py` | 76 % | | `structured.py` | 98 % |
| `reasoning_resilient.py` | 93 % | | `distributed.py` | 88 % (24 of 145 sampled) |
| `semantic_cache.py` | 75 % (32 of 146 sampled) | | | |

Surviving mutants are listed by `scripts/mutation_check.py`. The ones reviewed were equivalent (default argument values,
`mkdir(exist_ok=…)` flags, JSON indent, float tolerances) rather than missing checks, but that review was done by the author of the
tests, so treat it with the same scepticism as any self-assessment.

## 2b. Defects found while implementing the 2.2.0 classes

1. **The book's `SemanticCache` does not exist in the repo**: the repo's class is an exact-match LRU. A real embedding-based baseline had to be written before FAISS could be compared honestly.
2. **My first "vectorised" baseline was slower than a Python loop** (it rebuilt the matrix on every query), which would have made FAISS look better than it is; rewritten with a preallocated matrix.
3. **False misses from stale neighbours**: evicted entries still in the index could fill the top-k and hide a valid entry; the search now widens when (and only when) stale ids were seen.
4. **`AsyncAIGateway` was a simulation** (`sleep(0.1)`), not a Kafka gateway; `DistributedAIGateway` is the real one.
5. **The base `TreeOfThoughts` evaluator was not reproducible across processes** (salted `hash()`).
6. **`import sintetico` would have failed without numpy**; two `assert`s and two swallowed exceptions in production paths removed.

## 2. Defects that this process found (before and after the first "all tests pass")

The first version passed all of its own tests. Independent checks then found:

1. **Evidence overwrite race** — `exists()` then write; two writers could both succeed. Now exclusive-create.
2. **Audit log forked across processes** — each process extended its own in-memory tail. Now `flock` + tail sync (POSIX).
3. **Crash mid-write corrupted the next append** — now fails closed (`AuditCorruptionError`).
4. **Tail truncation invisible to an HMAC chain** — now detectable with an externally stored `head` anchor.
5. **Webhook read an unbounded body** — now capped at 1 MiB (413).
6. **Webhook blocked every `DELETE`** of an agent (Kubernetes sends `object: null`) — found by reading the official types.
7. **`extract_json` crashed on deep nesting** (`RecursionError`) and was quadratic on pathological input — found by fuzzing.
8. **IaC review missed `aws_security_group_rule`, `aws_vpc_security_group_ingress_rule`, protocol `-1`, IPv6 `::/0`** — found by reading the provider docs; unknown protocol is now treated as the worst case.
9. **Burst-window parser ignored minutes** (`23:00-02:30` ended at 02:00) — found by a boundary test.
10. **Model registry ids and prices were out of date** versus the vendor documentation.
11. **`ShadowAgent` only understood Spanish** model output; **two tests failed without redis-py**; **`hypothesis` was missing from the dev extras**.
12. Mutation testing exposed 40+ unpinned boundaries (e.g. `error_rate == 0.20`, `trust == threshold`, the six database ports) that are now asserted with literals.

## 3. Reproduce from scratch

```bash
git clone https://github.com/sergioide007/synthetic-code.git && cd synthetic-code
pip install -e ".[dev]" ruff mypy bandit pip-audit        # + jq and openssl from your OS
sh scripts/verify.sh                                         # ~1 minute
sh scripts/verify.sh --full                                  # + mutation testing, ~10 minutes
# independent re-check of an audit log, with NO project code:
SINTETICO_AUDIT_KEY=... sh scripts/verify_audit_with_openssl.sh audit.jsonl <stored-head-hash>
```

## 4. What was NOT verified (read this before relying on the code)

* **No real Kafka broker was available** in the environment used by the author (Maven Central / apache.org are unreachable, no Docker). The real-broker tests exist and are wired into CI; **until that job is green, treat the Kafka path as unproven in production conditions** (rebalances, partitions >1, broker failover, exactly-once are untested).
* **The native structured-output hooks** (Anthropic forced tool use, OpenAI `json_schema`) are checked against fake SDK clients only.
* **FAISS concurrency scaling** was not measurable (1 CPU); latency/recall numbers are from that single machine.
* **`HashingEmbedder` is lexical**; semantic quality and the right similarity threshold depend on your embedder and data.
* **At-least-once, not exactly-once**: handlers must be idempotent; a stuck handler thread is abandoned, not killed.

* **No live provider calls.** `RealAnthropicProvider`, `RealOpenAIProvider` and `LocalLLMProvider` were not exercised against real
  APIs in this environment (no keys). The `AIGateway` is tested with deterministic fakes. Run `scripts/check_model_registry.py` and one real call.
* **Pricing.** The docs list Haiku as "from $0.10 / $0.50"; confirm your tier. Prices change; `AIGateway` budgets are only as good as `model_registry.py`.
* **Only Linux was tested.** Cross-process audit safety uses POSIX `flock`; on Windows it falls back to in-process locking only. Multi-host or network-filesystem writers are **not** safe.
* **AWS behaviour was not tested against the real `aws` provider** (no registry access here): the plan shape was verified with OpenTofu built-in resources, attribute names against AWS docs.
* **GitHub Actions has not run this CI** (it needs a push). Treat the first green run as part of the verification.
* **Legal/regulatory content is illustrative.** `examples/profiles/*.json` are example values, not legal advice; the code enforces whatever you put in them.
* **Security properties are limited to what the tests state.** The HMAC log proves *integrity given a secret key*, not that events were truthful; anyone with the key can forge entries. No penetration test or third-party code audit was performed.
* **Scale and performance** were not measured.
* **The mutation review of surviving mutants is the author's own** (section 1).
