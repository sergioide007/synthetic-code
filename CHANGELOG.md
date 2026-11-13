# Changelog

## Unreleased

### Fixed
- `RealAnthropicProvider` no longer sends `temperature`. The Claude 5.5 models in the registry reject sampling
  parameters with HTTP 400 (Opus 5.5 always, Sonnet and Haiku 5.5 for non-default values), so every real call with
  the default `sonnet` alias failed. OpenAI and local providers still honor `LLMRequest.temperature`.
- Anthropic structured output (`complete_structured`) offers the schema tool with `tool_choice: auto` and asks for
  one call in the system prompt, instead of forcing the tool, which the Claude 5.5 models reject with HTTP 400. A
  text answer is validated and re-asked like any other provider's.
- `RealOpenAIProvider` and the OpenAI structured-output hook send `max_completion_tokens` instead of the deprecated
  `max_tokens`, which OpenAI's reasoning models (GPT-5 and later, GPT-6 included) don't accept, and send
  `temperature` only to the GPT-4 family (`openai_sampling_kwargs`).

### Changed
- `model_registry`: OpenAI catalog refreshed from OpenAI's published Standard-tier prices (2026-11-13). Added
  `gpt-6-astra` (quality, $10 / $50 per 1M tokens), `gpt-6.1-sol` (balanced, $2 / $10) and `gpt-6-luna` (fast,
  $0.10 / $0.50), all with a 1.05M-token context. `gpt-4o` corrected to $2.50 / $10. Removed `gpt-4-turbo` and
  `gpt-3.5-turbo`, which OpenAI no longer lists. Anthropic entries were already current (Claude 5.5 family).
- `adapters.PROFILES` now recommends current tiers (`gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna`), and a test
  guarantees every model a profile names is registered.

## 2.2.2 (2026-11-13)
Zenodo: [10.5281/zenodo.24196751](https://doi.org/10.5281/zenodo.24196751).

### Fixed
- `EmergencyOverrideGateway`: separation of duties. The person who requests an override can no longer approve it
  (case- and whitespace-insensitive), an approver must be named, and a request must name its requestor. A refused
  approval leaves the request pending for someone else.
- `sintetico gate` now reads the output of `sintetico eval` directly: it accepts `critical_failures` as well as
  `critical_bugs` (the larger count wins), so `eval > results.json && gate --results results.json` no longer loses
  a critical failure on the way.
### Changed
- Every user-facing string in `sintetico`, `sintetico_api` and `trazabilidad` is now English: exceptions, log
  messages, LLM prompts, the threat matrix, autonomy labels, OpenAPI tags and descriptions, and the dashboard. The
  docstrings and comments of `sintetico_api` and `trazabilidad` were translated too (verified by comparing syntax
  trees: only strings changed, apart from the fixes listed here).
- Heuristics that read user or model text understand English and Spanish: `EscalationPolicy` default keywords
  (cancel, lawsuit, lawyer, attorney, CEO, "talk to a human", plus the Spanish ones), `ModelRouter` rules
  (architecture, system, strategy, migration, where), `SecurityScanner` patterns (English jailbreak role-play and
  "repeat everything above"; finding type `exfiltracion_de_contexto` is now `context_exfiltration`), and the Tree of
  Thoughts parser (`Option N:` or `Opción N:`).
- `ShadowAgent` (adversarial) and `TreeOfThoughts` prompts now state the exact answer format their parsers expect.
- CI tests Python 3.13 as well; workflow token restricted to read-only (`permissions: contents: read`).
### Added
- `tests/test_i18n.py`: no Spanish user-facing strings, bilingual detection, and the prompt/parser contracts above.
- Zenodo archive and DOI ([10.5281/zenodo.24161299](https://doi.org/10.5281/zenodo.24161299)): badge and "How to cite" in the READMEs, `doi`, `identifiers`, `abstract` and `keywords` in `CITATION.cff`.
- `SECURITY.md` (private vulnerability reporting), `CONTRIBUTING.md`, Dependabot for pip and GitHub Actions, CodeQL workflow.

## 2.2.1 (2026-11-12)
### Changed
- Docstrings and comments of the core modules (`budget`, `governance`, `orchestration`, `quality`, `security`, ...) translated to English; code unchanged (checked by comparing syntax trees).
- `pyproject.toml`: English description, keywords, classifiers and project URLs. `CITATION.cff`, `.env.example`, `requirements.txt`, CI step names and `notebooks/README.md` in English.
- README: table of the eleven examples, Windows commands, repository layout and a note on which parts are still in Spanish.
- `.gitignore` covers the files the examples write (`audit.jsonl`, `*.db`) and tool caches; `.gitattributes` fixes line endings to LF.
### Fixed
- `examples/10_semantic_cache.py` failed without FAISS; it now falls back to the exact NumPy cache.
- `test_audit_chain_survives_concurrent_processes` failed on Windows (no `fork`); it is now skipped there, as documented.
- 18 lint findings in the tests (unused imports and variables).

## 2.2.0
### Added
- `structured` + `reasoning_resilient`: `AgentStep`, `Hypothesis`/`HypothesisBatch` (Pydantic), `complete_structured` (validate + re-ask with the errors, native hooks for Anthropic/OpenAI), `ReActAgentResilient` (three-layer loop protection) and `TreeOfThoughtsResilient`.
- `semantic_cache`: `HashingEmbedder`, `EmbeddingSemanticCache` (exact linear baseline) and `HighConcurrencySemanticCache` (FAISS flat -> IVF, readers-writer lock, LRU + TTL by tombstones, automatic rebuild).
- `distributed`: `InMemoryBus`, `KafkaBus` (confluent-kafka), `AgentWorker` (at-least-once, retries with backoff, DLQ, dedupe, deadlines), `DistributedAIGateway`, `react_handler`.
- Examples 09-11, `docker/kafka-compose.yml`, `scripts/bench_semantic_cache.py`, CI job against a real Kafka broker, extras `structured`, `faiss`, `kafka`.
- `docs/BOOK_ERRATA.md`: where the book overstates or misdescribes these classes, with measurements.
### Fixed
- `TreeOfThoughts._default_evaluator` used salted `hash()` (not reproducible between runs); now SHA-256.
- `import sintetico` and `from sintetico import *` work with no optional dependency installed.
## 2.1.0
### Added
- `sintetico` command-line tool (`gate`, `eval`, `audit-verify`, `evidence-verify`, `iac-review`, `admission-serve`) with CI-friendly exit codes.
- `AgentAuditLogger`: HMAC-chained, tamper-evident audit log with JSONL persistence and verification.
- `AIGateway`, `AIGatewayWithProjects`, `ContextBudgetManager`: routing by complexity, per-team and per-project budgets, audit.
- `EvalSet`, `CostOptimizerWithEvals`; `DomainRiskProfile`/`compute_max_allowed_autonomy`; `DynamicThresholdRACIGate`.
- `DeployGate`, `AgentArchitectureReviewer`, `ProductionPipeline`, `SwarmOrchestrator`; `CodeReviewAgent` and review patterns; `DetectionGuard`.
- `PromptAdapter`, `OutputAdapter`, `CapabilityProfile`; `DocumentGenerator`, `ADRGenerator`, `ADRArchive`.
- `sintetico.devops` (book Part VI): compliance evidence, residency policy, IaC review, incident toolbox + DLQ replay, supervised migration, GitOps trust webhook.
- Eight offline runnable examples; 227 new tests (suite total: 415 with real services).
- Verification tooling: `scripts/verify.sh`, `scripts/mutation_check.py`, `scripts/check_model_registry.py`, `scripts/verify_{audit,evidence}_with_openssl.sh`, `docs/VERIFICATION.md`, SBOM and dependency-audit reports.
- `AgentAuditLogger.head` / `verify(expected_head=...)` to detect truncation; `AuditCorruptionError`.
- CI `quality` job (ruff, mypy, bandit, pip-audit, independent verifiers).
### Fixed
- Evidence store overwrite race; audit log forking across processes and corruption after a crash; unbounded webhook body; webhook blocking `DELETE`.
- `extract_json` recursion/quadratic behaviour; IaC review gaps (modern security-group rules, protocol `-1`, IPv6, unknown protocol, burst-window minutes).
- Model registry ids and prices updated from the vendor documentation (verify with `scripts/check_model_registry.py`).
- Two tests that failed without `redis-py`; `hypothesis` added to the `dev` extras.
### Changed
- `ShadowAgent` heuristics now understand English as well as Spanish model output.
- `MPCHost` renamed to `MCPHost` (the old name remains as an alias).
- English `README.md`; the previous Spanish README is now `README.es.md`.
