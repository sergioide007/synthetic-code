# Book ↔ code: what the implementation showed the text gets wrong or overstates

Found while implementing `AgentWorker`, `ReActAgentResilient`, `TreeOfThoughtsResilient` and `HighConcurrencySemanticCache`
(English edition, `book_en.md`, line numbers of the manuscript snapshot used).

**Status (Kindle edition, November 2026):** items 1-4, 6-10 are corrected in the text of the book, and the semantic-cache table now uses the measured figures below. Item 5 (false-hit risk) was added to the text. Item 11 needed no change. Earlier printings keep the original wording, so this table remains the reference for them.

| # | Where | The book says | What the code / measurements show | Suggested fix |
|---|-------|---------------|-----------------------------------|---------------|
| 1 | l.2713 | With 10,000 entries each linear `get()` takes "50-100 ms" | Measured p50 with 384-d embeddings: **13.6 ms** (pure-Python loop), **0.83 ms** (NumPy matrix product). A Python loop reaches ~100 ms at **50,000** entries (101 ms) | "...a Python loop costs ~14 ms at 10,000 entries and ~100 ms at 50,000" |
| 2 | l.2719 | 100 entries: "~5 ms linear vs ~0.1 ms FAISS"; 10,000 entries: "~150 ms vs ~1 ms" | 100 entries: 0.20 ms loop / 0.02 ms NumPy / 0.04 ms FAISS. 10,000: 13.6 / 0.83 / **0.47** ms. 50,000: 101 / 8.2 / **1.1** ms. The linear numbers are overstated ~10x-50x | Replace with the table below, and say the numbers depend on hardware and dimension |
| 3 | l.2719 | FAISS reduces search "from O(n) to O(log n)" | `IndexIVFFlat` is *sub-linear and approximate* (≈ nlist + nprobe·n/nlist). Measured recall@1: **0.98** (10k), **0.95** (50k) with default `nprobe`. `nprobe` trades recall for latency | "...reduces the cost sub-linearly, at the price of approximate results (tune `nprobe`)" |
| 4 | l.2719 | Implies FAISS is needed once the cache grows | A vectorised exact search (NumPy) is as fast as FAISS up to a few thousand entries and exact. FAISS wins clearly only from ~10^4 entries (1.8x at 10k, 7.5x at 50k) and under concurrency | Say so: "below ~5,000 entries a matrix product is enough" |
| 5 | l.2719 | (omission) | Embedding the query costs 5-50 ms with a real model and no index removes it; the lexical `HashingEmbedder` scores "change my password" as high as "reset my password" (0.79) -> a wrong cached answer is possible | Add the false-hit risk and the need to tune the threshold on your own questions (see `examples/10_semantic_cache.py`) |
| 6 | l.879 | `AsyncAIGateway` "publishes each task to Kafka" | The repository's `AsyncAIGateway` is an **in-process thread-pool stand-in** (it sleeps and returns a fixed string). The Kafka-backed one is **`DistributedAIGateway`** | Name `DistributedAIGateway` in the Kafka sentence; keep `AsyncAIGateway` as the local stand-in |
| 7 | l.879 | "...or non-blocking by subscribing to a webhook" | Not implemented. `wait_for_result(task_id, timeout)` blocks; `get_task` polls | Remove, or mark as an exercise |
| 8 | l.879 | `AgentWorker` ... "runs the ReAct cycle, and publishes the result" | True via `react_handler(...)`. Semantics are **at-least-once** (commit after publishing the result): a crash re-runs the task; handlers must be idempotent. Not exactly-once | Add "at-least-once; make handlers idempotent" |
| 9 | l.1105 | The resilient agent "reuses exactly the same `_should_stop`" | The repository's base `ReActAgent` has **no** `_should_stop`. `ReActAgentResilient._should_stop` is new and has three layers: thought similarity (>= 85 % over the last 3), tool signature (same tool + args + result more than twice), and `AgentCircuitBreaker` (flooding) | "...combines three independent stop conditions" |
| 10 | l.1105 | Ask the provider to "**guarantee**" the schema | Anthropic (forced tool use) and OpenAI (`response_format=json_schema`) hooks exist but are **not live-tested** (no API keys here). For any other provider the schema goes in the prompt and invalid answers are re-asked with the validation errors (default 3 attempts), then `StructuredOutputError` | "...enforce the schema natively where the API supports it, and validate + re-ask where it does not" |
| 11 | l.1125 | `TreeOfThoughtsResilient` "replaces `_generate_children`" | It overrides `solve` as well (to use the validated scores) and keeps `_generate_children` for compatibility; the base `_default_evaluator` used Python's salted `hash()` and was **not reproducible between processes** (fixed with SHA-256) | Fine as text; mention reproducibility if the demo is shown |

Measured with `scripts/bench_semantic_cache.py` (Python 3.12, faiss 1.15, numpy 2.4, **1 CPU**, 384-d random unit vectors, noisy-copy queries, search cost only):

| Entries | Python loop p50 / p99 | NumPy exact p50 / p99 | FAISS p50 / p99 | FAISS recall@1 |
|--------:|----------------------:|----------------------:|----------------:|---------------:|
| 100     | 0.20 / 0.30 ms        | 0.02 / 0.07 ms        | 0.04 / 0.14 ms  | 1.00 |
| 1,000   | 1.06 / 2.0 ms         | 0.07 / 0.19 ms        | 0.09 / 0.22 ms  | 1.00 |
| 10,000  | 13.6 / 22 ms          | 0.83 / 1.5 ms         | 0.47 / 0.79 ms  | 0.98 |
| 50,000  | 101 / 117 ms          | 8.2 / 11 ms           | 1.1 / 2.3 ms    | 0.95 |

Not measurable here: **concurrency scaling** (the sandbox has one CPU). `HighConcurrencySemanticCache` lets readers proceed in parallel
(readers-writer lock, FAISS releases the GIL during search); whether that scales on your hardware must be measured there
(`python scripts/bench_semantic_cache.py` prints queries/second with N threads).
