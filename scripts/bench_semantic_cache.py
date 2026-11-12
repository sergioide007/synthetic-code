#!/usr/bin/env python3
"""Measure (not assume) the cost of a semantic-cache lookup: Python loop vs NumPy matrix product vs FAISS.

    python scripts/bench_semantic_cache.py [--dim 384] [--sizes 100 1000 10000 50000] [--queries 200] [--out reports/semantic_cache_benchmark.json]

Embeddings are precomputed random unit vectors, so the numbers isolate SEARCH cost (a real embedder adds its own latency,
usually 5-50 ms per query, which no cache index can remove). Queries are noisy copies of stored vectors.
"""
import argparse
import json
import os
import platform
import sys
import threading
import time
import numpy as np
from sintetico.semantic_cache import EmbeddingSemanticCache, HighConcurrencySemanticCache


class Table:
    def __init__(self, m): self.m, self.dim = m, m.shape[1]
    def embed(self, text): return self.m[int(text.split(":")[1])]


def pct(xs, p): return float(np.percentile(xs, p))


def timed_gets(cache, names, n):
    lat = []
    for name in names[:n]:
        t = time.perf_counter(); cache.get(name); lat.append((time.perf_counter() - t) * 1000)
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=384); ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000, 50000])
    ap.add_argument("--queries", type=int, default=200); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--out", default="")
    a = ap.parse_args()
    import faiss
    rng = np.random.default_rng(0)
    rows = []
    for n in a.sizes:
        q_n = a.queries
        base = rng.normal(size=(n + q_n, a.dim)).astype(np.float32); base /= np.linalg.norm(base, axis=1, keepdims=True)
        targets = rng.integers(0, n, size=q_n)
        noisy = base[targets] + 0.03 * rng.normal(size=(q_n, a.dim)).astype(np.float32)
        base[n:] = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
        emb = Table(base)
        exact = [int(np.argmax(base[:n] @ base[n + j])) for j in range(q_n)]
        caches = {"python-loop": EmbeddingSemanticCache(emb, max_size=n, similarity_threshold=0.0, vectorized=False),
                  "numpy": EmbeddingSemanticCache(emb, max_size=n, similarity_threshold=0.0, vectorized=True),
                  "faiss": HighConcurrencySemanticCache(emb, max_size=n, similarity_threshold=0.0, search_k=1)}
        qnames = [f"v:{n + j}" for j in range(q_n)]
        row = {"entries": n, "dim": a.dim}
        for name, c in caches.items():
            t0 = time.perf_counter()
            for i in range(n): c.set(f"v:{i}", str(i))
            fill_s = time.perf_counter() - t0
            reps = max(20, q_n // 4) if name == "python-loop" and n >= 10000 else q_n
            lat = timed_gets(c, qnames, reps)
            got = [c.get(qnames[j]) for j in range(min(q_n, 100))]
            recall = float(np.mean([g == str(exact[j]) for j, g in enumerate(got)]))
            # concurrent throughput: T threads each doing a share of the queries
            share = max(10, min(q_n, 100))
            def work(c=c, qnames=qnames, q_n=q_n, share=share):
                for j in range(share):
                    c.get(qnames[j % q_n])
            ts = [threading.Thread(target=work) for _ in range(a.threads)]
            t0 = time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]
            qps = a.threads * share / (time.perf_counter() - t0)
            row[name] = {"p50_ms": round(pct(lat, 50), 3), "p95_ms": round(pct(lat, 95), 3), "p99_ms": round(pct(lat, 99), 3),
                         "recall_at_1": round(recall, 3), "fill_s": round(fill_s, 2), f"qps_{a.threads}_threads": round(qps),
                         **({"index": c.stats()["index"]} if name == "faiss" else {})}
        rows.append(row)
        print(f"N={n:>6}  " + "   ".join(f"{k}: p50 {row[k]['p50_ms']:>8.3f} ms  p99 {row[k]['p99_ms']:>8.3f}  recall {row[k]['recall_at_1']:.2f}  {row[k][f'qps_{a.threads}_threads']:>6} q/s"
                                         for k in ("python-loop", "numpy", "faiss")), flush=True)
    meta = {"python": platform.python_version(), "faiss": faiss.__version__, "numpy": np.__version__, "cpus": os.cpu_count(), "machine": platform.machine(),
            "note": "search cost only (embeddings precomputed); noisy-copy queries; see scripts/bench_semantic_cache.py"}
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"meta": meta, "results": rows}, open(a.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
