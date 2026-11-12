"""Semantic caching: answer a rephrased question from the cache instead of paying for the model again - and its main risk.

The built-in HashingEmbedder is LEXICAL (it compares shared words), so it runs offline but is NOT true semantic search.
Look at the output: it scores "how do I change my password" exactly as high as "how can I reset my password", although
they may need different answers. A cache that returns the answer to a DIFFERENT question is worse than a cache miss.
For real use: pass your own embedder (sentence-transformers, an embeddings API) with `.dim` and `.embed(text)`, pick the
threshold by measuring false hits on YOUR questions, and keep an eval set for it.
Needs NumPy:  pip install numpy          (with FAISS as well:  pip install "codigo-sintetico[faiss]")
"""
try:
    from sintetico import EmbeddingSemanticCache, HashingEmbedder, HighConcurrencySemanticCache
except ImportError:
    raise SystemExit("This example needs NumPy:  pip install numpy")

emb = HashingEmbedder(256)
try:                                    # FAISS index: the version for large caches and many concurrent readers
    cache = HighConcurrencySemanticCache(emb, max_size=1000, similarity_threshold=0.75, ttl_seconds=3600)
except ImportError:                     # no FAISS installed: exact NumPy search, same get/set interface
    cache = EmbeddingSemanticCache(emb, max_size=1000, similarity_threshold=0.75)
print("cache:", type(cache).__name__)
FAQ = {"how do i reset my password": "Use the 'Forgot password' link on the sign-in page.",
       "what are your opening hours": "We are open Monday to Friday, 9:00-18:00."}
for q, a in FAQ.items():
    cache.set(q, a)

print(f"{'query':34} {'best cosine':>11}  decision (threshold 0.75)")
for q in ("how do i reset my password", "how can i reset my password", "reset password how", "how do i change my password",
          "what time do you open", "do you ship to chile"):
    best = max(float(emb.embed(q) @ emb.embed(s)) for s in FAQ)
    hit = cache.get(q)
    print(f"{q:34} {best:11.2f}  {'HIT' if hit else 'miss -> call the model, then cache.set(...)'}")
if hasattr(cache, "stats"):
    print("stats:", cache.stats())
assert cache.get("how do i reset my password") and cache.get("how can i reset my password")      # a useful paraphrase hit
assert cache.get("how do i change my password")                                                  # ... and the lexical embedder's FALSE hit
assert cache.get("do you ship to chile") is None and cache.get("what time do you open") is None   # unrelated or too far: miss
print("OK")
