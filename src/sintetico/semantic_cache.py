"""Embedding-based semantic caches: a linear baseline and a FAISS-backed high-concurrency version (book Chapter 18).

* :class:`HashingEmbedder` -- dependency-light, deterministic, offline. It is **lexical** (hashed word uni/bigrams), so it
  catches rephrasings that share words, not paraphrases that share only meaning. For real semantics pass your own
  ``embedder`` (an object with ``dim`` and ``embed(text) -> np.ndarray``: sentence-transformers, an embeddings API...).
* :class:`EmbeddingSemanticCache` -- exact cosine search, O(n) per ``get``. The "reference implementation" the book warns about.
* :class:`HighConcurrencySemanticCache` -- same API over a FAISS index: exact search while small, ``IndexIVFFlat`` once big
  (approximate: raise ``nprobe`` for recall at the cost of latency), many concurrent readers, LRU + TTL eviction by tombstones
  (no O(n) index surgery per write), automatic rebuild when the index has grown or filled with tombstones.

Requires numpy; the FAISS class additionally needs ``faiss-cpu`` (``pip install "codigo-sintetico[faiss]"``).
Both caches are in-memory and per-process; they are not persisted and not shared between processes.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Set, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without numpy
    raise ImportError('sintetico.semantic_cache needs numpy: pip install "codigo-sintetico[faiss]"') from exc

faiss: Any
try:  # optional
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised only without the extra installed
    faiss = None

__all__ = ["Embedder", "HashingEmbedder", "EmbeddingSemanticCache", "HighConcurrencySemanticCache"]

_WORD = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Feature-hashing embedder (signed hashed unigrams + bigrams, L2-normalised). Deterministic across processes."""

    def __init__(self, dim: int = 256, bigram_weight: float = 0.5) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim, self.bigram_weight = dim, bigram_weight

    def _slot(self, feature: str) -> Tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, text: str) -> np.ndarray:
        words = _WORD.findall(text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        for w in words:
            i, s = self._slot(w)
            vec[i] += s
        for a, b in zip(words, words[1:], strict=False):
            i, s = self._slot(a + " " + b)
            vec[i] += s * self.bigram_weight
        n = float(np.linalg.norm(vec))
        return vec / n if n > 0 else vec          # an empty text embeds to the zero vector, which never matches anything


def _normalise_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _unit(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class _Stats:
    def __init__(self) -> None:
        self.hits = self.misses = 0
        self.lock = threading.Lock()

    def record(self, hit: bool) -> None:
        with self.lock:
            if hit:
                self.hits += 1
            else:
                self.misses += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ========================================================================================= linear baseline
class EmbeddingSemanticCache:
    """Exact nearest-neighbour by cosine similarity: O(n) per ``get``. Thread-safe, LRU-bounded.

    Vectors live in one pre-allocated float32 matrix (swap-delete on eviction), so ``vectorized=True`` is a single
    matrix-vector product with no per-query copying: an honest, fast baseline. ``vectorized=False`` loops in Python over
    the rows like a textbook implementation (kept as the slow reference for benchmarks)."""

    def __init__(self, embedder: Optional[Embedder] = None, *, max_size: int = 10_000, similarity_threshold: float = 0.92,
                 vectorized: bool = True) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.embedder = embedder or HashingEmbedder()
        self.max_size, self.threshold, self.vectorized = max_size, similarity_threshold, vectorized
        self._mat: Optional[np.ndarray] = None
        self._keys: List[str] = []                       # row -> key
        self._row: Dict[str, int] = {}                   # key -> row
        self._resp: Dict[str, str] = {}
        self._order: "OrderedDict[str, None]" = OrderedDict()   # LRU order, oldest first
        self._lock = threading.Lock()
        self._stats = _Stats()

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def hits(self) -> int: return self._stats.hits
    @property
    def misses(self) -> int: return self._stats.misses
    @property
    def hit_rate(self) -> float: return self._stats.hit_rate

    def get(self, query: str) -> Optional[str]:
        found = self.get_with_score(query)
        return found[0] if found else None

    def get_with_score(self, query: str) -> Optional[Tuple[str, float]]:
        key = _normalise_text(query)
        q = _unit(self.embedder.embed(key))
        with self._lock:
            best_key, best = None, -1.0
            n = len(self._keys)
            if key in self._row:
                best_key, best = key, 1.0
            elif n and self._mat is not None:
                if self.vectorized:
                    sims = self._mat[:n] @ q
                    j = int(np.argmax(sims))
                    best_key, best = self._keys[j], float(sims[j])
                else:
                    for j in range(n):
                        s = float(np.dot(self._mat[j], q))
                        if s > best:
                            best_key, best = self._keys[j], s
            if best_key is not None and best >= self.threshold:
                self._order.move_to_end(best_key)
                self._stats.record(True)
                return self._resp[best_key], min(best, 1.0)
        self._stats.record(False)
        return None

    def _remove_row(self, key: str) -> None:
        j, last = self._row.pop(key), len(self._keys) - 1
        if j != last:                                     # swap-delete keeps the matrix compact
            moved = self._keys[last]
            self._mat[j] = self._mat[last]                # type: ignore[index]
            self._keys[j], self._row[moved] = moved, j
        self._keys.pop()
        self._resp.pop(key, None)
        self._order.pop(key, None)

    def set(self, query: str, response: str) -> None:
        key = _normalise_text(query)
        vec = _unit(self.embedder.embed(key))
        with self._lock:
            if self._mat is None:
                self._mat = np.zeros((self.max_size, vec.shape[0]), dtype=np.float32)
            if key in self._row:
                self._order.move_to_end(key)
                self._resp[key] = response
                return
            if len(self._keys) >= self.max_size:
                self._remove_row(next(iter(self._order)))
            self._row[key] = len(self._keys)
            self._mat[len(self._keys)] = vec
            self._keys.append(key)
            self._resp[key] = response
            self._order[key] = None


# ===================================================================================== FAISS, high concurrency
class _RWLock:
    """Readers-writer lock, writer-preferring so a stream of reads cannot starve writes."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers, self._writer, self._writers_waiting = 0, False, 0

    @contextmanager
    def read(self) -> Iterator[None]:
        with self._cond:
            while self._writer or self._writers_waiting:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if not self._readers:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        with self._cond:
            self._writers_waiting += 1
            while self._writer or self._readers:
                self._cond.wait()
            self._writers_waiting -= 1
            self._writer = True
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


@dataclass
class _Entry:
    key: str
    response: str
    vec: np.ndarray
    expires_at: Optional[float]


class HighConcurrencySemanticCache:
    def __init__(self, embedder: Optional[Embedder] = None, *, max_size: int = 100_000, similarity_threshold: float = 0.92,
                 nlist: Optional[int] = None, nprobe: Optional[int] = None, ivf_min_vectors: int = 2048,
                 ttl_seconds: Optional[float] = None, search_k: int = 8, rebuild_growth: float = 2.0,
                 tombstone_ratio: float = 0.25, clock: Callable[[], float] = time.monotonic) -> None:
        if faiss is None:
            raise ImportError('HighConcurrencySemanticCache needs faiss: pip install "codigo-sintetico[faiss]"')
        if max_size <= 0 or search_k <= 0 or rebuild_growth <= 1.0 or not 0 < tombstone_ratio < 1:
            raise ValueError("invalid cache parameters")
        self.embedder = embedder or HashingEmbedder()
        self.dim = int(self.embedder.dim)
        self.max_size, self.threshold, self.ttl = max_size, similarity_threshold, ttl_seconds
        self._nlist_cfg, self._nprobe_cfg, self.ivf_min_vectors = nlist, nprobe, ivf_min_vectors
        self.search_k, self.rebuild_growth, self.tombstone_ratio, self._clock = search_k, rebuild_growth, tombstone_ratio, clock
        self._rw = _RWLock()
        self._meta = threading.Lock()                               # LRU order only; never held while waiting for _rw
        self._stats = _Stats()
        self._entries: Dict[int, _Entry] = {}
        self._by_key: Dict[str, int] = {}
        self._lru: "OrderedDict[int, None]" = OrderedDict()
        self._tombstones: Set[int] = set()
        self._next_id = 0
        self._trained_at = 0
        self.rebuilds = self.evictions = 0
        self._index = self._build_index([], np.empty((0, self.dim), dtype=np.float32))

    # ------------------------------------------------------------------------------------------------ index
    def _build_index(self, ids: List[int], vecs: np.ndarray) -> Any:
        n = len(ids)
        if n >= self.ivf_min_vectors:
            nlist = self._nlist_cfg or max(4, int(math.sqrt(n) / 2))
            nlist = max(1, min(nlist, n))
            quantizer = faiss.IndexFlatIP(self.dim)
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vecs)
            index.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))
            index.nprobe = self._nprobe_cfg or max(1, min(nlist, max(8, nlist // 8)))
            self._trained_at = n
            return index
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        if n:
            index.add_with_ids(vecs, np.asarray(ids, dtype=np.int64))
        self._trained_at = 0
        return index

    def _rebuild_locked(self) -> None:
        ids = list(self._entries)
        vecs = np.stack([self._entries[i].vec for i in ids]).astype(np.float32) if ids else np.empty((0, self.dim), dtype=np.float32)
        self._index = self._build_index(ids, vecs)
        self._tombstones.clear()
        self.rebuilds += 1

    def rebuild(self) -> None:
        """Force a rebuild (also compacts tombstones). Blocks readers while it runs."""
        with self._rw.write():
            self._rebuild_locked()

    def _maintain_locked(self) -> None:
        live, dead = len(self._entries), len(self._tombstones)
        is_ivf = self._trained_at > 0
        if dead and dead >= self.tombstone_ratio * (live + dead):
            self._rebuild_locked()
        elif not is_ivf and live >= self.ivf_min_vectors:
            self._rebuild_locked()                                  # small -> big: switch from exact to IVF
        elif is_ivf and live >= self._trained_at * self.rebuild_growth:
            self._rebuild_locked()                                  # centroids are stale after the data doubled

    # ------------------------------------------------------------------------------------------------ eviction
    def _drop_locked(self, entry_id: int) -> None:
        entry = self._entries.pop(entry_id, None)
        if entry is None:
            return
        self._by_key.pop(entry.key, None)
        self._tombstones.add(entry_id)
        with self._meta:
            self._lru.pop(entry_id, None)

    def purge_expired(self) -> int:
        """Remove every expired entry now; returns how many. (``get`` already ignores expired entries.)"""
        now = self._clock()
        with self._rw.write():
            dead = [i for i, e in self._entries.items() if e.expires_at is not None and e.expires_at <= now]
            for i in dead:
                self._drop_locked(i)
            if dead:
                self._maintain_locked()
            return len(dead)

    # ---------------------------------------------------------------------------------------------- public API
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hits(self) -> int: return self._stats.hits
    @property
    def misses(self) -> int: return self._stats.misses
    @property
    def hit_rate(self) -> float: return self._stats.hit_rate

    def stats(self) -> Dict[str, object]:
        with self._rw.read():
            ivf = self._trained_at > 0
            return {"size": len(self._entries), "tombstones": len(self._tombstones), "index": "ivf" if ivf else "flat",
                    "nlist": int(self._index.nlist) if ivf else None, "nprobe": int(self._index.nprobe) if ivf else None,
                    "hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 4),
                    "rebuilds": self.rebuilds, "evictions": self.evictions}

    def get(self, query: str) -> Optional[str]:
        found = self.get_with_score(query)
        return found[0] if found else None

    def get_with_score(self, query: str) -> Optional[Tuple[str, float]]:
        key = _normalise_text(query)
        q = _unit(self.embedder.embed(key))                         # embedding happens OUTSIDE every lock
        now = self._clock()
        hit_id, score, response, expired_seen = None, 0.0, "", None
        with self._rw.read():
            cand = self._by_key.get(key)
            if cand is not None:
                hit_id, score = cand, 1.0
            elif self._entries and float(np.dot(q, q)) > 0:
                total = max(1, self._index.ntotal)
                k = min(self.search_k, total)
                while True:
                    sims, ids = self._index.search(q.reshape(1, -1), k)
                    stale = False
                    for s, i in zip(sims[0], ids[0], strict=True):
                        if i < 0:
                            continue
                        if int(i) in self._entries:
                            hit_id, score = int(i), float(s)
                            break
                        stale = True                                # an evicted id that is still physically in the index
                    # Only widen the search when stale ids hid the answer; otherwise the nearest live entry is already known.
                    if hit_id is not None or not stale or k >= total:
                        break
                    k = min(k * 4, total)
            if hit_id is not None:
                e = self._entries[hit_id]
                if e.expires_at is not None and e.expires_at <= now:
                    expired_seen, hit_id = hit_id, None
                elif score >= self.threshold:
                    response = e.response
                else:
                    hit_id = None
        if expired_seen is not None:                                # mutate only under the write lock, after releasing the read lock
            with self._rw.write():
                self._drop_locked(expired_seen)
        if hit_id is None:
            self._stats.record(False)
            return None
        with self._meta:
            if hit_id in self._lru:
                self._lru.move_to_end(hit_id)
        self._stats.record(True)
        return response, min(score, 1.0)

    def set(self, query: str, response: str) -> None:
        key = _normalise_text(query)
        vec = _unit(self.embedder.embed(key))
        if vec.shape[0] != self.dim:
            raise ValueError(f"embedder returned dim {vec.shape[0]}, expected {self.dim}")
        expires = self._clock() + self.ttl if self.ttl is not None else None
        with self._rw.write():
            existing = self._by_key.get(key)
            if existing is not None:
                e = self._entries[existing]
                e.response, e.expires_at = response, expires
                with self._meta:
                    self._lru.move_to_end(existing)
                return
            new_id, self._next_id = self._next_id, self._next_id + 1
            self._entries[new_id] = _Entry(key, response, vec, expires)
            self._by_key[key] = new_id
            with self._meta:
                self._lru[new_id] = None
            self._index.add_with_ids(vec.reshape(1, -1), np.asarray([new_id], dtype=np.int64))
            while len(self._entries) > self.max_size:
                with self._meta:
                    oldest = next(iter(self._lru))
                self._drop_locked(oldest)
                self.evictions += 1
            self._maintain_locked()

    def clear(self) -> None:
        with self._rw.write():
            self._entries.clear(); self._by_key.clear(); self._tombstones.clear()
            with self._meta:
                self._lru.clear()
            self._index = self._build_index([], np.empty((0, self.dim), dtype=np.float32))
