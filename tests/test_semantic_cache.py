import os, subprocess, sys, threading, time
import numpy as np
import pytest
faiss = pytest.importorskip("faiss")
from hypothesis import HealthCheck, given, settings, strategies as st
from sintetico.semantic_cache import EmbeddingSemanticCache, HashingEmbedder, HighConcurrencySemanticCache, _RWLock


class Table:
    """Embedder with hand-placed vectors, so similarities are known exactly."""
    def __init__(self, dim=8): self.dim, self.v = dim, {}
    def put(self, text, vec): self.v[text.lower()] = np.asarray(vec, dtype=np.float32)
    def embed(self, text): return self.v.get(text.lower(), np.zeros(self.dim, dtype=np.float32))


def unit(*xs, dim=8):
    v = np.zeros(dim, dtype=np.float32); v[: len(xs)] = xs
    return v / np.linalg.norm(v)


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


# ------------------------------------------------------------------------------------------ embedder
def test_hashing_embedder_is_unit_length_deterministic_and_rewards_shared_words():
    e = HashingEmbedder(128)
    a, b, c = e.embed("How do I reset my password"), e.embed("how do i RESET my password?"), e.embed("quarterly revenue forecast")
    assert abs(np.linalg.norm(a) - 1) < 1e-5 and float(a @ b) > 0.99 and float(a @ c) < 0.4
    assert not e.embed("").any() and not e.embed("!!! ???").any()
    with pytest.raises(ValueError):
        HashingEmbedder(4)


def test_hashing_embedder_is_identical_across_processes_despite_hash_randomisation():
    code = "from sintetico.semantic_cache import HashingEmbedder as H;print(H(64).embed('same text every time').round(4).tolist())"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": s}).stdout for s in ("1", "2", "random")}
    assert len(outs) == 1 and outs.pop().startswith("[")


# -------------------------------------------------------------------------------------- linear baseline
@pytest.mark.parametrize("vectorized", [True, False])
def test_linear_cache_exact_near_and_far_queries(vectorized):
    c = EmbeddingSemanticCache(max_size=3, similarity_threshold=0.8, vectorized=vectorized)
    c.set("How do I reset my password?", "answer-1")
    assert c.get("how do i   reset my password?") == "answer-1"                    # normalised exact
    assert c.get("How do I reset my password please") == "answer-1"                # near duplicate
    assert c.get("quarterly revenue forecast") is None
    assert (c.hits, c.misses, c.hit_rate) == (2, 1, pytest.approx(2 / 3))
    for i in range(3):
        c.set(f"filler topic number {i} words", f"f{i}")
    assert len(c) == 3 and c.get("How do I reset my password?") is None             # oldest evicted
    with pytest.raises(ValueError):
        EmbeddingSemanticCache(max_size=0)


# ------------------------------------------------------------------------------------------------ FAISS
def make(max_size=100, threshold=0.9, **kw):
    t = Table(); return t, HighConcurrencySemanticCache(t, max_size=max_size, similarity_threshold=threshold, **kw)


def test_hit_miss_and_threshold_is_inclusive():
    t, c = make(threshold=0.9)
    t.put("stored", unit(1, 0)); t.put("close", unit(0.95, 0.3122)); t.put("far", unit(0.5, 0.866))
    c.set("stored", "S")
    assert c.get("stored") == "S" and c.get("close") == "S" and c.get("far") is None and c.get("unknown words") is None
    resp, score = c.get_with_score("close")
    assert resp == "S" and 0.94 < score < 0.96
    t2, edge = make(threshold=score)                                                    # threshold exactly at the similarity: still a hit
    t2.put("stored", unit(1, 0)); t2.put("close", unit(0.95, 0.3122)); edge.set("stored", "S")
    assert edge.get("close") == "S"
    t3, above = make(threshold=score + 1e-4)
    t3.put("stored", unit(1, 0)); t3.put("close", unit(0.95, 0.3122)); above.set("stored", "S")
    assert above.get("close") is None


def test_best_match_wins_when_several_are_above_threshold():
    t, c = make(threshold=0.5)
    va, vb = unit(1, 0), unit(0.9, 0.436)
    for name, v in (("a", va), ("b", vb), ("near_b", unit(0.8, 0.6)), ("near_a", unit(0.99, 0.1))):
        t.put(name, v)
    c.set("a", "A"); c.set("b", "B")
    for q, expected in (("near_b", "B"), ("near_a", "A")):
        sims = {"A": float(va @ t.v[q]), "B": float(vb @ t.v[q])}
        assert max(sims, key=sims.get) == expected and abs(sims["A"] - sims["B"]) > 0.01      # the expectation is derived, and unambiguous
        assert c.get(q) == expected


def test_lru_eviction_and_tombstoned_ids_never_leak_into_results():
    t, c = make(max_size=3, threshold=0.99, tombstone_ratio=0.99)                       # keep tombstones in the index on purpose
    for i in range(5):
        t.put(f"k{i}", unit(*[1.0 if j == i else 0.0 for j in range(5)]))
    for i in range(3):
        c.set(f"k{i}", f"v{i}")
    assert c.get("k0") == "v0"                                                           # refresh k0 -> k1 is now the oldest
    c.set("k3", "v3")
    assert len(c) == 3 and c.get("k1") is None and c.get("k0") == "v0" and c.get("k2") == "v2" and c.get("k3") == "v3"
    st_ = c.stats()
    assert st_["evictions"] == 1 and st_["tombstones"] == 1                              # k1 is still physically in the FAISS index...
    assert c.get("k1") is None                                                           # ...and must never be returned
    c.set("k4", "v4"); c.set("k1", "again")
    assert c.get("k1") == "again" and len(c) == 3


def test_stale_neighbours_in_the_index_do_not_hide_a_valid_entry():
    t, c = make(max_size=1, threshold=0.9, search_k=2, tombstone_ratio=0.99)
    t.put("query", unit(1, 0))
    for i in range(6):                                                                    # six evicted entries, all nearer to the query than the survivor
        t.put(f"old{i}", unit(1, 0.01 * (i + 1))); c.set(f"old{i}", "stale")
    t.put("live", unit(0.95, 0.3122)); c.set("live", "LIVE")
    assert c.stats()["tombstones"] >= 6 and len(c) == 1
    assert c.get("query") == "LIVE"                                                       # with a fixed k=2 this would be a false miss


def test_updating_an_existing_key_changes_the_answer_without_growing():
    t, c = make(); c.set("same", "one"); c.set("SAME ", "two")
    assert len(c) == 1 and c.get("same") == "two" and c.stats()["tombstones"] == 0


def test_ttl_uses_the_clock_expires_and_set_refreshes():
    clock, t = Clock(), Table()
    c = HighConcurrencySemanticCache(t, ttl_seconds=10, similarity_threshold=0.9, clock=clock)
    t.put("a", unit(1, 0)); t.put("b", unit(0, 1))
    c.set("a", "A"); clock.t = 5; c.set("b", "B")
    clock.t = 9.9; assert c.get("a") == "A"
    clock.t = 10.0; assert c.get("a") is None and len(c) == 1                           # expired and physically dropped
    clock.t = 14.9; c.set("b", "B2"); clock.t = 24.0; assert c.get("b") == "B2"        # refreshed at 14.9 -> lives until 24.9
    clock.t = 25.0
    assert c.purge_expired() == 1 and len(c) == 0 and c.purge_expired() == 0


def test_empty_cache_empty_query_and_wrong_dimension():
    t, c = make()
    assert c.get("anything") is None and c.get("") is None
    c.set("", "empty-query-response")
    assert c.get("") == "empty-query-response" and c.get("  ") == "empty-query-response"   # exact key still works; zero vectors never match by similarity
    t.put("x", np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError):
        c.set("x", "bad dim")
    for bad in (dict(max_size=0), dict(search_k=0), dict(rebuild_growth=1.0), dict(tombstone_ratio=1.0)):
        with pytest.raises(ValueError):
            HighConcurrencySemanticCache(Table(), **bad)


def test_clear_resets_everything():
    t, c = make(); t.put("a", unit(1, 0)); c.set("a", "A"); c.clear()
    assert len(c) == 0 and c.get("a") is None and c.stats()["tombstones"] == 0


# ---------------------------------------------------------------------------- IVF: scale, recall, rebuilds
class Vectors:
    def __init__(self, dim, n, seed=0):
        rng = np.random.default_rng(seed)
        m = rng.normal(size=(n, dim)).astype(np.float32)
        self.m, self.dim = m / np.linalg.norm(m, axis=1, keepdims=True), dim
    def embed(self, text): return self.m[int(text.split(":")[1])]


def test_switches_to_ivf_and_keeps_good_recall_against_exact_search():
    n, dim = 6000, 48
    emb = Vectors(dim, n + 400)
    c = HighConcurrencySemanticCache(emb, max_size=n, similarity_threshold=-1.0, ivf_min_vectors=2000, search_k=1)
    for i in range(n):
        c.set(f"v:{i}", str(i))
    s = c.stats()
    assert s["index"] == "ivf" and s["rebuilds"] >= 1 and 0 < s["nprobe"] <= s["nlist"]
    rng = np.random.default_rng(1)
    exact_hits = 0
    for j in range(300):
        target = int(rng.integers(0, n))
        noisy = emb.m[target] + 0.05 * rng.normal(size=dim).astype(np.float32)
        emb.m[n + j % 400] = noisy / np.linalg.norm(noisy)
        got = c.get(f"v:{n + j % 400}")
        exact = int(np.argmax(emb.m[:n] @ emb.m[n + j % 400]))
        exact_hits += got == str(exact)
    assert exact_hits / 300 >= 0.9, exact_hits                                           # approximate by design; documented recall floor
    full = HighConcurrencySemanticCache(emb, max_size=n, similarity_threshold=-1.0, ivf_min_vectors=2000, search_k=1, nprobe=10_000)
    for i in range(n):
        full.set(f"v:{i}", str(i))
    assert all(full.get(f"v:{n + j % 400}") == str(int(np.argmax(emb.m[:n] @ emb.m[n + j % 400]))) for j in range(0, 300, 7))   # nprobe=all lists -> exact


def test_rebuilds_when_data_doubles_and_when_tombstones_pile_up():
    emb = Vectors(32, 5000)
    c = HighConcurrencySemanticCache(emb, max_size=10_000, similarity_threshold=0.0, ivf_min_vectors=500, rebuild_growth=2.0)
    for i in range(1100):
        c.set(f"v:{i}", "x")
    r1 = c.stats()["rebuilds"]
    for i in range(1100, 2100):
        c.set(f"v:{i}", "x")
    assert c.stats()["rebuilds"] > r1                                                     # 1000 -> 2000 live: retrained
    small = HighConcurrencySemanticCache(emb, max_size=100, tombstone_ratio=0.25)
    for i in range(1000):
        small.set(f"v:{i}", "x")
    st_ = small.stats()
    assert st_["size"] == 100 and st_["evictions"] == 900 and st_["tombstones"] <= 0.25 * (100 + st_["tombstones"]) + 1 and st_["rebuilds"] >= 5


# ------------------------------------------------------------------------------------------ concurrency
def test_rwlock_allows_parallel_readers_blocks_writers_and_prefers_writers():
    lock, both_inside, log = _RWLock(), threading.Barrier(2, timeout=5), []
    def reader():
        with lock.read():
            both_inside.wait()                                                           # only passes if BOTH readers hold the lock at once
    ts = [threading.Thread(target=reader) for _ in range(2)]; [x.start() for x in ts]; [x.join(5) for x in ts]
    assert not any(x.is_alive() for x in ts) and not both_inside.broken
    gate, writer_waiting = threading.Event(), threading.Event()
    def holder():
        with lock.read():
            gate.wait(5)
    def writer():
        writer_waiting.set()
        with lock.write():
            log.append("writer")
    def late_reader():
        with lock.read():
            log.append("late-reader")
    h = threading.Thread(target=holder); h.start(); time.sleep(0.05)
    w = threading.Thread(target=writer); w.start(); writer_waiting.wait(1); time.sleep(0.1)
    r = threading.Thread(target=late_reader); r.start(); time.sleep(0.1)
    assert log == []                                                                     # writer waits for the holder; the late reader waits behind the writer
    gate.set(); [x.join(5) for x in (h, w, r)]
    assert log == ["writer", "late-reader"]


def test_eight_threads_mixed_reads_and_writes_stay_consistent():
    emb = HashingEmbedder(64)
    c = HighConcurrencySemanticCache(emb, max_size=200, similarity_threshold=0.95, ivf_min_vectors=150, tombstone_ratio=0.2)
    errors, counts = [], {"gets": 0}
    lock = threading.Lock()
    def worker(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(600):
                k = f"topic {int(rng.integers(0, 400))} about subject {int(rng.integers(0, 5))}"
                if rng.random() < 0.4:
                    c.set(k, f"resp::{k}")
                else:
                    got = c.get(k)
                    with lock: counts["gets"] += 1
                    if got is not None:
                        assert got.startswith("resp::")
                        if got != f"resp::{k}":
                            assert float(emb.embed(got[6:]) @ emb.embed(k)) >= 0.94      # a near hit must really be near
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
    ts = [threading.Thread(target=worker, args=(s,)) for s in range(8)]; [t.start() for t in ts]; [t.join(60) for t in ts]
    assert not errors and not any(t.is_alive() for t in ts)
    s = c.stats()
    assert s["size"] <= 200 and s["hits"] + s["misses"] == counts["gets"] and s["rebuilds"] >= 1


# ------------------------------------------------- property: FAISS (exact mode) == linear exact baseline
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(entries=st.lists(st.tuples(st.lists(st.sampled_from("alpha beta gamma delta epsilon zeta eta theta".split()), min_size=1, max_size=4),
                                  st.integers(0, 99)), min_size=1, max_size=12),
       queries=st.lists(st.lists(st.sampled_from("alpha beta gamma delta epsilon zeta eta theta iota".split()), min_size=1, max_size=4), min_size=1, max_size=8))
def test_flat_faiss_agrees_with_the_linear_baseline_on_hit_and_score(entries, queries):
    emb = HashingEmbedder(64)
    lin = EmbeddingSemanticCache(emb, max_size=1000, similarity_threshold=0.6)
    fast = HighConcurrencySemanticCache(emb, max_size=1000, similarity_threshold=0.6, ivf_min_vectors=10**9, search_k=16)
    for words, r in entries:
        text = " ".join(words)
        lin.set(text, str(r)); fast.set(text, str(r))
    for words in queries:
        a, b = lin.get_with_score(" ".join(words)), fast.get_with_score(" ".join(words))
        assert (a is None) == (b is None)
        if a:
            assert abs(a[1] - b[1]) < 1e-4


def test_linear_cache_swap_delete_keeps_rows_consistent_under_heavy_eviction():
    emb = Vectors(16, 400)
    c = EmbeddingSemanticCache(emb, max_size=50, similarity_threshold=0.999)
    for i in range(400):
        c.set(f"v:{i}", str(i))
        if i % 7 == 0 and i >= 50:
            assert c.get(f"v:{i - 3}") == str(i - 3)                       # recent entries are still found with their own answer
    assert len(c) == 50 and sorted(c._row.values()) == list(range(50))
    assert all(c._keys[j] == k for k, j in c._row.items()) and set(c._order) == set(c._keys) == set(c._resp)
    assert c.get("v:0") is None and c.get("v:399") == "399"


# ------------------------------------------------------------------ pinned after mutation testing
def test_embedder_validation_zero_vectors_and_bigram_weight():
    with pytest.raises(ValueError): HashingEmbedder(7)
    assert HashingEmbedder(8).dim == 8 and HashingEmbedder().dim == 256 and HashingEmbedder().bigram_weight == 0.5
    e = HashingEmbedder(64)
    z = e.embed("")
    assert not z.any() and not np.isnan(z).any() and not np.isnan(e.embed("...")).any()
    bi, uni = HashingEmbedder(64, bigram_weight=0.5), HashingEmbedder(64, bigram_weight=0.0)
    assert not np.allclose(bi.embed("new york city"), bi.embed("city york new"))                  # word order matters when bigrams count
    assert np.allclose(uni.embed("new york city"), uni.embed("city york new"))                    # ... and not when they do not
    c = EmbeddingSemanticCache(e); c.set("", "empty"); assert c.get("") == "empty" and c.get("anything else") is None and c.hit_rate == 0.5


def test_defaults_are_the_documented_ones():
    lin = EmbeddingSemanticCache()
    assert (lin.max_size, lin.threshold, lin.vectorized, lin.hit_rate, lin.hits, lin.misses) == (10_000, 0.92, True, 0.0, 0, 0)
    fast = HighConcurrencySemanticCache()
    assert (fast.max_size, fast.threshold, fast.ttl, fast.search_k, fast.rebuild_growth, fast.tombstone_ratio, fast.ivf_min_vectors) == (100_000, 0.92, None, 8, 2.0, 0.25, 2048)
    assert fast.hit_rate == 0.0 and fast.stats()["index"] == "flat" and fast.stats()["nlist"] is None and len(fast) == 0


def test_negative_similarity_is_a_valid_match_when_the_threshold_allows_it():
    for vectorized in (True, False):
        t = Table(); t.put("a", unit(1, 0)); t.put("opposite", unit(-1, 0.1))                    # cosine = -1/sqrt(1.01) = -0.995
        c = EmbeddingSemanticCache(t, similarity_threshold=-0.999, vectorized=vectorized); c.set("a", "A")
        resp, score = c.get_with_score("opposite")
        assert resp == "A" and score == pytest.approx(-0.995, abs=1e-3)                           # (not -1.0 exactly: float32 rounding makes that a flaky edge)
        strict = EmbeddingSemanticCache(t, similarity_threshold=-0.99, vectorized=vectorized); strict.set("a", "A")
        assert strict.get("opposite") is None
    assert EmbeddingSemanticCache(t).get("x") is None and EmbeddingSemanticCache(t, vectorized=False).get("x") is None      # empty cache, both paths


def test_scores_never_exceed_one_despite_float32_rounding_and_exact_hits_score_exactly_one():
    rng = np.random.default_rng(3); t = Table(dim=64)
    lin, fast = EmbeddingSemanticCache(t, similarity_threshold=0.5), HighConcurrencySemanticCache(t, similarity_threshold=0.5)
    for i in range(300):
        v = rng.normal(size=64).astype(np.float32); v /= np.linalg.norm(v)
        t.put(f"stored {i}", v); t.put(f"alias {i}", v)                                   # same vector under another text
        lin.set(f"stored {i}", "r"); fast.set(f"stored {i}", "r")
        assert lin.get_with_score(f"alias {i}")[1] <= 1.0 and fast.get_with_score(f"alias {i}")[1] <= 1.0
    assert fast.get_with_score("stored 7")[1] == 1.0 and lin.get_with_score("stored 7")[1] == 1.0     # exact-text path


def test_a_writer_really_excludes_readers():
    lock, order = _RWLock(), []
    with lock.write():
        r = threading.Thread(target=lambda: (lock.read().__enter__(), order.append("read")))
        r.start(); time.sleep(0.15)
        assert order == []                                                                # the reader is blocked while the writer holds the lock
        order.append("writer-done")
    r.join(3)
    assert order == ["writer-done", "read"]


def test_ivf_geometry_defaults_and_overrides_and_purge_boundary():
    emb = Vectors(16, 2600)
    auto = HighConcurrencySemanticCache(emb, max_size=3000, similarity_threshold=0.0, ivf_min_vectors=2500)
    for i in range(2500): auto.set(f"v:{i}", "x")
    s = auto.stats()
    assert s["index"] == "ivf" and s["nlist"] == 25 and s["nprobe"] == 8                       # nlist = sqrt(2500)/2, nprobe = max(8, nlist//8)
    manual = HighConcurrencySemanticCache(emb, max_size=3000, ivf_min_vectors=2500, nlist=10, nprobe=3)
    for i in range(2500): manual.set(f"v:{i}", "x")
    assert (manual.stats()["nlist"], manual.stats()["nprobe"]) == (10, 3)
    clock, t = Clock(), Table(); c = HighConcurrencySemanticCache(t, ttl_seconds=10, clock=clock); t.put("a", unit(1, 0)); c.set("a", "A")
    clock.t = 9.999; assert c.purge_expired() == 0
    clock.t = 10.0; assert c.purge_expired() == 1                                                # expiry is inclusive: expires_at <= now
