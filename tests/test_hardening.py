"""Regression tests for the concurrency / resource-exhaustion defects found in review."""
import http.client, multiprocessing as mp, threading, time
from concurrent.futures import ThreadPoolExecutor
import pytest
from sintetico.adapters import OutputAdapter, OutputParseError
from sintetico.audit import AgentAuditLogger
from sintetico.devops.compliance import EvidenceStore, StageResult, build_evidence_package
from sintetico.devops.gitops import serve


def _writer(path, tag, n):
    log = AgentAuditLogger("k", path=path)
    for i in range(n):
        log.log(f"proc-{tag}", "tick", i=i)


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="needs the POSIX fork start method (and flock): not available on Windows")
def test_audit_chain_survives_concurrent_processes(tmp_path):
    """Before the flock fix, each process extended its own in-memory tail and the file forked."""
    path = tmp_path / "audit.jsonl"
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_writer, args=(path, t, 25)) for t in range(4)]
    [p.start() for p in procs]; [p.join(30) for p in procs]
    assert all(p.exitcode == 0 for p in procs)
    reloaded = AgentAuditLogger("k", path=path)            # raises if the chain forked
    assert len(reloaded.events) == 100 and reloaded.verify() == (True, None)
    assert sorted({e.actor for e in reloaded.events}) == [f"proc-{i}" for i in range(4)]


def test_truncating_the_tail_is_only_detectable_with_an_external_anchor():
    log = AgentAuditLogger("k")
    for i in range(5):
        log.log("a", "x", i=i)
    anchor = log.head                                       # stored somewhere else (ticket, other system)
    del log.events[3:]                                       # attacker drops the last two events
    assert log.verify() == (True, None)                      # the chain alone cannot see it ...
    assert log.verify(expected_head=anchor) == (False, 3)    # ... the anchor can


def test_evidence_write_is_atomic_under_races(tmp_path):
    store = EvidenceStore(tmp_path)
    pkg = build_evidence_package("racy", [StageResult("sast-analysis", "PASS", "t")], "k")
    def attempt(_):
        try:
            store.write(pkg); return "ok"
        except FileExistsError:
            return "exists"
    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(attempt, range(16)))
    assert results.count("ok") == 1 and results.count("exists") == 15
    assert len(store.query()) == 1                           # exactly one index line


def test_webhook_rejects_oversized_bodies():
    srv = serve(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for length in ("50000000", "-5"):
            conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
            try:
                conn.putrequest("POST", "/validate"); conn.putheader("Content-Length", length); conn.endheaders()
                assert conn.getresponse().status == 413
            finally:
                conn.close()
    finally:
        srv.shutdown(); srv.server_close()


@pytest.mark.parametrize("payload", ["{" * 1_000_000, '{"a":' * 200_000, "[" * 500_000 + "]" * 10, "{" * 5000 + "x" * 5000])
def test_json_extraction_is_bounded_on_pathological_input(payload):
    t0 = time.perf_counter()
    with pytest.raises(OutputParseError):
        OutputAdapter.extract_json(payload)
    assert time.perf_counter() - t0 < 3.0
