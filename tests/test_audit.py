import json
import pytest
from sintetico.audit import AgentAuditLogger, AuditKeyMissingError


def test_chain_verifies_and_detects_tampering():
    log = AgentAuditLogger("k")
    for i in range(5):
        log.log("agent", "step", n=i, cost_usd=0.01, tokens=10)
    assert log.verify() == (True, None)
    log.events[2].details["n"] = 999            # rewrite history
    assert log.verify() == (False, 2)


def test_deleting_an_event_breaks_the_chain():
    log = AgentAuditLogger("k")
    for i in range(4):
        log.log("a", "x", i=i)
    del log.events[1]
    ok, bad = log.verify()
    assert not ok and bad == 1


def test_wrong_key_fails_verification():
    a = AgentAuditLogger("key-a")
    a.log("a", "x")
    b = AgentAuditLogger("key-b")
    b.events = list(a.events)
    assert b.verify()[0] is False


def test_requires_key(monkeypatch):
    monkeypatch.delenv("SINTETICO_AUDIT_KEY", raising=False)
    with pytest.raises(AuditKeyMissingError):
        AgentAuditLogger()


def test_key_from_environment(monkeypatch):
    monkeypatch.setenv("SINTETICO_AUDIT_KEY", "from-env")
    assert AgentAuditLogger().log("a", "x").own_hash


def test_persistence_roundtrip_and_continuation(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AgentAuditLogger("k", path=p)
    a.log("a", "one")
    a.log("a", "two")
    b = AgentAuditLogger("k", path=p)           # reload + continue the chain
    assert len(b.events) == 2
    b.log("a", "three")
    assert AgentAuditLogger("k", path=p).verify() == (True, None)
    assert len(p.read_text().strip().splitlines()) == 3


def test_tampered_file_refuses_to_load(tmp_path):
    p = tmp_path / "audit.jsonl"
    a = AgentAuditLogger("k", path=p)
    a.log("a", "one", cost_usd=1)
    a.log("a", "two", cost_usd=1)
    lines = p.read_text().splitlines()
    d = json.loads(lines[0]); d["details"]["cost_usd"] = 0
    p.write_text(json.dumps(d) + "\n" + lines[1] + "\n")
    with pytest.raises(ValueError):
        AgentAuditLogger("k", path=p)


def test_export_report_aggregates():
    log = AgentAuditLogger("k")
    log.log("g", "llm_call", cost_usd=0.5, tokens=100)
    log.log("g", "llm_call", cost_usd=0.25, tokens=50)
    log.log("human", "human_approval")
    r = log.export_report()
    assert r["total_cost_usd"] == 0.75 and r["total_tokens"] == 150 and r["human_approvals"] == 1 and r["chain_valid"]
