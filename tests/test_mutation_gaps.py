"""Behaviours that mutation testing showed were NOT pinned down by the first version of the tests."""
import dataclasses
import pytest
from sintetico.audit import AgentAuditLogger, AuditCorruptionError
from sintetico.autonomy import (ActivitySpec, DomainRiskProfile, DynamicThresholdRACIGate, Verifiability, compute_max_allowed_autonomy)
from sintetico.devops.migration import MigrationAdvisor, Phase, PHASE_CRITERIA
from sintetico.governance import RACILevel


# ---- autonomy: exact flags, exact boundary, caps, immutability, defaults
def _gate(role, cost=5):
    return DynamicThresholdRACIGate({"a": ActivitySpec(role, "owner", cost)})


def test_gate_decisions_carry_exact_flags():
    d = _gate(RACILevel.ACCOUNTABLE).authorize("a", 1.0)
    assert (d.allowed, d.requires_human_approval, d.owner) == (False, True, "owner")
    d = _gate(RACILevel.INFORMED).authorize("a", 1.0)
    assert (d.allowed, d.requires_human_approval) == (False, False)
    d = _gate(RACILevel.CONSULTED).authorize("a", 1.0)
    assert (d.allowed, d.requires_human_approval) == (False, False)
    d = _gate(RACILevel.RESPONSIBLE).authorize("a", 1.0)
    assert (d.allowed, d.requires_human_approval, d.threshold) == (True, False, 0.2)
    d = _gate(RACILevel.RESPONSIBLE).authorize("a", 0.1)
    assert (d.allowed, d.requires_human_approval) == (False, True)


def test_trust_exactly_at_threshold_is_enough():
    assert _gate(RACILevel.RESPONSIBLE, 5).authorize("a", 0.2).allowed          # 0.2 >= 0.2
    assert not _gate(RACILevel.RESPONSIBLE, 5).authorize("a", 0.1999).allowed


def test_automated_autonomy_is_capped_at_one_and_defaults_hold():
    assert compute_max_allowed_autonomy(DomainRiskProfile("d", Verifiability.AUTOMATED, 0.95)) == 1.0
    assert compute_max_allowed_autonomy(DomainRiskProfile("d", Verifiability.AUTOMATED, 0.5)) == 0.6
    assert compute_max_allowed_autonomy(DomainRiskProfile("d", Verifiability.QUICK_MANUAL)) == 0.6    # default threshold
    assert compute_max_allowed_autonomy(DomainRiskProfile("d", Verifiability.SLOW_MANUAL, 0.05)) == 0.05   # never above the threshold


def test_risk_profile_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DomainRiskProfile("d", Verifiability.AUTOMATED).name = "x"


# ---- audit: canonical form survives reload, anchor semantics, crash safety
def test_persisted_log_with_unsorted_detail_keys_still_verifies_after_reload(tmp_path):
    p = tmp_path / "deep" / "nested" / "a.jsonl"                                  # parent dirs must be created
    log = AgentAuditLogger("k", path=p)
    log.log("a", "x", zebra=1, apple={"z": 1, "a": [3, 2, 1]}, mango="m")           # keys deliberately not alphabetical
    assert AgentAuditLogger("k", path=p).verify() == (True, None)
    AgentAuditLogger("k", path=p).log("b", "y")                                    # second open, dir already exists
    assert len(AgentAuditLogger("k", path=p).events) == 2


def test_head_is_the_newest_hash_and_matching_anchor_passes():
    log = AgentAuditLogger("k")
    assert log.head == "genesis" and log.verify(expected_head="genesis") == (True, None)
    for i in range(3):
        log.log("a", "x", i=i)
    assert log.head == log.events[-1].own_hash != log.events[0].own_hash
    assert log.verify(expected_head=log.head) == (True, None)
    assert log.verify(expected_head="0" * 64) == (False, 3)


def test_partial_tail_after_a_crash_makes_appends_fail_closed(tmp_path):
    p = tmp_path / "a.jsonl"
    AgentAuditLogger("k", path=p).log("a", "one")
    with p.open("ab") as fh:
        fh.write(b'{"event_id": "half-written-when-the-proc')                      # crash mid-write: no trailing newline
    reopened = AgentAuditLogger("k", path=p)                                       # loading ignores the incomplete line
    assert len(reopened.events) == 1 and reopened.verify() == (True, None)
    with pytest.raises(AuditCorruptionError):
        reopened.log("a", "two")                                                   # appending would glue lines together
    assert p.read_bytes().count(b"\n") == 1                                        # nothing was written


# ---- migration: every documented threshold, both sides
@pytest.mark.parametrize("phase,days,dlq,clean,ok", [
    (Phase.SHADOW, 7, 100, 0, True), (Phase.SHADOW, 6.9, 100, 0, False), (Phase.SHADOW, 7, 101, 0, False),
    (Phase.DUAL_WRITE, 14, 10, 0, True), (Phase.DUAL_WRITE, 13.9, 10, 0, False), (Phase.DUAL_WRITE, 14, 11, 0, False),
    (Phase.VERIFY, 14, 0, 3, True), (Phase.VERIFY, 13.9, 0, 3, False), (Phase.VERIFY, 14, 1, 3, False), (Phase.VERIFY, 14, 0, 2, False),
])
def test_phase_thresholds_are_exact(phase, days, dlq, clean, ok):
    rec = MigrationAdvisor().evaluate(phase, days, dlq, clean, 0.0, True)
    assert (rec.action != "HOLD") is ok


def test_only_the_cutover_proposal_requires_a_human():
    adv = MigrationAdvisor()
    assert adv.evaluate(Phase.SHADOW, 99, 0).requires_human is False
    assert adv.evaluate(Phase.SHADOW, 0, 0).requires_human is False and adv.evaluate(Phase.SHADOW, 0, 0).action == "HOLD"
    assert adv.evaluate(Phase.CUTOVER, 0, 0).requires_human is False
    assert adv.evaluate(Phase.VERIFY, 99, 0, 9, 0.0, True).requires_human is True
    assert [p.value for p in Phase] == [1, 2, 3, 4] and set(PHASE_CRITERIA) == {Phase.SHADOW, Phase.DUAL_WRITE, Phase.VERIFY}
    assert adv.evaluate(Phase.VERIFY, 99, 0, 9, 0.0).action == "HOLD"               # rollback_tested defaults to False
