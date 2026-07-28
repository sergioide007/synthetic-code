import pytest

from sintetico import (
    AutonomyGate,
    AutonomyLevel,
    DatabaseMigrationGate,
    EmergencyOverrideGateway,
    OverrideReason,
    RACIGate,
    RACILevel,
)


class TestRACIGate:
    def test_requires_exactly_one_accountable(self):
        with pytest.raises(ValueError):
            RACIGate("deploy", {"alice": RACILevel.RESPONSIBLE})

    def test_agent_cannot_be_accountable(self):
        with pytest.raises(ValueError):
            RACIGate("deploy", {"agent-x": RACILevel.ACCOUNTABLE})

    def test_valid_assignment_succeeds(self):
        gate = RACIGate("deploy", {"alice": RACILevel.ACCOUNTABLE, "agent-x": RACILevel.RESPONSIBLE})
        assert gate.get_accountable() == "alice"


class TestEmergencyOverrideGateway:
    def test_approved_override_is_recorded_in_history(self):
        """Regresión: la v1 nunca escribía en `override_history`, así que
        el rastro de auditoría de overrides quedaba siempre vacío."""
        gateway = EmergencyOverrideGateway()
        req_id = gateway.request_override(
            "run-1",
            "TrustScore bajo",
            OverrideReason.FALSE_POSITIVE,
            "El fallo es un falso positivo confirmado manualmente",
            "dev-lead",
        )
        gateway.approve_override(req_id, "cto")
        assert len(gateway.override_history) == 1
        assert gateway.override_history[0].approver == "cto"
        assert req_id not in gateway.pending_requests

    def test_short_justification_is_rejected(self):
        gateway = EmergencyOverrideGateway()
        with pytest.raises(ValueError):
            gateway.request_override("run-1", "x", OverrideReason.KNOWN_ISSUE, "corto", "dev")


class TestAutonomyGate:
    def test_starts_at_workflow_only(self):
        gate = AutonomyGate("agent-1")
        assert gate.current_level == AutonomyLevel.WORKFLOW_ONLY

    def test_promotes_after_weeks_and_pass_rate(self):
        gate = AutonomyGate("agent-1")
        gate.eval_set_pass_rate = 0.97
        for _ in range(2):
            gate.advance_week()
        assert gate.current_level == AutonomyLevel.ROUTING

    def test_does_not_promote_with_insufficient_pass_rate(self):
        gate = AutonomyGate("agent-1")
        gate.eval_set_pass_rate = 0.50
        for _ in range(12):
            gate.advance_week()
        assert gate.current_level == AutonomyLevel.WORKFLOW_ONLY

    def test_critical_incident_demotes_immediately(self):
        gate = AutonomyGate("agent-1")
        gate.eval_set_pass_rate = 0.99
        for _ in range(12):
            gate.advance_week()
        assert gate.current_level == AutonomyLevel.FULL_AUTONOMY
        gate.record_incident("critical")
        assert gate.current_level == AutonomyLevel.TOOL_AGENT


class TestDatabaseMigrationGate:
    def test_rejects_excessive_downtime(self):
        gate = DatabaseMigrationGate()
        result = gate.propose_migration(
            "postgres",
            "cockroach",
            data_volume_gb=10,
            estimated_downtime_minutes=200,
            justification="Escalar horizontalmente",
            rollback_plan="x" * 60,
            proposed_by="alice",
        )
        assert result["status"] == "rejected"

    def test_valid_migration_requires_two_approvals(self):
        gate = DatabaseMigrationGate()
        result = gate.propose_migration(
            "postgres",
            "cockroach",
            data_volume_gb=10,
            estimated_downtime_minutes=30,
            justification="Escalar horizontalmente",
            rollback_plan="x" * 60,
            proposed_by="alice",
        )
        assert result["status"] == "pending_approvals"
        migration_id = result["migration_id"]

        first = gate.approve_migration(migration_id, "principal-engineer")
        assert first["status"] == "approved_partially"

        second = gate.approve_migration(migration_id, "cto")
        assert second["status"] == "approved"

    def test_unauthorized_approver_is_rejected(self):
        gate = DatabaseMigrationGate()
        result = gate.propose_migration(
            "postgres",
            "cockroach",
            data_volume_gb=1,
            estimated_downtime_minutes=10,
            justification="test",
            rollback_plan="x" * 60,
            proposed_by="alice",
        )
        outcome = gate.approve_migration(result["migration_id"], "random-person")
        assert outcome["status"] == "error"
