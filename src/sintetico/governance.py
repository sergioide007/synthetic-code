"""
sintetico.governance — RACI, invariantes de producción, autonomía
progresiva, overrides de emergencia, escalado a humanos y aprobación de
decisiones irreversibles (Capítulos 7, 9, 12, 15 y 16).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from .budget import TokenBudget

__all__ = [
    "RACILevel",
    "RACIGate",
    "ProductionInvariants",
    "TrueCostTracker",
    "AutonomyLevel",
    "AutonomyGate",
    "OverrideReason",
    "OverrideRequest",
    "EmergencyOverrideGateway",
    "Ticket",
    "EscalationDecision",
    "EscalationPolicy",
    "EscalationHandler",
    "DecisionIrreversibility",
    "DecisionCategory",
    "DecisionRequest",
    "HumanDecisionGate",
    "DatabaseMigrationGate",
]


# ═══════════════════════════════════════════════════════════════════
# RACI Gate
# ═══════════════════════════════════════════════════════════════════


class RACILevel(Enum):
    RESPONSIBLE = "R"
    ACCOUNTABLE = "A"
    CONSULTED = "C"
    INFORMED = "I"


class RACIGate:
    """Obliga a que toda actividad tenga exactamente un responsable último
    (Accountable) humano; un agente jamás puede ostentar esa posición."""

    def __init__(self, activity: str, assignments: Dict[str, RACILevel]):
        self.activity = activity
        self.assignments = assignments
        self._validate()

    def _validate(self) -> None:
        accountable = [actor for actor, level in self.assignments.items() if level == RACILevel.ACCOUNTABLE]
        if len(accountable) != 1:
            raise ValueError(f"Debe haber exactamente 1 Accountable para '{self.activity}'")
        if any("agent" in actor.lower() for actor in accountable):
            raise ValueError("Un agente nunca puede ser Accountable")

    def can_proceed(self, pending_approval: bool = False) -> bool:
        return not pending_approval

    def get_accountable(self) -> Optional[str]:
        for actor, level in self.assignments.items():
            if level == RACILevel.ACCOUNTABLE:
                return actor
        return None


# ═══════════════════════════════════════════════════════════════════
# Invariantes de producción y coste total real (TCO)
# ═══════════════════════════════════════════════════════════════════


class ProductionInvariants:
    """Invariantes que deben cumplirse *antes* de cada llamada a un agente
    en producción: presupuesto disponible y confianza mínima acorde al
    nivel de autonomía solicitado."""

    def __init__(self, audit, budget: TokenBudget):
        self.audit = audit
        self.budget = budget

    def verify_before_call(self, estimated_cost: float, trust_score: float, autonomy_level: int) -> bool:
        if estimated_cost > self.budget.remaining:
            return False
        if autonomy_level > 2 and trust_score < 60:
            return False
        return True


class TrueCostTracker:
    """Acumula el coste total real (tokens + tiempo humano de verificación)
    y detecta cuándo la verificación humana está dominando el coste, señal
    de que el agente no es lo bastante fiable para el nivel de autonomía
    actual."""

    def __init__(self, hourly_rate_usd: float):
        if hourly_rate_usd <= 0:
            raise ValueError("hourly_rate_usd debe ser positivo")
        self.hourly_rate = hourly_rate_usd
        self.token_cost_usd = 0.0
        self.verify_minutes_total = 0.0
        self.volume = 0

    def record_agent_call(self, tokens_usd: float, verification_minutes: float) -> None:
        self.token_cost_usd += tokens_usd
        self.verify_minutes_total += verification_minutes
        self.volume += 1

    def get_tco(self) -> Dict:
        verify_cost = (self.verify_minutes_total / 60) * self.hourly_rate
        return {
            "token_cost_usd": round(self.token_cost_usd, 2),
            "verification_cost_usd": round(verify_cost, 2),
            "total_tco_usd": round(self.token_cost_usd + verify_cost, 2),
            "volume": self.volume,
            "avg_verify_time_min": round(self.verify_minutes_total / self.volume, 2) if self.volume else 0,
        }

    def detect_debt_spike(self, ratio_threshold: float = 3.0) -> bool:
        """True si el coste de verificación humana supera en `ratio_threshold`
        veces el coste de tokens: síntoma de que el agente genera más
        trabajo de revisión del que ahorra en generación."""
        tco = self.get_tco()
        return tco["verification_cost_usd"] > (ratio_threshold * tco["token_cost_usd"] + 0.01)


# ═══════════════════════════════════════════════════════════════════
# Autonomía progresiva
# ═══════════════════════════════════════════════════════════════════


class AutonomyLevel(Enum):
    WORKFLOW_ONLY = 0
    ROUTING = 1
    TOOL_AGENT = 2
    LIMITED_AUTONOMY = 3
    FULL_AUTONOMY = 4


@dataclass
class _AutonomyMilestone:
    weeks_required: int
    min_pass_rate: float
    level: AutonomyLevel
    requires_zero_incidents: bool = False


class AutonomyGate:
    """Modela la promoción gradual de autonomía de un agente en producción,
    condicionada a tiempo en producción + tasa de éxito en el eval set,
    con degradación inmediata ante un incidente crítico.
    """

    _MILESTONES = [
        _AutonomyMilestone(2, 0.95, AutonomyLevel.ROUTING),
        _AutonomyMilestone(4, 0.90, AutonomyLevel.TOOL_AGENT),
        _AutonomyMilestone(8, 0.85, AutonomyLevel.LIMITED_AUTONOMY, requires_zero_incidents=True),
        _AutonomyMilestone(12, 0.0, AutonomyLevel.FULL_AUTONOMY, requires_zero_incidents=True),
    ]

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.current_level = AutonomyLevel.WORKFLOW_ONLY
        self.weeks_in_production = 0
        self.incidents_critical = 0
        self.eval_set_pass_rate = 0.0

    def can_execute(self, requested_level: AutonomyLevel) -> bool:
        return requested_level.value <= self.current_level.value

    def advance_week(self) -> None:
        """Avanza una semana e intenta promocionar de nivel.

        Los hitos se evalúan en orden estricto (nunca se puede saltar de
        WORKFLOW_ONLY a FULL_AUTONOMY sin pasar por los niveles
        intermedios), aunque hayan transcurrido suficientes semanas para
        el hito final: cada promoción exige cumplir el pass_rate y la
        ausencia de incidentes del *siguiente* nivel, no sólo del último.

        Nota de diseño (bug corregido): una primera versión comprobaba
        cada hito de forma independiente. Como el hito de FULL_AUTONOMY
        no exige una `min_pass_rate` adicional (ya viene filtrada por los
        hitos previos), un agente con una tasa de éxito baja pero mucho
        tiempo en producción podía saltar directamente a autonomía
        completa sin haber demostrado nunca un pass_rate aceptable — el
        tipo exacto de fallo de gobernanza que este gate existe para
        prevenir. Ahora la promoción es estrictamente secuencial.
        """
        self.weeks_in_production += 1
        while True:
            next_level_value = self.current_level.value + 1
            milestone = next((m for m in self._MILESTONES if m.level.value == next_level_value), None)
            if milestone is None:
                break  # ya en el nivel máximo
            if self.weeks_in_production < milestone.weeks_required:
                break
            if self.eval_set_pass_rate < milestone.min_pass_rate:
                break
            if milestone.requires_zero_incidents and self.incidents_critical > 0:
                break
            self.current_level = milestone.level

    def record_incident(self, severity: str) -> None:
        if severity == "critical":
            self.incidents_critical += 1
            self.current_level = AutonomyLevel.TOOL_AGENT


# ═══════════════════════════════════════════════════════════════════
# Emergency Override Gateway
# ═══════════════════════════════════════════════════════════════════


class OverrideReason(Enum):
    FALSE_POSITIVE = "false_positive"
    BUSINESS_PRIORITY = "business_priority"
    KNOWN_ISSUE = "known_issue"
    TEST_MISCONFIG = "test_misconfig"


@dataclass
class OverrideRequest:
    pipeline_run_id: str
    blocker_reason: str
    override_reason: OverrideReason
    justification: str
    requestor: str
    approver: Optional[str] = None
    risk_acknowledgment: bool = False
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None


class EmergencyOverrideGateway:
    """Bypass auditado para falsos positivos de calidad/seguridad.

    Todo override aprobado queda registrado en `override_history` de forma
    permanente e inmutable desde fuera (no se expone un método para
    borrarlo), porque es el rastro de auditoría que un compliance officer
    necesitará poder revisar.
    """

    def __init__(self):
        self.pending_requests: Dict[str, OverrideRequest] = {}
        self._override_history: List[OverrideRequest] = []
        self._id_counter = 0

    @property
    def override_history(self) -> List[OverrideRequest]:
        return list(self._override_history)

    def request_override(
        self,
        pipeline_run_id: str,
        blocker_reason: str,
        override_reason: OverrideReason,
        justification: str,
        requestor: str,
    ) -> str:
        if len(justification.strip()) < 10:
            raise ValueError("La justificación debe ser sustancial (>=10 caracteres)")
        self._id_counter += 1
        req_id = f"OVR-{self._id_counter:04d}"
        self.pending_requests[req_id] = OverrideRequest(
            pipeline_run_id, blocker_reason, override_reason, justification, requestor
        )
        return req_id

    def approve_override(self, request_id: str, approver: str) -> Dict:
        req = self.pending_requests.get(request_id)
        if not req:
            return {"status": "error", "message": "Request not found"}
        req.approver = approver
        req.risk_acknowledgment = True
        req.approved_at = datetime.now(timezone.utc)
        self._override_history.append(req)
        del self.pending_requests[request_id]
        return {"status": "approved", "request_id": request_id}


# ═══════════════════════════════════════════════════════════════════
# Escalado a humanos
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Ticket:
    user_query: str
    priority: str = "medium"


@dataclass
class EscalationDecision:
    should_escalate: bool
    reason: str
    confidence: float
    ticket: Ticket


class EscalationPolicy:
    DEFAULT_CRITICAL_KEYWORDS = ("cancelar", "demanda", "abogado", "ceo")

    def __init__(
        self, confidence_threshold: float = 0.7, max_retries: int = 2, critical_keywords: Optional[List[str]] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.critical_keywords = tuple(k.lower() for k in (critical_keywords or self.DEFAULT_CRITICAL_KEYWORDS))

    def should_escalate(
        self, ticket: Ticket, agent_confidence: float, retry_count: int, conversation_history: Optional[List] = None
    ) -> EscalationDecision:
        query_lower = ticket.user_query.lower()
        matched = [kw for kw in self.critical_keywords if kw in query_lower]
        if matched:
            return EscalationDecision(True, f"Palabra crítica detectada: {matched[0]}", 0.0, ticket)
        if agent_confidence < self.confidence_threshold:
            return EscalationDecision(
                True, f"Confianza insuficiente ({agent_confidence:.2f})", agent_confidence, ticket
            )
        if retry_count >= self.max_retries:
            return EscalationDecision(True, "Reintentos agotados", agent_confidence, ticket)
        return EscalationDecision(False, "OK", agent_confidence, ticket)


class EscalationHandler:
    def __init__(self):
        self.tickets_created: List[Dict] = []

    def escalate(self, decision: EscalationDecision, conversation: Optional[List] = None) -> Dict:
        ticket_id = f"ESC-{len(self.tickets_created) + 1:04d}"
        self.tickets_created.append(
            {
                "ticket_id": ticket_id,
                "ticket": decision.ticket,
                "reason": decision.reason,
                "conversation_length": len(conversation) if conversation else 0,
            }
        )
        return {"message": "Escalado a humano", "ticket_id": ticket_id}


# ═══════════════════════════════════════════════════════════════════
# Aprobación de decisiones irreversibles
# ═══════════════════════════════════════════════════════════════════


class DecisionIrreversibility(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionCategory(Enum):
    INFRASTRUCTURE = "infrastructure"
    DATA_MODEL = "data_model"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    FINANCIAL = "financial"


@dataclass
class DecisionRequest:
    category: DecisionCategory
    proposal: str
    alternatives: List[str]
    estimated_cost: float
    reversibility: DecisionIrreversibility
    proposer_agent: Optional[str] = None


class HumanDecisionGate:
    APPROVAL_MATRIX = {
        (DecisionCategory.INFRASTRUCTURE, DecisionIrreversibility.HIGH): "cto",
        (DecisionCategory.DATA_MODEL, DecisionIrreversibility.HIGH): "cto",
        (DecisionCategory.SECURITY, DecisionIrreversibility.HIGH): "ciso",
        (DecisionCategory.COMPLIANCE, DecisionIrreversibility.CRITICAL): "compliance+legal",
        (DecisionCategory.FINANCIAL, DecisionIrreversibility.CRITICAL): "cfo+board",
    }
    MAX_AUTONOMOUS_COST = 50_000

    def can_agent_decision(self, request: DecisionRequest) -> bool:
        if request.reversibility in (DecisionIrreversibility.HIGH, DecisionIrreversibility.CRITICAL):
            return False
        if request.estimated_cost > self.MAX_AUTONOMOUS_COST:
            return False
        return True

    def route_for_approval(self, request: DecisionRequest) -> str:
        return self.APPROVAL_MATRIX.get((request.category, request.reversibility), "principal-engineer")


class DatabaseMigrationGate:
    """Aprobación de doble control (2 firmas) para migraciones de base de
    datos, con validaciones duras antes de aceptar siquiera la propuesta."""

    APPROVALS_REQUIRED = ("principal-engineer", "cto")
    MAX_DOWNTIME_MINUTES = 120
    MIN_ROLLBACK_PLAN_CHARS = 50
    LARGE_VOLUME_GB_THRESHOLD = 100

    def __init__(self):
        self.pending_migrations: Dict[str, Dict] = {}

    def propose_migration(
        self,
        from_db: str,
        to_db: str,
        data_volume_gb: int,
        estimated_downtime_minutes: int,
        justification: str,
        rollback_plan: str,
        proposed_by: str,
    ) -> Dict:
        validation = self._validate(estimated_downtime_minutes, rollback_plan, data_volume_gb, justification)
        if not validation["valid"]:
            return {"status": "rejected", "reasons": validation["reasons"]}

        raw = f"{from_db}{to_db}{datetime.now(timezone.utc).isoformat()}"
        migration_id = hashlib.sha256(raw.encode()).hexdigest()[:12]
        self.pending_migrations[migration_id] = {
            "id": migration_id,
            "from": from_db,
            "to": to_db,
            "volume_gb": data_volume_gb,
            "downtime_minutes": estimated_downtime_minutes,
            "justification": justification,
            "rollback_plan": rollback_plan,
            "proposed_by": proposed_by,
            "approvals": [],
            "status": "pending",
        }
        return {
            "status": "pending_approvals",
            "migration_id": migration_id,
            "approvals_needed": len(self.APPROVALS_REQUIRED),
        }

    def _validate(self, downtime_minutes: int, rollback_plan: str, volume_gb: int, justification: str) -> Dict:
        reasons = []
        if downtime_minutes > self.MAX_DOWNTIME_MINUTES:
            reasons.append(
                f"Downtime > {self.MAX_DOWNTIME_MINUTES} min no permitido sin ventana de mantenimiento aprobada"
            )
        if len(rollback_plan) < self.MIN_ROLLBACK_PLAN_CHARS:
            reasons.append("Plan de rollback insuficiente (mínimo 50 caracteres)")
        if not justification.strip():
            reasons.append("Falta justificación de negocio")
        if volume_gb > self.LARGE_VOLUME_GB_THRESHOLD and "cto" not in self.APPROVALS_REQUIRED:
            # Nota: con la lista de aprobadores actual esta condición nunca
            # se cumple (el CTO ya es obligatorio). Se deja explícita porque,
            # si en el futuro se reduce APPROVALS_REQUIRED, este umbral debe
            # seguir exigiendo la firma del CTO para migraciones grandes.
            reasons.append(f"Volumen > {self.LARGE_VOLUME_GB_THRESHOLD}GB requiere aprobación explícita del CTO")
        return {"valid": len(reasons) == 0, "reasons": reasons}

    def approve_migration(self, migration_id: str, approver: str) -> Dict:
        migration = self.pending_migrations.get(migration_id)
        if not migration:
            return {"status": "error", "reason": "Not found"}
        if approver not in self.APPROVALS_REQUIRED:
            return {"status": "error", "reason": "No autorizado"}
        if approver in migration["approvals"]:
            return {"status": "already_approved"}
        migration["approvals"].append(approver)
        if len(migration["approvals"]) >= len(self.APPROVALS_REQUIRED):
            migration["status"] = "approved"
            return {"status": "approved"}
        return {"status": "approved_partially", "approvals_received": len(migration["approvals"])}
