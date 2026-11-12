"""
sintetico.security — threat model, heuristic prompt-injection scanner and
MCP host with schema validation (Chapter 17).

Important notice: `SecurityScanner` is a pattern-based heuristic filter,
meant as a cheap, low-latency *first line* (defense in depth), not as the
only defense. An attacker who knows the patterns can evade them easily.
In production it must be combined with a classifier (another LLM or a
dedicated model) that evaluates intent, and with controls at the level of
tool permissions (see `MCPHost`). That is what the `secondary_classifier`
parameter of `SecurityScanner` is for: it is the extension point designed
to plug in that real classifier without rewriting the scanner.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ThreatCategory",
    "MitigationEffectiveness",
    "Threat",
    "Mitigation",
    "THREAT_MATRIX",
    "SecurityScanner",
    "PermissionLevel",
    "MCPTool",
    "MPCHost",
    "SchemaValidationUnavailableError",
]


class ThreatCategory(Enum):
    PROMPT_INJECTION = "prompt_injection"
    TOOL_ABUSE = "tool_abuse"
    CONTEXT_POISONING = "context_poisoning"
    DATA_LEAKAGE = "data_leakage"
    AUTHORIZATION_BYPASS = "authorization_bypass"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


class MitigationEffectiveness(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Threat:
    category: ThreatCategory
    description: str
    attack_vector: str
    impact: str
    likelihood: str


@dataclass(frozen=True)
class Mitigation:
    threat: Threat
    technique: str
    effectiveness: MitigationEffectiveness
    implementation_cost: str
    coverage: str


THREAT_MATRIX: List[Threat] = [
    Threat(
        ThreatCategory.PROMPT_INJECTION,
        "Inyección de instrucciones",
        "Instrucciones ocultas que piden ignorar reglas anteriores",
        "critical",
        "high",
    ),
    Threat(
        ThreatCategory.TOOL_ABUSE,
        "Uso malicioso de herramientas",
        "Ejecutar comandos o llamadas no autorizadas vía una tool",
        "critical",
        "medium",
    ),
    Threat(
        ThreatCategory.CONTEXT_POISONING,
        "Envenenamiento de contexto RAG",
        "Documentos indexados con instrucciones ocultas para el LLM",
        "high",
        "medium",
    ),
    Threat(
        ThreatCategory.DATA_LEAKAGE,
        "Filtración de datos sensibles",
        "La respuesta del modelo contiene PII o secretos",
        "critical",
        "high",
    ),
    Threat(
        ThreatCategory.AUTHORIZATION_BYPASS,
        "Bypass de autorización",
        "El agente actúa fuera de los permisos concedidos al usuario",
        "critical",
        "low",
    ),
    Threat(
        ThreatCategory.RESOURCE_EXHAUSTION,
        "DoS por consumo de tokens",
        "Peticiones costosas repetidas que agotan presupuesto/cuota",
        "high",
        "medium",
    ),
]


class SecurityScanner:
    """Cheap heuristics that detect prompt injection and risks in tool
    definitions. See the module notice: it does not replace a semantic
    classifier or permission controls.

    `secondary_classifier`, if provided, is a second opinion consulted
    *only* when the heuristics find nothing — never to "overrule" a regex
    finding (a heuristic that detects something must not be silenced by a
    classifier that says otherwise; when in doubt, the more conservative
    signal wins). Expected signature: `Callable[[str], bool]`, returning
    `True` if the text is malicious according to the classifier. It is the
    place to plug in a second LLM that evaluates semantic intent, or a
    dedicated prompt-injection detection model — the regex itself, by
    design, will never be that component."""

    _SYSTEM_PROMPT_BYPASS_PATTERNS = (
        r"\bignora\b.{0,20}\b(instrucci|regla)",
        r"\bdisregard\b.{0,20}\b(instruction|rule)",
    )
    _USER_INPUT_PATTERNS = (
        (r"ignora.{0,20}instrucciones", "prompt_injection"),
        (r"ignore.{0,20}(previous|above).{0,20}instructions", "prompt_injection"),
        (r"\bsystem prompt\b", "exfiltracion_de_contexto"),
        (r"\bact(úa|ua)? como (dan|jailbreak)\b", "jailbreak_roleplay"),
        (r"\brepite (todo|literalmente) lo anterior\b", "exfiltracion_de_contexto"),
    )

    def __init__(self, secondary_classifier: Optional[Callable[[str], bool]] = None):
        self.secondary_classifier = secondary_classifier

    def scan_system_prompt(self, system_prompt: str) -> List[Dict]:
        findings = []
        lowered = system_prompt.lower()
        for pattern in self._SYSTEM_PROMPT_BYPASS_PATTERNS:
            if re.search(pattern, lowered):
                findings.append({"severity": "high", "vulnerability": "Instrucciones de bypass detectadas"})
                break
        return findings

    def scan_user_input(self, user_input: str) -> Dict:
        findings = []
        for pattern, category in self._USER_INPUT_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                findings.append({"type": category, "severity": "high"})

        if not findings and self.secondary_classifier is not None:
            try:
                if self.secondary_classifier(user_input):
                    findings.append({"type": "flagged_by_secondary_classifier", "severity": "medium"})
            except Exception:  # noqa: BLE001
                # A failure of the secondary classifier must not bring the scan
                # down: it degrades to the regex result already computed.
                logger.exception("El secondary_classifier de SecurityScanner falló; se ignora su veredicto")

        return {"is_malicious": len(findings) > 0, "findings": findings}

    def scan_tool_definition(self, tool: Dict) -> List[Dict]:
        risks = []
        description = tool.get("description", "")
        if len(description) < 50:
            risks.append(
                {
                    "risk": "description_insufficient",
                    "severity": "medium",
                    "detail": "Una descripción corta dificulta que el modelo entienda cuándo NO usar la herramienta",
                }
            )
        if tool.get("permission") == "admin" and not tool.get("requires_approval", False):
            risks.append({"risk": "admin_tool_without_approval", "severity": "critical"})
        return risks


class PermissionLevel(Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class MCPTool:
    name: str
    description: str
    permission: PermissionLevel
    requires_approval: bool
    audit_required: bool
    input_schema: dict


class SchemaValidationUnavailableError(RuntimeError):
    pass


class MCPHost:
    """MCP host with deterministic schema validation (JSON Schema) before
    any tool is executed, and mandatory human approval for the tools
    marked as requiring it."""

    def __init__(self):
        self.registered_tools: Dict[str, MCPTool] = {}
        self.active_session: Optional[dict] = None
        try:
            import jsonschema
        except ImportError as exc:
            raise SchemaValidationUnavailableError(
                "Instala jsonschema para validación MCP: pip install jsonschema"
            ) from exc
        self._validator = jsonschema

    def register_tool(self, tool: MCPTool) -> None:
        self.registered_tools[tool.name] = tool

    def set_session_context(self, context: dict) -> None:
        self.active_session = {"context": context}

    def invoke_tool(self, tool_name: str, input_data: dict, approval_granted: bool = False) -> Dict:
        tool = self.registered_tools.get(tool_name)
        if not tool:
            return {"error": "Tool not found"}

        try:
            self._validator.validate(instance=input_data, schema=tool.input_schema)
        except self._validator.exceptions.ValidationError as exc:
            return {"error": f"ValidationError: {exc.message}"}

        if tool.requires_approval and not approval_granted:
            return {
                "status": "pending_approval",
                "message": f"'{tool_name}' requiere aprobación humana antes de ejecutarse",
            }

        return {"result": f"Ejecutado {tool_name} con {input_data}"}


# Backwards-compatible alias for the original (misspelled) name.
MPCHost = MCPHost
