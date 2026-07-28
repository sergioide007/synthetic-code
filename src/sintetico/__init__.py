"""
sintetico — Librería de patrones para sistemas multiagente de producción.

Implementación de referencia del libro "Código Sintético: El Arte de
Orquestar Agentes de IA". Organizada por dominio (no por capítulo) para
que cada módulo tenga una única responsabilidad y pueda entenderse,
probarse y versionarse de forma independiente:

    sintetico.budget          Presupuesto de tokens (local / Redis)
    sintetico.providers       Contrato agnóstico de proveedor de LLM
    sintetico.real_providers  Proveedores reales (Anthropic, OpenAI, local)
    sintetico.model_registry  Catálogo de modelos y precios
    sintetico.orchestration   Routing, caché semántica, gateway async, enjambres
    sintetico.reasoning       ReAct, Tree of Thoughts, Reflexion
    sintetico.resilience      Circuit breakers, harness, auditoría
    sintetico.quality         Autolimpieza de código, TrustScore, ShadowAgent
    sintetico.governance      RACI, autonomía progresiva, overrides, escalado
    sintetico.debugging       Agente de debugging con guardas de timeout
    sintetico.security        Modelo de amenazas, scanner, host MCP
    sintetico.economics       TCO y ROI

Ejemplo rápido:

    >>> from sintetico import ModelRouter, TokenBudget
    >>> router = ModelRouter()
    >>> router.select_model("¿Cómo estás?")
    'haiku'

Para trabajar con APIs reales:

    >>> from sintetico import create_provider, LLMRequest
    >>> provider = create_provider("anthropic", default_model="haiku")
    >>> response = provider.complete(LLMRequest(
    ...     system_prompt="Eres conciso.",
    ...     messages=[{"role": "user", "content": "Hola"}],
    ... ))
"""

from .budget import (
    BudgetBackend,
    LocalMemoryBudgetBackend,
    RedisBudgetBackend,
    RedisUnavailableError,
    TokenBudget,
)
from .providers import (
    LLMRequest,
    LLMResponse,
    LLMProvider,
    MockProvider,
    BackendAgnosticAgent,
    AllProvidersFailedError,
)
from .model_registry import ModelConfig, MODEL_REGISTRY, get_model_config, resolve_model_id
from .orchestration import (
    TaskStatus,
    AgentTask,
    AsyncAIGateway,
    CycleDetector,
    SwarmTokenOrchestrator,
    ModelRouter,
    SemanticCache,
    RAGConfig,
    RAGOptimizer,
)
from .reasoning import Thought, Observation, ReActAgent, ThoughtNode, TreeOfThoughts, ReflexionAgent
from .resilience import CircuitBreaker, AgentCircuitBreaker, AuditLogger, AgentResult, AgentHarness
from .quality import (
    CodeSmellType,
    CodeSmell,
    AgentAnalyzer,
    AgentRefactor,
    AgentTests,
    SelfCleaningCodeLoop,
    TrustScore,
    ShadowVerdict,
    ShadowDecision,
    ShadowAgent,
    AgentDebt,
    DebtTracker,
    TrustLevel,
    TrustScoreResult,
    TrustScoreCalculator,
)
from .governance import (
    RACILevel,
    RACIGate,
    ProductionInvariants,
    TrueCostTracker,
    AutonomyLevel,
    AutonomyGate,
    OverrideReason,
    OverrideRequest,
    EmergencyOverrideGateway,
    Ticket,
    EscalationDecision,
    EscalationPolicy,
    EscalationHandler,
    DecisionIrreversibility,
    DecisionCategory,
    DecisionRequest,
    HumanDecisionGate,
    DatabaseMigrationGate,
)
from .debugging import DebugPhase, DebugProgress, ProductionDebugAgent, DebugTimeoutGuard
from .security import (
    ThreatCategory,
    MitigationEffectiveness,
    Threat,
    Mitigation,
    THREAT_MATRIX,
    SecurityScanner,
    PermissionLevel,
    MCPTool,
    MPCHost,
    SchemaValidationUnavailableError,
)
from .economics import compute_tco, calculate_roas

# real_providers requiere SDKs opcionales (anthropic/openai/requests); se
# importa de forma perezosa vía create_provider para no romper `import
# sintetico` en entornos donde esos paquetes no están instalados.


def create_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factoría de proveedores reales. Ver `sintetico.real_providers.create_provider`.

    Se re-expone aquí, en vez de importarla directamente en el bloque de
    imports de arriba, para que `import sintetico` no falle en un entorno
    sin `anthropic`/`openai`/`requests` instalados: esos SDKs sólo se
    importan cuando de verdad se pide un proveedor que los necesita.
    """
    from .real_providers import create_provider as _create_provider

    return _create_provider(provider_type, **kwargs)


__version__ = "2.0.0"

__all__ = [
    "BudgetBackend",
    "LocalMemoryBudgetBackend",
    "RedisBudgetBackend",
    "RedisUnavailableError",
    "TokenBudget",
    "LLMRequest",
    "LLMResponse",
    "LLMProvider",
    "MockProvider",
    "BackendAgnosticAgent",
    "AllProvidersFailedError",
    "create_provider",
    "ModelConfig",
    "MODEL_REGISTRY",
    "get_model_config",
    "resolve_model_id",
    "TaskStatus",
    "AgentTask",
    "AsyncAIGateway",
    "CycleDetector",
    "SwarmTokenOrchestrator",
    "ModelRouter",
    "SemanticCache",
    "RAGConfig",
    "RAGOptimizer",
    "Thought",
    "Observation",
    "ReActAgent",
    "ThoughtNode",
    "TreeOfThoughts",
    "ReflexionAgent",
    "CircuitBreaker",
    "AgentCircuitBreaker",
    "AuditLogger",
    "AgentResult",
    "AgentHarness",
    "CodeSmellType",
    "CodeSmell",
    "AgentAnalyzer",
    "AgentRefactor",
    "AgentTests",
    "SelfCleaningCodeLoop",
    "TrustScore",
    "ShadowVerdict",
    "ShadowDecision",
    "ShadowAgent",
    "AgentDebt",
    "DebtTracker",
    "TrustLevel",
    "TrustScoreResult",
    "TrustScoreCalculator",
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
    "DebugPhase",
    "DebugProgress",
    "ProductionDebugAgent",
    "DebugTimeoutGuard",
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
    "compute_tco",
    "calculate_roas",
    "__version__",
]
