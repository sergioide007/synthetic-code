"""
sintetico — a library of patterns for production multi-agent systems.

Reference implementation of the book "Synthetic Code: The Art of
Orchestrating AI Agents". It is organized by domain (not by chapter) so
that each module has a single responsibility and can be understood,
tested and versioned on its own:

    sintetico.budget          Token budget (local / Redis)
    sintetico.providers       Provider-agnostic LLM contract
    sintetico.real_providers  Real providers (Anthropic, OpenAI, local)
    sintetico.model_registry  Catalog of models and prices
    sintetico.orchestration   Routing, response cache, async gateway, swarms
    sintetico.reasoning       ReAct, Tree of Thoughts, Reflexion
    sintetico.resilience      Circuit breakers, harness, audit trail
    sintetico.quality         Self-cleaning code, TrustScore, ShadowAgent
    sintetico.governance      RACI, progressive autonomy, overrides, escalation
    sintetico.debugging       Debugging agent with timeout guards
    sintetico.security        Threat model, scanner, MCP host
    sintetico.economics       TCO and ROI

Quick example:

    >>> from sintetico import ModelRouter, TokenBudget
    >>> router = ModelRouter()
    >>> router.select_model("What is the status of my order?")
    'haiku'

Working with real APIs:

    >>> from sintetico import create_provider, LLMRequest
    >>> provider = create_provider("anthropic", default_model="haiku")
    >>> response = provider.complete(LLMRequest(
    ...     system_prompt="Be concise.",
    ...     messages=[{"role": "user", "content": "Hello"}],
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

# real_providers needs optional SDKs (anthropic/openai/requests); it is
# imported lazily through create_provider so that `import sintetico` keeps
# working in environments where those packages are not installed.


def create_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory for real providers. See `sintetico.real_providers.create_provider`.

    It is re-exported here, instead of being imported in the import block
    above, so that `import sintetico` does not fail in an environment
    without `anthropic`/`openai`/`requests` installed: those SDKs are only
    imported when a provider that needs them is actually requested.
    """
    from .real_providers import create_provider as _create_provider

    return _create_provider(provider_type, **kwargs)

# --- Added in 2.1.0: book Parts II-VI reference implementations -------------
from .adapters import CapabilityProfile, OutputAdapter, OutputParseError, PromptAdapter
from .audit import AgentAuditLogger, AuditEvent, AuditKeyMissingError
from .autonomy import (ActivitySpec, DomainRiskProfile, DynamicThresholdRACIGate, GateDecision, Verifiability,
                       compute_max_allowed_autonomy, threshold_for_error_cost)
from .delivery import (AgentArchitectureReviewer, DeployGate, DeployMetrics, DeployPhase, PipelineDecision,
                       ProductionPipeline, SwarmOrchestrator)
from .docs import ADR, ADRArchive, ADRGenerator, DocumentGenerator
from .evals import CostOptimizerWithEvals, EvalCase, EvalReport, EvalSet
from .gateway import (AIGateway, AIGatewayWithProjects, BudgetExceededError, ContextBudgetExceeded, ContextBudgetManager,
                      GatewayResult, ProjectPolicy)
from .guards import DetectionGuard, GuardDecision
from .review import BugFinding, CodeReviewAgent, ConfidenceEscalationReviewer, HumanInLoopReviewer
from .security import MCPHost
from . import devops

# --- Added in 2.2.0: resilient agents, FAISS cache, distributed execution ----
from .distributed import (AgentWorker, BusError, DistributedAIGateway, GatewayOverloaded, InMemoryBus, KafkaBus, RetryableError, TaskTimeout,
                          react_handler)
try:  # optional extra `structured` (pydantic): importing sintetico must work without it
    from .reasoning_resilient import ReActAgentResilient, TreeOfThoughtsResilient
    from .structured import AgentStep, Hypothesis, HypothesisBatch, StructuredOutputError, complete_structured
except ImportError:  # pragma: no cover
    pass
try:  # optional extra `faiss` (numpy; faiss-cpu for the FAISS class)
    from .semantic_cache import EmbeddingSemanticCache, HashingEmbedder, HighConcurrencySemanticCache
except ImportError:  # pragma: no cover
    pass

__version__ = "2.2.2"

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
    "AgentWorker",
    "BusError",
    "DistributedAIGateway",
    "GatewayOverloaded",
    "InMemoryBus",
    "KafkaBus",
    "RetryableError",
    "TaskTimeout",
    "react_handler",
    "ReActAgentResilient",
    "TreeOfThoughtsResilient",
    "AgentStep",
    "Hypothesis",
    "HypothesisBatch",
    "StructuredOutputError",
    "complete_structured",
    "EmbeddingSemanticCache",
    "HashingEmbedder",
    "HighConcurrencySemanticCache",
    "CapabilityProfile",
    "OutputAdapter",
    "OutputParseError",
    "PromptAdapter",
    "AgentAuditLogger",
    "AuditEvent",
    "AuditKeyMissingError",
    "ActivitySpec",
    "DomainRiskProfile",
    "DynamicThresholdRACIGate",
    "GateDecision",
    "Verifiability",
    "compute_max_allowed_autonomy",
    "threshold_for_error_cost",
    "AgentArchitectureReviewer",
    "DeployGate",
    "DeployMetrics",
    "DeployPhase",
    "PipelineDecision",
    "ProductionPipeline",
    "SwarmOrchestrator",
    "ADR",
    "ADRArchive",
    "ADRGenerator",
    "DocumentGenerator",
    "CostOptimizerWithEvals",
    "EvalCase",
    "EvalReport",
    "EvalSet",
    "AIGateway",
    "AIGatewayWithProjects",
    "BudgetExceededError",
    "ContextBudgetExceeded",
    "ContextBudgetManager",
    "GatewayResult",
    "ProjectPolicy",
    "DetectionGuard",
    "GuardDecision",
    "BugFinding",
    "CodeReviewAgent",
    "ConfidenceEscalationReviewer",
    "HumanInLoopReviewer",
    "MCPHost",
    "devops",
]

# Optional extras may be missing: only advertise what actually imported, so `from sintetico import *` always works.
__all__ = [n for n in __all__ if n in globals()]
