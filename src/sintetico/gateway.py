"""Synchronous AI Gateway: one entry point for routing, budgets and auditing (book Chapter 3),
plus a context-window budget manager (Chapter 9)."""

from __future__ import annotations

import enum
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .audit import AgentAuditLogger
from .budget import LocalMemoryBudgetBackend, TokenBudget
from .model_registry import ModelConfig, get_model_config
from .providers import LLMProvider, LLMRequest

__all__ = ["BudgetExceededError", "ProjectPolicy", "GatewayResult", "AIGateway", "AIGatewayWithProjects",
           "ContextBudgetExceeded", "ContextBudgetManager", "estimate_tokens"]

TIER_DEFAULT_ALIAS = {"fast": "haiku", "balanced": "sonnet", "quality": "opus"}
COMPLEXITY_TO_TIER = {"simple": "fast", "medium": "balanced", "complex": "quality"}


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class GatewayResult:
    content: str
    model: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    request_id: str
    team_id: str
    degraded: bool = False
    project_id: Optional[str] = None


class AIGateway:
    """Routes each request to a model tier, enforces per-team budgets, records an audit trail.

    ``provider`` may be a single :class:`LLMProvider` (used for every model) or a mapping
    ``{"anthropic": ..., "openai": ..., "local": ...}`` keyed by the registry's ``provider`` field."""

    def __init__(self, provider: Union[LLMProvider, Mapping[str, LLMProvider]], audit: Optional[AgentAuditLogger] = None,
                 tier_models: Optional[Mapping[str, str]] = None) -> None:
        self._provider = provider
        self.audit = audit
        self.tier_models = dict(TIER_DEFAULT_ALIAS)
        if tier_models:
            self.tier_models.update(tier_models)
        self.budgets: Dict[str, TokenBudget] = {}
        self._backend = LocalMemoryBudgetBackend()
        self._lock = threading.Lock()

    def add_team(self, team_id: str, monthly_budget_usd: float, backend=None) -> TokenBudget:
        self.budgets[team_id] = TokenBudget(monthly_budget_usd, backend=backend or self._backend, team_id=team_id)
        return self.budgets[team_id]

    def select_model(self, complexity: str = "medium", force_model: Optional[str] = None, tier: Optional[str] = None) -> ModelConfig:
        if force_model:
            return get_model_config(force_model)
        tier = tier or COMPLEXITY_TO_TIER.get(complexity)
        if tier is None or tier not in self.tier_models:
            raise ValueError(f"Unknown complexity '{complexity}'. Use simple|medium|complex")
        return get_model_config(self.tier_models[tier])

    def _provider_for(self, cfg: ModelConfig) -> LLMProvider:
        if isinstance(self._provider, Mapping):
            try:
                return self._provider[cfg.provider]
            except KeyError as exc:
                raise KeyError(f"No provider configured for '{cfg.provider}'") from exc
        return self._provider

    @staticmethod
    def cost_usd(cfg: ModelConfig, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * cfg.cost_per_million_input + output_tokens * cfg.cost_per_million_output) / 1_000_000

    def _pre_check(self, team_id: str, cfg: ModelConfig, prompt_tokens: int, max_tokens: int) -> None:
        if team_id not in self.budgets:
            raise KeyError(f"Team '{team_id}' has no budget; call add_team() first")
        worst_case = self.cost_usd(cfg, prompt_tokens, max_tokens)
        if worst_case > self.budgets[team_id].remaining:
            raise BudgetExceededError(f"Team '{team_id}': worst-case cost ${worst_case:.4f} exceeds remaining ${self.budgets[team_id].remaining:.4f}")

    def complete(self, team_id: str, user_message: str, system_prompt: str = "", complexity: str = "medium",
                 model: Optional[str] = None, max_tokens: int = 1024, _degraded: bool = False,
                 _project_id: Optional[str] = None) -> GatewayResult:
        cfg = self.select_model(complexity, force_model=model)
        prompt_tokens = estimate_tokens(system_prompt + user_message)
        self._pre_check(team_id, cfg, prompt_tokens, max_tokens)
        request = LLMRequest(system_prompt=system_prompt, messages=[{"role": "user", "content": user_message}],
                             model=cfg.id, max_tokens=max_tokens)
        resp = self._provider_for(cfg).complete(request)
        in_tok = int(resp.usage.get("input_tokens", prompt_tokens))
        out_tok = int(resp.usage.get("output_tokens", estimate_tokens(resp.content)))
        cost = self.cost_usd(cfg, in_tok, out_tok)
        request_id = str(uuid.uuid4())
        recorded = self.budgets[team_id].record_cost(cost)
        if self.audit:
            self.audit.log(actor=f"gateway:{team_id}", action="llm_call", request_id=request_id, model=cfg.id,
                           tokens=in_tok + out_tok, cost_usd=round(cost, 8), budget_recorded=recorded,
                           degraded=_degraded, project_id=_project_id)
        if not recorded:
            raise BudgetExceededError(f"Team '{team_id}': budget exhausted while recording ${cost:.6f}")
        return GatewayResult(resp.content, cfg.id, cost, in_tok, out_tok, request_id, team_id, _degraded, _project_id)


class ProjectPolicy(enum.Enum):
    HARD_STOP = "hard_stop"  # refuse once the project reaches 95% of its budget
    DEGRADE = "degrade"      # keep serving, but only with the cheapest tier, flagged `degraded`


class AIGatewayWithProjects(AIGateway):
    """Adds per-project budgets with an explicit exhaustion policy (book exercise, Chapter 3)."""

    CUTOFF = 0.95

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._projects: Dict[str, tuple] = {}

    def add_project(self, project_id: str, monthly_budget_usd: float, policy: ProjectPolicy = ProjectPolicy.HARD_STOP) -> None:
        budget = TokenBudget(monthly_budget_usd, backend=LocalMemoryBudgetBackend(), team_id=f"project:{project_id}")
        self._projects[project_id] = (budget, policy)

    def project_spent_fraction(self, project_id: str) -> float:
        budget, _ = self._projects[project_id]
        return budget.spent / budget.monthly_budget

    def complete(self, team_id: str, user_message: str, *args: Any, project_id: Optional[str] = None,  # type: ignore[override]
                 **kwargs: Any) -> GatewayResult:
        # project_id is keyword-only so positional arguments keep the parent's meaning (system_prompt, complexity, ...).
        if len(args) > 4:
            raise TypeError("complete() takes at most 4 positional arguments after user_message")
        kwargs.update(zip(("system_prompt", "complexity", "model", "max_tokens"), args))
        if project_id is None:
            return super().complete(team_id, user_message, **kwargs)
        if project_id not in self._projects:
            raise KeyError(f"Unknown project '{project_id}'")
        budget, policy = self._projects[project_id]
        degraded = False
        if budget.spent / budget.monthly_budget >= self.CUTOFF:
            if policy is ProjectPolicy.HARD_STOP:
                raise BudgetExceededError(f"Project '{project_id}' reached {self.CUTOFF:.0%} of its budget (HARD_STOP)")
            kwargs["complexity"] = "simple"
            kwargs.pop("model", None)
            degraded = True
        result = super().complete(team_id, user_message, _degraded=degraded, _project_id=project_id, **kwargs)
        budget.record_cost(min(result.cost_usd, budget.remaining))
        return result


class ContextBudgetExceeded(RuntimeError):
    pass


class ContextBudgetManager:
    """Treat the context window as scarce capital: per-category token budgets,
    with RAG capped at a fraction of the total (default 50%)."""

    def __init__(self, max_total_tokens: int, category_limits: Optional[Mapping[str, int]] = None,
                 rag_max_fraction: float = 0.5, token_counter: Callable[[str], int] = estimate_tokens) -> None:
        if max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        self.max_total = max_total_tokens
        self.limits = dict(category_limits or {})
        self.rag_cap = int(max_total_tokens * rag_max_fraction)
        self.count = token_counter
        self.used: Dict[str, int] = {}

    @property
    def total_used(self) -> int:
        return sum(self.used.values())

    @property
    def remaining(self) -> int:
        return self.max_total - self.total_used

    def available(self, category: str) -> int:
        cap = self.limits.get(category, self.max_total)
        if category == "rag":
            cap = min(cap, self.rag_cap)
        return max(0, min(cap - self.used.get(category, 0), self.remaining))

    def allocate(self, category: str, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("tokens must be >= 0")
        if tokens > self.available(category):
            raise ContextBudgetExceeded(f"category '{category}': {tokens} tokens requested, {self.available(category)} available")
        self.used[category] = self.used.get(category, 0) + tokens

    def fit_chunks(self, chunks: Sequence[str], category: str = "rag") -> List[str]:
        """Keep as many leading chunks as the category budget allows (pass them best-first)."""
        kept: List[str] = []
        for chunk in chunks:
            need = self.count(chunk)
            if need > self.available(category):
                break
            self.allocate(category, need)
            kept.append(chunk)
        return kept
