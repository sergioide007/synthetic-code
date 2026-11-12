"""Part VI of the book: running agents in production infrastructure.

Everything here is dependency-free and works offline. Each module documents which
real service it stands in for and what you must replace before regulated use.
"""

from .compliance import CONTROL_MAP, EvidenceStore, StageResult, build_evidence_package, verify_evidence_package
from .gitops import TrustAdmission
from .iac_review import Decision, IaCReviewer, InstanceMetrics, ReviewResult
from .incident import ApprovalRequired, GuardedToolbox, InMemoryQueue, IncidentSession, ToolOutcome, ToolPolicy, inspect_dlq, replay_dlq
from .migration import DualWriteRepository, MigrationAdvisor, Phase, Recommendation, row_hash, verify_batch
from .residency import RegulatoryProfile, ResidencyDecision, ResidencyPolicy, load_profile, load_profiles

__all__ = [n for n in dir() if not n.startswith("_")]
