"""Policy as data: regulatory profiles and data-residency evaluation (book Chapters 19 and 27).

Profiles are plain JSON (or YAML if PyYAML is installed). A regulatory change is a
data edit reviewed in a pull request, not a code deploy.

IMPORTANT: the example profiles shipped with this project contain *illustrative*
values. They are not legal advice. Have counsel validate every profile you use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

__all__ = ["RegulatoryProfile", "load_profile", "load_profiles", "ResidencyDecision", "ResidencyPolicy"]


@dataclass
class RegulatoryProfile:
    country: str
    regulation: str
    regulator: str = ""
    allowed_regions: List[str] = field(default_factory=list)
    cross_border_requires_consent: bool = True
    breach_notification_hours: int = 72
    retention_years: Dict[str, int] = field(default_factory=dict)
    ai_decisions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RegulatoryProfile":
        res = d.get("data_residency", {})
        return cls(
            country=d["country"],
            regulation=d.get("regulation", ""),
            regulator=d.get("regulator", ""),
            allowed_regions=list(res.get("allowed_regions", d.get("allowed_regions", []))),
            cross_border_requires_consent=bool(res.get("cross_border_requires_consent", True)),  # fail safe default
            breach_notification_hours=int(d.get("breach_notification", {}).get("deadline_hours", d.get("breach_notification_hours", 72))),
            retention_years=dict(d.get("retention", d.get("retention_years", {}))),
            ai_decisions=dict(d.get("ai_decisions", {})),
        )


def _read(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("PyYAML is required for YAML profiles: pip install pyyaml") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def load_profile(path: Union[str, Path]) -> RegulatoryProfile:
    return RegulatoryProfile.from_dict(_read(Path(path)))


def load_profiles(directory: Union[str, Path]) -> Dict[str, RegulatoryProfile]:
    out: Dict[str, RegulatoryProfile] = {}
    for p in sorted(Path(directory).iterdir()):
        if p.suffix in (".json", ".yaml", ".yml"):
            prof = load_profile(p)
            out[prof.country.upper()] = prof
    return out


@dataclass
class ResidencyDecision:
    allowed: bool
    reason: str
    country: str = ""
    region: str = ""


class ResidencyPolicy:
    """Fail closed: unknown country, unknown profile or missing consent => deny."""

    def __init__(self, profiles: Mapping[str, RegulatoryProfile]) -> None:
        self.profiles = {k.upper(): v for k, v in profiles.items()}

    def evaluate_write(self, country: str, target_region: str, consent: Optional[Mapping[str, Any]] = None) -> ResidencyDecision:
        c = (country or "").upper()
        prof = self.profiles.get(c)
        if prof is None:
            return ResidencyDecision(False, f"no regulatory profile for '{country}' (fail closed)", c, target_region)
        if target_region in prof.allowed_regions:
            return ResidencyDecision(True, f"{target_region} is an allowed region for {c}", c, target_region)
        if not prof.cross_border_requires_consent:
            return ResidencyDecision(True, "cross-border transfer allowed without consent by profile", c, target_region)
        if consent and consent.get("explicit") is True and str(consent.get("country", "")).upper() == c:
            return ResidencyDecision(True, "cross-border transfer covered by explicit consent", c, target_region)
        return ResidencyDecision(False, f"{c} data cannot be written to {target_region} without explicit consent", c, target_region)

    def evaluate_deployment(self, placements: Mapping[str, Tuple[str, str]], consents: Optional[Mapping[str, Mapping[str, Any]]] = None) -> Dict[str, Any]:
        """``placements``: resource -> (customer_country, region). Returns every violation, not just the first."""
        violations = []
        for resource, (country, region) in placements.items():
            d = self.evaluate_write(country, region, (consents or {}).get(resource))
            if not d.allowed:
                violations.append({"resource": resource, "country": d.country, "region": region, "reason": d.reason})
        return {"compliant": not violations, "violations": violations}
