"""Documentation generated from code and tests, plus Architecture Decision Records (book Chapter 14)."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Callable, List, Optional, Sequence, Union

__all__ = ["DocumentGenerator", "ADR", "ADRGenerator", "ADRArchive", "DECISION_KEYWORDS"]


class DocumentGenerator:
    """Document the *state* of the code from real signatures and real test usages (never invented examples)."""

    def __init__(self, test_dirs: Sequence[Union[str, Path]] = (), max_examples: int = 3) -> None:
        self.test_sources: List[str] = []
        for d in test_dirs:
            for f in Path(d).rglob("test_*.py"):
                self.test_sources.append(f.read_text(encoding="utf-8", errors="replace"))
        self.max_examples = max_examples

    def examples_for(self, name: str) -> List[str]:
        rx = re.compile(rf"^\s*(?:\w+\s*=\s*)?(?:assert\s+)?[\w\.]*{re.escape(name)}\(.*$")
        found: List[str] = []
        for src in self.test_sources:
            for line in src.splitlines():
                if rx.match(line) and line.strip() not in found:
                    found.append(line.strip())
                if len(found) >= self.max_examples:
                    return found
        return found

    def document_module(self, module: ModuleType) -> str:
        out = [f"# `{module.__name__}`", "", (inspect.getdoc(module) or "").split("\n\n")[0], ""]
        for name, obj in sorted(vars(module).items()):
            if name.startswith("_") or not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            try:
                sig = str(inspect.signature(obj)).replace("'", "")   # postponed annotations print quoted
            except (TypeError, ValueError):
                sig = "(...)"
            doc = (inspect.getdoc(obj) or "").split("\n")[0]
            out += [f"## `{name}{sig}`", "", doc, ""]
            ex = self.examples_for(name)
            if ex:
                out += ["```python", *ex, "```", ""]
        return "\n".join(out)


DECISION_KEYWORDS = ("architecture", "migrate", "migration", "rfc", "adr", "replace", "deprecate", "introduce", "switch to")


@dataclass
class ADR:
    title: str
    context: str
    decision: str
    consequences: str
    status: str = "proposed"
    date: str = ""

    def to_markdown(self, number: int = 0) -> str:
        d = self.date or date.today().isoformat()
        return (f"# ADR-{number:04d}: {self.title}\n\n- Status: {self.status}\n- Date: {d}\n\n"
                f"## Context\n{self.context}\n\n## Decision\n{self.decision}\n\n## Consequences\n{self.consequences}\n")


class ADRGenerator:
    def __init__(self, llm_call: Optional[Callable[[str], str]] = None) -> None:
        self.llm_call = llm_call

    @staticmethod
    def looks_like_decision(message: str) -> bool:
        low = message.lower()
        return any(k in low for k in DECISION_KEYWORDS)

    def from_decision(self, title: str, context: str, decision: str, consequences: str, status: str = "accepted") -> ADR:
        return ADR(title, context, decision, consequences, status)

    def from_commit(self, message: str, diff: str = "") -> Optional[ADR]:
        """Rescue undocumented decisions: only commits that *sound* architectural produce an ADR draft (status=proposed)."""
        if not self.looks_like_decision(message):
            return None
        title = message.strip().splitlines()[0][:80]
        if self.llm_call:
            text = self.llm_call(f"Write Context/Decision/Consequences (one paragraph each) for this commit.\nMessage: {message}\nDiff:\n{diff[:4000]}")
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            parts += ["(not provided)"] * (3 - len(parts))
            return ADR(title, parts[0], parts[1], parts[2], "proposed")
        return ADR(title, "Inferred from commit message; needs review.", message.strip(), "To be completed by the author.", "proposed")


class ADRArchive:
    def __init__(self, directory: Union[str, Path]) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _next_number(self) -> int:
        nums = [int(m.group(1)) for p in self.dir.glob("*.md") if (m := re.match(r"(\d{4})-", p.name))]
        return max(nums, default=0) + 1

    def save(self, adr: ADR) -> Path:
        n = self._next_number()
        slug = re.sub(r"[^a-z0-9]+", "-", adr.title.lower()).strip("-")[:50] or "decision"
        path = self.dir / f"{n:04d}-{slug}.md"
        path.write_text(adr.to_markdown(n), encoding="utf-8")
        return path

    def list(self) -> List[Path]:
        return sorted(self.dir.glob("*.md"))

    def search(self, query: str, limit: int = 5) -> List[Path]:
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
        scored = []
        for p in self.list():
            text = p.read_text(encoding="utf-8").lower()
            score = sum(text.count(t) for t in terms)
            if score:
                scored.append((score, p))
        return [p for _, p in sorted(scored, key=lambda x: (-x[0], x[1].name))[:limit]]
