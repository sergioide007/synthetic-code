"""Command-line entry point: ``sintetico <command>``.

Designed for CI: every command prints one JSON document to stdout and returns a
meaningful exit code, so a pipeline step can gate on it without parsing anything.

    sintetico gate            trust gate from eval results        0 ok | 1 quality | 2 critical bug | 3 drift
    sintetico eval            run an EvalSet against your agent   0 ok | 1 below pass-rate | 2 critical failure
    sintetico audit-verify    verify an HMAC-chained audit log    0 intact | 1 tampered
    sintetico evidence-verify verify signed evidence packages     0 valid | 1 invalid
    sintetico iac-review      review a `terraform show -json`     0 approve | 2 human review | 1 block
    sintetico admission-serve run the GitOps trust webhook
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import __version__
from .audit import AgentAuditLogger
from .devops.compliance import verify_evidence_package
from .devops.gitops import serve
from .devops.iac_review import IaCReviewer, InstanceMetrics
from .evals import EvalSet
from .quality import TrustScoreCalculator

EXIT_OK, EXIT_QUALITY, EXIT_CRITICAL, EXIT_DRIFT = 0, 1, 2, 3


def _emit(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _key(env_name: str) -> str:
    key = os.environ.get(env_name)
    if not key:
        raise SystemExit(f"error: environment variable {env_name} is not set")
    return key


def _critical_count(r: Dict[str, Any]) -> int:
    """`gate` accepts its own format (``critical_bugs``) and the output of ``sintetico eval``
    (``critical_failures``), so ``eval > results.json && gate --results results.json`` works.
    If both keys are present, the larger count wins (fail safe)."""
    return max(int(r.get("critical_bugs", 0) or 0), int(r.get("critical_failures", 0) or 0))


def cmd_gate(a: argparse.Namespace) -> int:
    r = _read_json(a.results)
    res = TrustScoreCalculator().calculate(
        pass_rate=float(r["pass_rate"]), precision=float(r.get("precision", r["pass_rate"])),
        human_approval=float(r.get("human_approval", 0.0)), critical_bugs=_critical_count(r),
        consistency_score=float(r.get("consistency_score", 1.0)))
    code = res.ci_exit_code
    reasons: List[str] = [] if code == 0 else [res.recommendation]
    if code == EXIT_OK and res.score < a.min_trust:
        code = EXIT_QUALITY; reasons.append(f"trust score {res.score} < --min-trust {a.min_trust}")
    if code == EXIT_OK and float(r["pass_rate"]) < a.min_pass_rate:
        code = EXIT_QUALITY; reasons.append(f"pass rate {r['pass_rate']} < --min-pass-rate {a.min_pass_rate}")
    if code == EXIT_OK and a.previous_score is not None and a.previous_score - res.score > 0.10:
        code = EXIT_DRIFT; reasons.append(f"trust dropped {a.previous_score - res.score:.2f} vs previous {a.previous_score} (> 0.10)")
    _emit({"trust_score": res.score, "level": res.level.name, "exit_code": code, "reasons": reasons})
    return code


def _load_callable(spec: str) -> Callable[[str], str]:
    if ":" not in spec:
        raise SystemExit("error: --agent must look like 'package.module:function'")
    sys.path.insert(0, os.getcwd())
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


def cmd_eval(a: argparse.Namespace) -> int:
    rep = EvalSet.from_json(a.evalset).run(_load_callable(a.agent))
    code = EXIT_CRITICAL if rep.critical_failures else (EXIT_QUALITY if rep.pass_rate < a.min_pass_rate else EXIT_OK)
    failed = [{"input": r.input, "error": r.error} for r in rep.results if not r.passed][:10]
    _emit({**rep.to_dict(), "min_pass_rate": a.min_pass_rate, "exit_code": code, "failed_sample": failed})
    return code


def cmd_audit_verify(a: argparse.Namespace) -> int:
    try:
        log = AgentAuditLogger(_key(a.key_env), path=a.file)
    except ValueError as exc:
        _emit({"chain_valid": False, "error": str(exc)}); return EXIT_QUALITY
    _emit(log.export_report())
    return EXIT_OK


def cmd_evidence_verify(a: argparse.Namespace) -> int:
    key = _key(a.key_env)
    paths = sorted(Path(a.path).rglob("evidence-package.json")) if Path(a.path).is_dir() else [Path(a.path)]
    results = {str(p): verify_evidence_package(json.loads(p.read_text(encoding="utf-8")), key) for p in paths}
    _emit({"verified": results, "all_valid": bool(results) and all(results.values())})
    return EXIT_OK if results and all(results.values()) else EXIT_QUALITY


def cmd_iac_review(a: argparse.Namespace) -> int:
    metrics = {k: InstanceMetrics(float(v["p99_cpu_pct"]), float(v["min_freeable_memory_gb"])) for k, v in (_read_json(a.metrics) if a.metrics else {}).items()}
    result = IaCReviewer().review(_read_json(a.plan), _read_json(a.deps) if a.deps else None, metrics, a.change_hour, a.cost_delta)
    _emit(result.to_dict())
    return result.exit_code


def cmd_admission_serve(a: argparse.Namespace) -> int:  # pragma: no cover - blocking server
    srv = serve(a.host, a.port, a.min_trust)
    print(f"trust webhook on http://{a.host}:{a.port}/validate (min trust {a.min_trust}). Put TLS in front for production.", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass  # Ctrl+C is the normal way to stop the server
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sintetico", description="Production patterns for multi-agent systems")
    p.add_argument("--version", action="version", version=f"sintetico {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gate", help="CI trust gate from an eval-results JSON file")
    g.add_argument("--results", required=True, help='JSON: {"pass_rate":0.9,"precision":0.9,"human_approval":1,"critical_bugs":0,"consistency_score":1}; the output of `sintetico eval` is accepted too')
    g.add_argument("--min-trust", type=float, default=0.70)
    g.add_argument("--min-pass-rate", type=float, default=0.80)
    g.add_argument("--previous-score", type=float, default=None, help="previous trust score, to detect drift")
    g.set_defaults(fn=cmd_gate)

    e = sub.add_parser("eval", help="run an eval set against your agent callable")
    e.add_argument("--evalset", required=True)
    e.add_argument("--agent", required=True, help="'package.module:function' taking a str and returning a str")
    e.add_argument("--min-pass-rate", type=float, default=0.80)
    e.set_defaults(fn=cmd_eval)

    v = sub.add_parser("audit-verify", help="verify an HMAC-chained audit log")
    v.add_argument("--file", required=True)
    v.add_argument("--key-env", default="SINTETICO_AUDIT_KEY")
    v.set_defaults(fn=cmd_audit_verify)

    ev = sub.add_parser("evidence-verify", help="verify signed evidence package(s)")
    ev.add_argument("path", help="a package file or a directory of packages")
    ev.add_argument("--key-env", default="SINTETICO_EVIDENCE_KEY")
    ev.set_defaults(fn=cmd_evidence_verify)

    i = sub.add_parser("iac-review", help="review a `terraform show -json` plan")
    i.add_argument("--plan", required=True)
    i.add_argument("--deps", help="SERVICE_DEPENDENCIES as JSON")
    i.add_argument("--metrics", help='JSON: {"<db-identifier>": {"p99_cpu_pct": 12, "min_freeable_memory_gb": 40}}')
    i.add_argument("--change-hour", type=int, default=None, help="UTC hour the change will be applied")
    i.add_argument("--cost-delta", type=float, default=0.0, help="estimated monthly cost delta in USD")
    i.set_defaults(fn=cmd_iac_review)

    s = sub.add_parser("admission-serve", help="run the GitOps trust webhook (Kubernetes AdmissionReview)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--min-trust", type=float, default=0.70)
    s.set_defaults(fn=cmd_admission_serve)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
