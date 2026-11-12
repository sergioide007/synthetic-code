"""Trust gate for GitOps (book Chapter 23): block a sync of an agent whose TrustScore
annotation is missing or below the minimum. Pure function + a tiny stdlib HTTP server
that speaks the Kubernetes AdmissionReview protocol."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterable, Mapping, Tuple

__all__ = ["TrustAdmission", "serve", "ANNOTATION_PREFIX"]

ANNOTATION_PREFIX = "synthetic-code/"
MAX_BODY_BYTES = 1_048_576   # an AdmissionReview is a few KB; refuse anything absurd


class TrustAdmission:
    def __init__(self, min_trust: float = 0.70, gated_kinds: Iterable[str] = ("ClaudeAgent", "Agent")) -> None:
        self.min_trust = min_trust
        self.gated = set(gated_kinds)

    def decide(self, kind: str, annotations: Mapping[str, str]) -> Tuple[bool, str]:
        if kind not in self.gated:
            return True, "kind not gated"
        raw = annotations.get(ANNOTATION_PREFIX + "trust-score")
        if raw is None:
            return False, "missing annotation synthetic-code/trust-score (run the eval gate in CI)"
        try:
            score = float(raw)
        except ValueError:
            return False, f"invalid trust-score '{raw}'"
        if not 0.0 <= score <= 1.0:
            return False, f"trust-score {score} out of range"
        sha = annotations.get(ANNOTATION_PREFIX + "prompt-sha256", "")
        ev = annotations.get(ANNOTATION_PREFIX + "eval-set-version", "")
        if score < self.min_trust:
            return False, f"TrustScore {score:.2f} < minimum {self.min_trust:.2f} (prompt {sha[:8]}, eval-set {ev})"
        if not sha:
            return False, "missing annotation synthetic-code/prompt-sha256"
        return True, f"TrustScore {score:.2f} >= {self.min_trust:.2f}"

    def review(self, admission_review: Mapping[str, Any]) -> Dict[str, Any]:
        req = admission_review.get("request", {})
        op = str(req.get("operation", "")).upper()
        if op in ("DELETE", "CONNECT"):       # DELETE carries `object: null` (the resource is in `oldObject`): never block removal
            return {"apiVersion": "admission.k8s.io/v1", "kind": "AdmissionReview", "response": {"uid": req.get("uid", ""), "allowed": True}}
        obj = req.get("object", {}) or {}
        kind = (req.get("kind") or {}).get("kind", obj.get("kind", ""))
        allowed, why = self.decide(kind, (obj.get("metadata") or {}).get("annotations") or {})
        resp: Dict[str, Any] = {"uid": req.get("uid", ""), "allowed": allowed}
        if not allowed:
            resp["status"] = {"message": why}
        return {"apiVersion": "admission.k8s.io/v1", "kind": "AdmissionReview", "response": resp}


def serve(host: str = "127.0.0.1", port: int = 8080, min_trust: float = 0.70, server_cls=HTTPServer) -> HTTPServer:
    """Return a configured server (call ``serve_forever()``). In production put it behind TLS."""
    gate = TrustAdmission(min_trust)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/validate":
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n < 0 or n > MAX_BODY_BYTES:
                    self.send_error(413, "request body too large")
                    return
                body = json.loads(self.rfile.read(n) or b"{}")
                out = json.dumps(gate.review(body)).encode()
                self.send_response(200)
            except (ValueError, json.JSONDecodeError):
                out = json.dumps({"error": "invalid AdmissionReview"}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args: Any) -> None:  # silence default stderr logging
            pass

    return server_cls((host, port), Handler)
