#!/bin/sh
# One command to re-run the evidence. Everything here uses third-party tools; nothing asks you to trust the author.
#   sh scripts/verify.sh            # quick: tests, lint, types, security scan, dependency audit
#   sh scripts/verify.sh --full     # also: mutation testing of the security-critical modules (slow, ~10 min)
# Needs: python>=3.10, `pip install -e ".[dev]" ruff mypy bandit pip-audit`; jq + openssl for the independent signature tests.
set -u
cd "$(dirname "$0")/.."
NEW="src/sintetico/audit.py src/sintetico/autonomy.py src/sintetico/cli.py src/sintetico/delivery.py src/sintetico/docs.py src/sintetico/evals.py src/sintetico/gateway.py src/sintetico/guards.py src/sintetico/review.py src/sintetico/adapters.py src/sintetico/structured.py src/sintetico/reasoning_resilient.py src/sintetico/semantic_cache.py src/sintetico/distributed.py src/sintetico/devops"
fail=0
step() { printf '\n== %s\n' "$1"; }
run()  { "$@" || { echo "FAILED: $*"; fail=1; }; }
have() { command -v "$1" >/dev/null 2>&1; }

step "1. tests: unit + property-based (Hypothesis) + contract vs primary sources + examples + jq/openssl independent verifiers"
run python -m pytest -q
have ruff      && { step "2. ruff (bugs + security rules)";  run ruff check --select E,F,W,B,S --ignore E501,S101,E701,E702 $NEW scripts/*.py; }   || echo "(ruff not installed: skipped)"
have mypy      && { step "3. mypy (types)";                  run mypy --ignore-missing-imports --check-untyped-defs --follow-imports=silent $NEW; } || echo "(mypy not installed: skipped)"
have bandit    && { step "4. bandit (security scan)";        run bandit -q -r $NEW; }                                                               || echo "(bandit not installed: skipped)"
have pip-audit && { step "5. pip-audit (known vulnerabilities in the DECLARED dependency tree, not in your whole interpreter)"; run pip-audit -r requirements.txt; }                         || echo "(pip-audit not installed: skipped)"
if have jq && have openssl; then step "6. signatures re-verified with openssl + jq only (no project code)"; echo "(covered by tests/test_independent_verification.py)"; else echo "(jq/openssl missing: independent verifier tests skipped)"; fi
[ -n "${ANTHROPIC_API_KEY:-}" ] && { step "7. model ids vs the live Models API"; run python scripts/check_model_registry.py; } || echo "(set ANTHROPIC_API_KEY to also check model ids against the live API)"
if [ "${1:-}" = "--full" ]; then
  step "8. mutation testing (does the suite notice deliberately broken code?)"
  : > reports/mutation_report.txt
  for spec in "src/sintetico/audit.py|tests/test_audit.py tests/test_hardening.py tests/test_mutation_gaps.py" \
              "src/sintetico/autonomy.py|tests/test_evals_autonomy.py tests/test_mutation_gaps.py" \
              "src/sintetico/guards.py|tests/test_delivery_review.py tests/test_mutation_gaps.py" \
              "src/sintetico/gateway.py|tests/test_gateway.py tests/test_mutation_gaps4.py" \
              "src/sintetico/delivery.py|tests/test_delivery_review.py tests/test_mutation_gaps4.py" \
              "src/sintetico/evals.py|tests/test_evals_autonomy.py tests/test_mutation_gaps3.py" \
              "src/sintetico/devops/migration.py|tests/test_devops_ops.py tests/test_mutation_gaps.py tests/test_mutation_gaps2.py" \
              "src/sintetico/devops/iac_review.py|tests/test_devops_ops.py tests/test_primary_sources.py tests/test_mutation_gaps2.py" \
              "src/sintetico/devops/gitops.py|tests/test_devops_ops.py tests/test_primary_sources.py tests/test_hardening.py tests/test_mutation_gaps2.py" \
              "src/sintetico/devops/compliance.py|tests/test_devops_compliance.py tests/test_independent_verification.py tests/test_mutation_gaps2.py" \
              "src/sintetico/devops/residency.py|tests/test_devops_compliance.py tests/test_mutation_gaps2.py" \
              "src/sintetico/structured.py|tests/test_structured.py tests/test_mutation_gaps5.py" \
              "src/sintetico/reasoning_resilient.py|tests/test_reasoning_resilient.py tests/test_mutation_gaps5.py" \
              "src/sintetico/semantic_cache.py|tests/test_semantic_cache.py" \
              "src/sintetico/distributed.py|tests/test_distributed.py tests/test_distributed_gateway.py tests/test_mutation_gaps6.py"; do
    python scripts/mutation_check.py "${spec%%|*}" ${spec##*|} --max 45 --workers 6 | head -1 | tee -a reports/mutation_report.txt
  done
fi
[ "$fail" -eq 0 ] && echo "\nALL CHECKS PASSED" || { echo "\nSOME CHECKS FAILED"; exit 1; }
