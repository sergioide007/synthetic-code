#!/bin/sh
# Verify a signed evidence package WITHOUT any code from this project: only jq + openssl.
#   SINTETICO_EVIDENCE_KEY=secret ./scripts/verify_evidence_with_openssl.sh evidence-package.json
set -eu
[ -n "${SINTETICO_EVIDENCE_KEY:-}" ] || { echo "set SINTETICO_EVIDENCE_KEY" >&2; exit 2; }
claimed=$(jq -r .hmac_sha256 "$1")
calc=$(jq -S -c -a 'del(.hmac_sha256)' "$1" | tr -d '\n' | openssl dgst -sha256 -hmac "$SINTETICO_EVIDENCE_KEY" -r | cut -d' ' -f1)
[ "$claimed" = "$calc" ] && echo "OK: signature valid" || { echo "INVALID: signature mismatch"; exit 1; }
