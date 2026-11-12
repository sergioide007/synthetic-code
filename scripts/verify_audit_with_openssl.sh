#!/bin/sh
# Verify an AgentAuditLogger JSONL file WITHOUT any code from this project: only jq + openssl.
#   SINTETICO_AUDIT_KEY=secret ./scripts/verify_audit_with_openssl.sh audit.jsonl
# Optional 2nd argument: the chain head you stored elsewhere, to also detect truncation.
set -eu
[ -n "${SINTETICO_AUDIT_KEY:-}" ] || { echo "set SINTETICO_AUDIT_KEY" >&2; exit 2; }
prev=genesis; n=0
while IFS= read -r line; do
  [ -n "$line" ] || continue
  payload=$(printf '%s' "$line" | jq -S -c -a '{event_id,timestamp,actor,action,details,prev_hash}')
  own=$(printf '%s' "$line" | jq -r .own_hash)
  linked=$(printf '%s' "$line" | jq -r .prev_hash)
  calc=$(printf '%s' "$payload" | openssl dgst -sha256 -hmac "$SINTETICO_AUDIT_KEY" -r | cut -d' ' -f1)
  [ "$linked" = "$prev" ] || { echo "BROKEN LINK at event $n"; exit 1; }
  [ "$calc" = "$own" ]    || { echo "BAD SIGNATURE at event $n"; exit 1; }
  prev=$own; n=$((n+1))
done < "$1"
if [ -n "${2:-}" ] && [ "$2" != "$prev" ]; then echo "TRUNCATED: chain head $prev != expected $2"; exit 1; fi
echo "OK: $n events verified, chain head $prev"
