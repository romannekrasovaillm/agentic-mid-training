#!/usr/bin/env bash
# Quick health check for the vLLM server.
#
# Usage:
#   bash scripts/check_vllm_health.sh              # default localhost:8000
#   bash scripts/check_vllm_health.sh localhost 8001

set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8000}"
BASE="http://${HOST}:${PORT}"

ok=0
fail=0

check() {
    local label="$1" url="$2"
    printf "%-28s" "$label"
    status=$(curl -s -o /tmp/_vllm_check.json -w "%{http_code}" --max-time 5 "$url" 2>/dev/null) || status="000"
    if [[ "$status" == "200" ]]; then
        echo "OK (${status})"
        ok=$((ok + 1))
    else
        echo "FAIL (${status})"
        fail=$((fail + 1))
    fi
}

echo "═══════════════════════════════════════"
echo "  vLLM Health Check — ${BASE}"
echo "═══════════════════════════════════════"
echo ""

# 1. /health endpoint
check "GET /health" "${BASE}/health"

# 2. /v1/models — list loaded models
check "GET /v1/models" "${BASE}/v1/models"
if [[ -f /tmp/_vllm_check.json ]]; then
    models=$(python3 -c "
import json, sys
try:
    d = json.load(open('/tmp/_vllm_check.json'))
    for m in d.get('data', []):
        print('  model:', m.get('id', '?'))
except Exception:
    pass
" 2>/dev/null)
    [[ -n "$models" ]] && echo "$models"
fi

# 3. Smoke-test: tiny completion request
echo ""
printf "%-28s" "POST /v1/completions"
resp=$(curl -s --max-time 30 -X POST "${BASE}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"__any__","prompt":"Hi","max_tokens":1,"temperature":0}' 2>/dev/null) || resp=""

# vLLM may reject unknown model name — try with the model from /v1/models
if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d" 2>/dev/null; then
    echo "OK"
    ok=$((ok + 1))
else
    # Retry with actual model name from /v1/models
    model_id=$(python3 -c "
import json
try:
    d = json.load(open('/tmp/_vllm_check.json'))
    print(d['data'][0]['id'])
except Exception:
    pass
" 2>/dev/null)
    if [[ -n "$model_id" ]]; then
        resp=$(curl -s --max-time 30 -X POST "${BASE}/v1/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${model_id}\",\"prompt\":\"Hi\",\"max_tokens\":1,\"temperature\":0}" 2>/dev/null) || resp=""
        if echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d" 2>/dev/null; then
            echo "OK (model=${model_id})"
            ok=$((ok + 1))
        else
            echo "FAIL"
            fail=$((fail + 1))
        fi
    else
        echo "FAIL"
        fail=$((fail + 1))
    fi
fi

# Summary
echo ""
echo "═══════════════════════════════════════"
if [[ "$fail" -eq 0 ]]; then
    echo "  All ${ok} checks passed"
else
    echo "  ${ok} passed, ${fail} FAILED"
fi
echo "═══════════════════════════════════════"

rm -f /tmp/_vllm_check.json
exit "$fail"
