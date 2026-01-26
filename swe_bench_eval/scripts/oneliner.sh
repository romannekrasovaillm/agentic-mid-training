#!/bin/bash
#
# ONE-LINER SWE-bench Lite Runner
#
# Копируй и запускай на удалённой машине:
#
# bash <(curl -sSL https://raw.githubusercontent.com/romannekrasovaillm/agentic-mid-training/claude/integrate-lora-swe-benchmark-SEgGh/swe_bench_eval/scripts/oneliner.sh) \
#     --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
#     --lora-path /path/to/lora
#
# Или одной строкой (замени параметры):
#
# export BASE_MODEL="ai-sage/GigaChat3-10B-A1.8B-bf16" LORA_PATH="/path/to/lora" && \
# curl -sSL https://raw.githubusercontent.com/romannekrasovaillm/agentic-mid-training/claude/integrate-lora-swe-benchmark-SEgGh/swe_bench_eval/scripts/oneliner.sh | bash
#

set -eo pipefail

# Defaults (можно переопределить через export)
BASE_MODEL="${BASE_MODEL:-}"
LORA_PATH="${LORA_PATH:-}"
MAX_INSTANCES="${MAX_INSTANCES:-}"
SERVER_PORT="${SERVER_PORT:-8080}"
USE_VLLM="${USE_VLLM:-true}"
SANDBOX="${SANDBOX:-docker}"

# Parse args if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --lora-path) LORA_PATH="$2"; shift 2 ;;
        --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --port) SERVER_PORT="$2"; shift 2 ;;
        --no-vllm) USE_VLLM="false"; shift ;;
        --sandbox) SANDBOX="$2"; shift 2 ;;
        *) shift ;;
    esac
done

[[ -z "$BASE_MODEL" ]] && { echo "Error: BASE_MODEL or --base-model required"; exit 1; }
[[ -z "$LORA_PATH" ]] && { echo "Error: LORA_PATH or --lora-path required"; exit 1; }
[[ -d "$LORA_PATH" ]] || { echo "Error: LoRA path not found: $LORA_PATH"; exit 1; }

LORA_PATH="$(cd "$(dirname "$LORA_PATH")" && pwd)/$(basename "$LORA_PATH")"
WORK_DIR="/tmp/swe-lite-$(date +%s)"
OUTPUT_DIR="$WORK_DIR/outputs"

echo "=== SWE-bench Lite Isolated Run ==="
echo "Work dir: $WORK_DIR"
echo "Model: $BASE_MODEL"
echo "LoRA: $LORA_PATH"

mkdir -p "$WORK_DIR" && cd "$WORK_DIR"

# Clone
git clone --depth 1 -b claude/integrate-lora-swe-benchmark-SEgGh \
    https://github.com/romannekrasovaillm/agentic-mid-training.git repo
cd repo

# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch transformers peft accelerate vllm flask bitsandbytes litellm
pip install -q mini-swe-agent || pip install -q git+https://github.com/SWE-agent/mini-swe-agent.git

mkdir -p "$OUTPUT_DIR/logs"

# Start server
cleanup() { [[ -n "$SPID" ]] && kill "$SPID" 2>/dev/null || true; }
trap cleanup EXIT

if [[ "$USE_VLLM" == "true" ]]; then
    python swe_bench_eval/models/vllm_lora_server.py \
        --base-model "$BASE_MODEL" --lora-path "$LORA_PATH" --port "$SERVER_PORT" \
        > "$OUTPUT_DIR/logs/server.log" 2>&1 &
else
    python swe_bench_eval/models/lora_server.py \
        --base-model "$BASE_MODEL" --lora-path "$LORA_PATH" --port "$SERVER_PORT" \
        > "$OUTPUT_DIR/logs/server.log" 2>&1 &
fi
SPID=$!

# Wait for server
echo "Waiting for server..."
for i in {1..60}; do
    curl -s "http://localhost:$SERVER_PORT/health" >/dev/null 2>&1 && break
    sleep 10
    kill -0 "$SPID" 2>/dev/null || { echo "Server died"; cat "$OUTPUT_DIR/logs/server.log"; exit 1; }
done
curl -s "http://localhost:$SERVER_PORT/health" >/dev/null || { echo "Server timeout"; exit 1; }
echo "Server ready!"

# Run benchmark
export OPENAI_API_KEY="not-needed" OPENAI_API_BASE="http://localhost:$SERVER_PORT/v1"

CMD="mini swebench --model openai/default --dataset princeton-nlp/SWE-bench_Lite --sandbox $SANDBOX --output $OUTPUT_DIR/trajectories"
[[ -n "$MAX_INSTANCES" ]] && CMD="$CMD --max-instances $MAX_INSTANCES"

echo "Running: $CMD"
eval "$CMD" 2>&1 | tee "$OUTPUT_DIR/logs/agent.log"

# Summary
echo ""
echo "=== Done ==="
echo "Output: $OUTPUT_DIR"
TOTAL=$(find "$OUTPUT_DIR/trajectories" -name "*.json" 2>/dev/null | wc -l)
RESOLVED=$(grep -l '"resolved": true' "$OUTPUT_DIR/trajectories"/*.json 2>/dev/null | wc -l || echo 0)
echo "Resolved: $RESOLVED / $TOTAL"
