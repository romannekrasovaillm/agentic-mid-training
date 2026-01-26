#!/bin/bash
#
# One-liner SWE-bench Lite Runner
#
# Клонирует репозиторий в изолированную директорию и запускает бенчмарк.
# Не затрагивает другие ветки/PR.
#
# Использование (одной командой):
#   curl -sSL <url>/run_swe_lite_isolated.sh | bash -s -- \
#       --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
#       --lora-path /path/to/lora
#
# Или напрямую:
#   ./run_swe_lite_isolated.sh \
#       --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
#       --lora-path /path/to/lora
#

set -e

# =============================================================================
# Configuration
# =============================================================================
REPO_URL="https://github.com/romannekrasovaillm/agentic-mid-training.git"
BRANCH="claude/integrate-lora-swe-benchmark-SEgGh"
WORK_DIR="/tmp/swe-bench-eval-$(date +%s)"

BASE_MODEL=""
LORA_PATH=""
MAX_INSTANCES=""
SERVER_PORT=8080
OUTPUT_DIR=""
USE_VLLM=true
GPU_MEMORY=0.9
TENSOR_PARALLEL=1
SANDBOX="docker"
SKIP_SETUP=false

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --no-vllm)
            USE_VLLM=false
            shift
            ;;
        --gpu-memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --sandbox)
            SANDBOX="$2"
            shift 2
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --help)
            cat << 'HELP'
SWE-bench Lite Isolated Runner

Usage:
  ./run_swe_lite_isolated.sh --base-model <model> --lora-path <path> [options]

Required:
  --base-model       Base model (HuggingFace repo or local path)
  --lora-path        Path to LoRA adapter directory

Optional:
  --max-instances    Limit instances (default: all ~300 for Lite)
  --output-dir       Output directory (default: ./outputs in work dir)
  --port             Model server port (default: 8080)
  --work-dir         Working directory (default: /tmp/swe-bench-eval-<timestamp>)
  --no-vllm          Use transformers instead of vLLM
  --gpu-memory       GPU memory utilization (default: 0.9)
  --tensor-parallel  Number of GPUs (default: 1)
  --sandbox          docker, podman, singularity (default: docker)
  --skip-setup       Skip pip install (if already set up)

Example:
  ./run_swe_lite_isolated.sh \
      --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
      --lora-path ./checkpoints/my-lora \
      --max-instances 50
HELP
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate
# =============================================================================
if [[ -z "$BASE_MODEL" ]]; then
    echo "Error: --base-model is required"
    exit 1
fi

if [[ -z "$LORA_PATH" ]]; then
    echo "Error: --lora-path is required"
    exit 1
fi

# Convert to absolute path
LORA_PATH="$(cd "$(dirname "$LORA_PATH")" && pwd)/$(basename "$LORA_PATH")"

if [[ ! -d "$LORA_PATH" ]]; then
    echo "Error: LoRA path does not exist: $LORA_PATH"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$WORK_DIR/outputs"
fi

# =============================================================================
# Setup
# =============================================================================
echo "=============================================="
echo "SWE-bench Lite - Isolated Run"
echo "=============================================="
echo "Work Directory:   $WORK_DIR"
echo "Base Model:       $BASE_MODEL"
echo "LoRA Path:        $LORA_PATH"
echo "Output Dir:       $OUTPUT_DIR"
echo "Server Port:      $SERVER_PORT"
echo "=============================================="

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository
echo ""
echo "[1/5] Cloning repository..."
if [[ ! -d "agentic-mid-training" ]]; then
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" agentic-mid-training
else
    echo "Repository already exists, pulling latest..."
    cd agentic-mid-training
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
    cd ..
fi

cd agentic-mid-training

# Install dependencies
if ! $SKIP_SETUP; then
    echo ""
    echo "[2/5] Installing dependencies..."

    # Create virtual environment if needed
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate

    pip install --upgrade pip -q
    pip install -r swe_bench_eval/requirements.txt -q

    # Install mini-swe-agent
    pip install mini-swe-agent -q || pip install git+https://github.com/SWE-agent/mini-swe-agent.git -q
else
    echo ""
    echo "[2/5] Skipping setup (--skip-setup)"
    if [[ -d ".venv" ]]; then
        source .venv/bin/activate
    fi
fi

# Make scripts executable
chmod +x swe_bench_eval/scripts/*.sh

# =============================================================================
# Start Model Server
# =============================================================================
echo ""
echo "[3/5] Starting model server..."

mkdir -p "$OUTPUT_DIR"/{trajectories,predictions,logs,results}

SERVER_LOG="$OUTPUT_DIR/logs/server.log"
SERVER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [[ -n "$SERVER_PID" ]]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if $USE_VLLM; then
    python swe_bench_eval/models/vllm_lora_server.py \
        --base-model "$BASE_MODEL" \
        --lora-path "$LORA_PATH" \
        --port "$SERVER_PORT" \
        --gpu-memory-utilization "$GPU_MEMORY" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
else
    python swe_bench_eval/models/lora_server.py \
        --base-model "$BASE_MODEL" \
        --lora-path "$LORA_PATH" \
        --port "$SERVER_PORT" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
fi

echo "Server PID: $SERVER_PID"
echo "Server log: $SERVER_LOG"

# Wait for server
echo "Waiting for server to start..."
MAX_WAIT=600
WAIT_COUNT=0
while ! curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; do
    sleep 10
    WAIT_COUNT=$((WAIT_COUNT + 10))
    if [[ $WAIT_COUNT -ge $MAX_WAIT ]]; then
        echo "Error: Server failed to start within $MAX_WAIT seconds"
        echo "Last 50 lines of server log:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    echo "  Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"

    # Check if server process died
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: Server process died"
        echo "Server log:"
        cat "$SERVER_LOG"
        exit 1
    fi
done
echo "Server is ready!"

# =============================================================================
# Run SWE-bench Lite
# =============================================================================
echo ""
echo "[4/5] Running SWE-bench Lite..."

export OPENAI_API_KEY="not-needed"
export OPENAI_API_BASE="http://localhost:$SERVER_PORT/v1"

# Build command
MINI_CMD="mini swebench"
MINI_CMD="$MINI_CMD --model openai/default"
MINI_CMD="$MINI_CMD --dataset princeton-nlp/SWE-bench_Lite"
MINI_CMD="$MINI_CMD --sandbox $SANDBOX"
MINI_CMD="$MINI_CMD --output $OUTPUT_DIR/trajectories"

if [[ -n "$MAX_INSTANCES" ]]; then
    MINI_CMD="$MINI_CMD --max-instances $MAX_INSTANCES"
fi

echo "Running: $MINI_CMD"
echo ""
eval "$MINI_CMD" 2>&1 | tee "$OUTPUT_DIR/logs/agent.log"

# =============================================================================
# Results Summary
# =============================================================================
echo ""
echo "[5/5] Generating results summary..."

# Count results
TOTAL_TRAJ=$(find "$OUTPUT_DIR/trajectories" -name "*.json" 2>/dev/null | wc -l)
RESOLVED=$(grep -l '"resolved": true' "$OUTPUT_DIR/trajectories"/*.json 2>/dev/null | wc -l || echo 0)

echo ""
echo "=============================================="
echo "SWE-bench Lite Complete!"
echo "=============================================="
echo "Work Directory: $WORK_DIR"
echo "Output Dir:     $OUTPUT_DIR"
echo ""
echo "Results:"
echo "  Total instances: $TOTAL_TRAJ"
echo "  Resolved:        $RESOLVED"
if [[ $TOTAL_TRAJ -gt 0 ]]; then
    RATE=$(echo "scale=1; $RESOLVED * 100 / $TOTAL_TRAJ" | bc)
    echo "  Success rate:    ${RATE}%"
fi
echo ""
echo "Files:"
echo "  Trajectories: $OUTPUT_DIR/trajectories/"
echo "  Server log:   $OUTPUT_DIR/logs/server.log"
echo "  Agent log:    $OUTPUT_DIR/logs/agent.log"
echo ""
echo "View trajectories:"
echo "  mini trajectory-browser $OUTPUT_DIR/trajectories"
echo "=============================================="
