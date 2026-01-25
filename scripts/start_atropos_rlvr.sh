#!/bin/bash
#
# Start Atropos RLVR Training Pipeline for GigaChat
#
# This script starts the full Atropos pipeline:
# 1. Trajectory API (central rollout storage)
# 2. vLLM inference server
# 3. Environment server (generates rollouts + rewards)
# 4. Trainer (policy gradient on collected trajectories)
#
# Based on NousResearch/atropos framework.
#
# Usage:
#   ./scripts/start_atropos_rlvr.sh                    # Full pipeline
#   ./scripts/start_atropos_rlvr.sh --api-only         # Only trajectory API
#   ./scripts/start_atropos_rlvr.sh --env-only         # Only environment server
#   ./scripts/start_atropos_rlvr.sh --trainer-only     # Only trainer
#   ./scripts/start_atropos_rlvr.sh --kill             # Kill all processes
#

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME="${MODEL_NAME:-ai-sage/GigaChat3-10B-A1.8B-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/atropos-rlvr}"

# Ports
TRAJECTORY_API_PORT="${TRAJECTORY_API_PORT:-8000}"
VLLM_PORT="${VLLM_PORT:-8001}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-8}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORK_DIR="${PROJECT_ROOT}/experiments/gigachat-a100"
VENV_DIR="${WORK_DIR}/venv"

# Log files
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$LOG_DIR"

TRAJECTORY_API_LOG="${LOG_DIR}/trajectory_api.log"
VLLM_LOG="${LOG_DIR}/vllm_server.log"
ENV_LOG="${LOG_DIR}/environment.log"
TRAINER_LOG="${LOG_DIR}/trainer.log"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

API_ONLY=false
ENV_ONLY=false
TRAINER_ONLY=false
VLLM_ONLY=false
KILL_ALL=false
SKIP_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-only)
            API_ONLY=true
            shift
            ;;
        --env-only)
            ENV_ONLY=true
            shift
            ;;
        --trainer-only)
            TRAINER_ONLY=true
            shift
            ;;
        --vllm-only)
            VLLM_ONLY=true
            shift
            ;;
        --kill)
            KILL_ALL=true
            shift
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --total-steps)
            TOTAL_STEPS="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --api-only        Start only Trajectory API"
            echo "  --env-only        Start only Environment server"
            echo "  --trainer-only    Start only Trainer"
            echo "  --vllm-only       Start only vLLM server"
            echo "  --kill            Kill all running processes"
            echo "  --skip-install    Skip package installation"
            echo "  --model NAME      Model name (default: $MODEL_NAME)"
            echo "  --output-dir DIR  Output directory (default: $OUTPUT_DIR)"
            echo "  --batch-size N    Batch size (default: $BATCH_SIZE)"
            echo "  --total-steps N   Total training steps (default: $TOTAL_STEPS)"
            echo "  --max-samples N   Max dataset samples (default: $MAX_SAMPLES)"
            echo "  --learning-rate   Learning rate (default: $LEARNING_RATE)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# FUNCTIONS
# =============================================================================

kill_processes() {
    echo "Killing all Atropos-related processes..."
    pkill -f "run-api" || true
    pkill -f "atropos" || true
    pkill -f "vllm.entrypoints" || true
    pkill -f "gigachat_tool_calling_env" || true
    pkill -f "atropos_trainer" || true
    sleep 2
    echo "Done."
}

wait_for_service() {
    local url="$1"
    local name="$2"
    local max_attempts="${3:-60}"
    local attempt=0

    echo "Waiting for $name at $url..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "$name is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done

    echo "ERROR: $name failed to start after $max_attempts attempts"
    return 1
}

activate_venv() {
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        source "${VENV_DIR}/bin/activate"
    else
        echo "Virtual environment not found. Creating..."
        python3 -m venv "$VENV_DIR"
        source "${VENV_DIR}/bin/activate"
    fi
}

install_packages() {
    echo "Installing required packages..."
    pip install --upgrade pip
    pip install atroposlib
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate datasets peft bitsandbytes
    pip install vllm openai aiohttp
    echo "Installation complete."
}

start_trajectory_api() {
    echo "Starting Trajectory API on port $TRAJECTORY_API_PORT..."

    # Check if atroposlib has run-api command
    if command -v run-api &> /dev/null; then
        nohup run-api --port "$TRAJECTORY_API_PORT" > "$TRAJECTORY_API_LOG" 2>&1 &
    else
        # Fallback: start simple trajectory API server
        nohup python -c "
import json
import asyncio
from aiohttp import web
from collections import deque

trajectories = deque(maxlen=100000)
training_metrics = []

async def health(request):
    return web.Response(text='ok')

async def add_trajectory(request):
    data = await request.json()
    trajectories.append(data)
    return web.json_response({'status': 'ok', 'queue_size': len(trajectories)})

async def get_batch(request):
    batch_size = int(request.query.get('batch_size', 8))
    batch = []
    for _ in range(min(batch_size, len(trajectories))):
        if trajectories:
            batch.append(trajectories.popleft())
    if not batch:
        return web.Response(status=204)
    return web.json_response(batch)

async def report_metrics(request):
    data = await request.json()
    training_metrics.append(data)
    return web.json_response({'status': 'ok'})

async def get_metrics(request):
    return web.json_response(training_metrics[-100:] if training_metrics else [])

app = web.Application()
app.router.add_get('/health', health)
app.router.add_post('/trajectories', add_trajectory)
app.router.add_get('/trajectories/batch', get_batch)
app.router.add_post('/training/metrics', report_metrics)
app.router.add_get('/training/metrics', get_metrics)

if __name__ == '__main__':
    web.run_app(app, port=$TRAJECTORY_API_PORT)
" > "$TRAJECTORY_API_LOG" 2>&1 &
    fi

    TRAJECTORY_API_PID=$!
    echo "Trajectory API started (PID: $TRAJECTORY_API_PID)"
}

start_vllm_server() {
    echo "Starting vLLM server on port $VLLM_PORT..."

    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 1536 \
        --gpu-memory-utilization 0.6 \
        > "$VLLM_LOG" 2>&1 &

    VLLM_PID=$!
    echo "vLLM server started (PID: $VLLM_PID)"
}

start_environment() {
    echo "Starting GigaChat Tool Calling Environment..."

    cd "$PROJECT_ROOT"

    nohup python environments/gigachat_tool_calling_env.py serve \
        --env.rollout_server_url "http://localhost:$TRAJECTORY_API_PORT" \
        --env.max_samples "$MAX_SAMPLES" \
        --env.group_size "$GROUP_SIZE" \
        --openai.base_url "http://localhost:$VLLM_PORT/v1" \
        --openai.model_name "$MODEL_NAME" \
        > "$ENV_LOG" 2>&1 &

    ENV_PID=$!
    echo "Environment started (PID: $ENV_PID)"
}

start_trainer() {
    echo "Starting Atropos Trainer..."

    cd "$PROJECT_ROOT"

    python src/training/atropos_trainer.py \
        --model "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --trajectory-api "http://localhost:$TRAJECTORY_API_PORT" \
        --batch-size "$BATCH_SIZE" \
        --total-steps "$TOTAL_STEPS" \
        --learning-rate "$LEARNING_RATE" \
        2>&1 | tee "$TRAINER_LOG"
}

# =============================================================================
# MAIN
# =============================================================================

echo "============================================================"
echo "  ATROPOS RLVR TRAINING PIPELINE"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Trajectory API port: $TRAJECTORY_API_PORT"
echo "  vLLM port: $VLLM_PORT"
echo "  Batch size: $BATCH_SIZE"
echo "  Total steps: $TOTAL_STEPS"
echo "  Max samples: $MAX_SAMPLES"
echo ""

# Kill all if requested
if [ "$KILL_ALL" = true ]; then
    kill_processes
    exit 0
fi

# Change to work directory
cd "$WORK_DIR"

# Activate virtual environment
activate_venv

# Install packages if needed
if [ "$SKIP_INSTALL" = false ]; then
    install_packages
fi

# Export environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start components based on arguments
if [ "$API_ONLY" = true ]; then
    start_trajectory_api
    wait_for_service "http://localhost:$TRAJECTORY_API_PORT/health" "Trajectory API"
    echo ""
    echo "Trajectory API running. Logs: $TRAJECTORY_API_LOG"
    echo "To stop: pkill -f 'run-api'"
    exit 0
fi

if [ "$VLLM_ONLY" = true ]; then
    start_vllm_server
    wait_for_service "http://localhost:$VLLM_PORT/health" "vLLM Server" 120
    echo ""
    echo "vLLM server running. Logs: $VLLM_LOG"
    echo "To stop: pkill -f 'vllm.entrypoints'"
    exit 0
fi

if [ "$ENV_ONLY" = true ]; then
    start_environment
    echo ""
    echo "Environment running. Logs: $ENV_LOG"
    echo "To stop: pkill -f 'gigachat_tool_calling_env'"
    exit 0
fi

if [ "$TRAINER_ONLY" = true ]; then
    start_trainer
    exit 0
fi

# Full pipeline
echo "Starting full Atropos pipeline..."
echo ""

# Step 1: Trajectory API
echo "[1/4] Starting Trajectory API..."
start_trajectory_api
wait_for_service "http://localhost:$TRAJECTORY_API_PORT/health" "Trajectory API"

# Step 2: vLLM Server
echo ""
echo "[2/4] Starting vLLM Inference Server..."
start_vllm_server
wait_for_service "http://localhost:$VLLM_PORT/health" "vLLM Server" 120

# Step 3: Environment
echo ""
echo "[3/4] Starting Environment Server..."
start_environment
sleep 5

# Step 4: Trainer
echo ""
echo "[4/4] Starting Trainer..."
echo ""
echo "============================================================"
echo "  TRAINING STARTED"
echo "============================================================"
echo ""
echo "Logs:"
echo "  Trajectory API: $TRAJECTORY_API_LOG"
echo "  vLLM Server:    $VLLM_LOG"
echo "  Environment:    $ENV_LOG"
echo "  Trainer:        $TRAINER_LOG"
echo ""
echo "To monitor: tail -f $LOG_DIR/*.log"
echo "To stop:    $0 --kill"
echo ""

start_trainer

echo ""
echo "============================================================"
echo "  TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Output saved to: $OUTPUT_DIR"

# Cleanup
kill_processes
