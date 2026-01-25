#!/bin/bash
#
# Full Atropos RLVR Training Pipeline for GigaChat
#
# This script runs the complete RLVR pipeline with all components:
# 1. vLLM Server - for fast inference
# 2. Trajectory API - manages rollout queue
# 3. Environment - generates rollouts, computes rewards
# 4. Trainer - policy gradient training
#
# Usage:
#   ./scripts/start_rlvr_full.sh                        # Default settings
#   ./scripts/start_rlvr_full.sh --steps 5000           # More training steps
#   ./scripts/start_rlvr_full.sh --trainer-only         # Run trainer only (if vLLM running)
#

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME="${MODEL_NAME:-ai-sage/GigaChat3-10B-A1.8B-bf16}"
DATASET_NAME="${DATASET_NAME:-nvidia/Nemotron-Agentic-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/rlvr-full}"

# Training
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

# Generation
GROUP_SIZE="${GROUP_SIZE:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"

# Ports
VLLM_PORT="${VLLM_PORT:-8001}"
API_PORT="${API_PORT:-8000}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORK_DIR="${PROJECT_ROOT}/experiments/gigachat-a100"
VENV_DIR="${WORK_DIR}/venv"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

SKIP_INSTALL=false
KILL_PYTHON=false
TRAINER_ONLY=false
VLLM_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --kill)
            KILL_PYTHON=true
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
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --steps)
            TOTAL_STEPS="$2"
            shift 2
            ;;
        --batch-size|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --group-size)
            GROUP_SIZE="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --vllm-port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Full Atropos RLVR Training Pipeline"
            echo ""
            echo "Options:"
            echo "  --skip-install     Skip package installation"
            echo "  --kill             Kill Python processes before starting"
            echo "  --trainer-only     Run only the trainer (assumes vLLM running)"
            echo "  --vllm-only        Run only vLLM server"
            echo "  --model NAME       Model name (default: $MODEL_NAME)"
            echo "  --dataset NAME     Dataset name (default: $DATASET_NAME)"
            echo "  --output-dir DIR   Output directory (default: $OUTPUT_DIR)"
            echo "  --steps N          Total training steps (default: $TOTAL_STEPS)"
            echo "  --batch-size N     Batch size (default: $BATCH_SIZE)"
            echo "  --learning-rate F  Learning rate (default: $LEARNING_RATE)"
            echo "  --samples N        Max dataset samples (default: $MAX_SAMPLES)"
            echo "  --group-size N     Rollouts per prompt (default: $GROUP_SIZE)"
            echo "  --temperature F    Generation temperature (default: $TEMPERATURE)"
            echo "  --vllm-port N      vLLM server port (default: $VLLM_PORT)"
            echo "  --api-port N       Trajectory API port (default: $API_PORT)"
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
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate datasets peft trl bitsandbytes
    pip install vllm openai aiohttp
    echo "Installation complete."
}

kill_processes() {
    echo "Killing Python/vLLM processes..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "rlvr_atropos_full" 2>/dev/null || true
    pkill -9 -f "atropos_trainer" 2>/dev/null || true
    sleep 3
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo "Done."
}

wait_for_vllm() {
    echo "Waiting for vLLM server..."
    for i in {1..60}; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server ready!"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    echo ""
    echo "ERROR: vLLM server not ready after 2 minutes"
    return 1
}

# =============================================================================
# MAIN
# =============================================================================

echo ""
echo "============================================================"
echo "  FULL ATROPOS RLVR TRAINING PIPELINE"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Training:"
echo "  Steps: $TOTAL_STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Samples: $MAX_SAMPLES"
echo ""
echo "Generation:"
echo "  Group size: $GROUP_SIZE (rollouts per prompt)"
echo "  Temperature: $TEMPERATURE"
echo ""
echo "Ports:"
echo "  vLLM: $VLLM_PORT"
echo "  Trajectory API: $API_PORT"
echo ""

# Kill processes if requested
if [ "$KILL_PYTHON" = true ]; then
    kill_processes
fi

# Change to work directory
mkdir -p "$WORK_DIR"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Mode: vLLM only
if [ "$VLLM_ONLY" = true ]; then
    echo ""
    echo "============================================================"
    echo "  STARTING VLLM SERVER ONLY"
    echo "============================================================"
    echo ""

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 1536 \
        --gpu-memory-utilization 0.6

    exit 0
fi

# Mode: Trainer only (assumes vLLM is running)
if [ "$TRAINER_ONLY" = true ]; then
    echo ""
    echo "============================================================"
    echo "  STARTING TRAINER ONLY"
    echo "============================================================"
    echo ""

    wait_for_vllm || exit 1

    python "$PROJECT_ROOT/src/training/rlvr_atropos_full.py" \
        --model "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR" \
        --dataset "$DATASET_NAME" \
        --vllm-url "http://localhost:${VLLM_PORT}/v1" \
        --vllm-port "$VLLM_PORT" \
        --api-port "$API_PORT" \
        --total-steps "$TOTAL_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --max-samples "$MAX_SAMPLES" \
        --group-size "$GROUP_SIZE" \
        --temperature "$TEMPERATURE"

    exit 0
fi

# Full pipeline mode
echo ""
echo "============================================================"
echo "  STARTING FULL RLVR PIPELINE"
echo "============================================================"
echo ""

# Step 1: Start vLLM in background
echo "[1/2] Starting vLLM server on port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$VLLM_PORT" \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 1536 \
    --gpu-memory-utilization 0.5 \
    > "${OUTPUT_DIR}/vllm.log" 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# Wait for vLLM
wait_for_vllm || {
    echo "Failed to start vLLM. Check ${OUTPUT_DIR}/vllm.log"
    kill $VLLM_PID 2>/dev/null
    exit 1
}

# Step 2: Start full pipeline (API + Environment + Trainer)
echo ""
echo "[2/2] Starting RLVR pipeline..."
echo ""

python "$PROJECT_ROOT/src/training/rlvr_atropos_full.py" \
    --model "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --dataset "$DATASET_NAME" \
    --vllm-url "http://localhost:${VLLM_PORT}/v1" \
    --vllm-port "$VLLM_PORT" \
    --api-port "$API_PORT" \
    --total-steps "$TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --max-samples "$MAX_SAMPLES" \
    --group-size "$GROUP_SIZE" \
    --temperature "$TEMPERATURE"

# Cleanup
echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "  RLVR TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Output saved to: $OUTPUT_DIR"
echo ""
