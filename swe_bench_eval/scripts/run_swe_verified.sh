#!/bin/bash
#
# Run SWE-Verified Benchmark with LoRA-adapted Model
#
# This script:
# 1. Starts the LoRA model server
# 2. Runs mini-swe-agent on SWE-bench_Verified
# 3. Evaluates predictions
#
# Usage:
#   ./run_swe_verified.sh --base-model <model> --lora-path <path> [options]
#
# Example:
#   ./run_swe_verified.sh \
#       --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
#       --lora-path ./checkpoints/swe-agent-lora \
#       --output-dir ./outputs/my-experiment
#

set -e

# =============================================================================
# Default Configuration
# =============================================================================
BASE_MODEL=""
LORA_PATH=""
OUTPUT_DIR="./outputs/swe-verified-$(date +%Y%m%d-%H%M%S)"
SERVER_PORT=8080
MAX_INSTANCES=""
SANDBOX="docker"
MAX_TURNS=30
TEMPERATURE=0.0
USE_VLLM=true
GPU_MEMORY_UTIL=0.9
TENSOR_PARALLEL=1
MAX_MODEL_LEN=8192
SKIP_EVAL=false

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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --sandbox)
            SANDBOX="$2"
            shift 2
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --no-vllm)
            USE_VLLM=false
            shift
            ;;
        --gpu-memory)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --help)
            echo "Usage: $0 --base-model <model> --lora-path <path> [options]"
            echo ""
            echo "Required:"
            echo "  --base-model       Base model path or HuggingFace repo ID"
            echo "  --lora-path        Path to LoRA adapter directory"
            echo ""
            echo "Optional:"
            echo "  --output-dir       Output directory (default: ./outputs/swe-verified-<timestamp>)"
            echo "  --port             Server port (default: 8080)"
            echo "  --max-instances    Limit number of instances (for testing)"
            echo "  --sandbox          Sandbox type: docker, podman, singularity (default: docker)"
            echo "  --max-turns        Maximum agent turns per instance (default: 30)"
            echo "  --temperature      Generation temperature (default: 0.0)"
            echo "  --no-vllm          Use transformers instead of vLLM"
            echo "  --gpu-memory       GPU memory utilization (default: 0.9)"
            echo "  --tensor-parallel  Tensor parallel size (default: 1)"
            echo "  --max-model-len    Maximum model context length (default: 8192)"
            echo "  --skip-eval        Skip evaluation after prediction"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Arguments
# =============================================================================
if [[ -z "$BASE_MODEL" ]]; then
    echo "Error: --base-model is required"
    exit 1
fi

if [[ -z "$LORA_PATH" ]]; then
    echo "Error: --lora-path is required"
    exit 1
fi

if [[ ! -d "$LORA_PATH" ]]; then
    echo "Error: LoRA path does not exist: $LORA_PATH"
    exit 1
fi

# =============================================================================
# Setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUTPUT_DIR"/{trajectories,predictions,logs,results}

echo "=============================================="
echo "SWE-Verified Benchmark with LoRA"
echo "=============================================="
echo "Base Model:     $BASE_MODEL"
echo "LoRA Path:      $LORA_PATH"
echo "Output Dir:     $OUTPUT_DIR"
echo "Server Port:    $SERVER_PORT"
echo "Sandbox:        $SANDBOX"
echo "Max Turns:      $MAX_TURNS"
echo "Temperature:    $TEMPERATURE"
echo "Use vLLM:       $USE_VLLM"
echo "=============================================="

# =============================================================================
# Start Model Server
# =============================================================================
echo ""
echo "[1/3] Starting model server..."

SERVER_LOG="$OUTPUT_DIR/logs/server.log"
SERVER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [[ -n "$SERVER_PID" ]]; then
        echo "Stopping model server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if $USE_VLLM; then
    python "$PROJECT_DIR/models/vllm_lora_server.py" \
        --base-model "$BASE_MODEL" \
        --lora-path "$LORA_PATH" \
        --port "$SERVER_PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --tensor-parallel-size "$TENSOR_PARALLEL" \
        --max-model-len "$MAX_MODEL_LEN" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
else
    python "$PROJECT_DIR/models/lora_server.py" \
        --base-model "$BASE_MODEL" \
        --lora-path "$LORA_PATH" \
        --port "$SERVER_PORT" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
fi

echo "Server PID: $SERVER_PID"
echo "Server log: $SERVER_LOG"

# Wait for server to be ready
echo "Waiting for server to start..."
MAX_WAIT=300
WAIT_COUNT=0
while ! curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; do
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [[ $WAIT_COUNT -ge $MAX_WAIT ]]; then
        echo "Error: Server failed to start within $MAX_WAIT seconds"
        echo "Check logs: $SERVER_LOG"
        exit 1
    fi
    echo "  Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done
echo "Server is ready!"

# =============================================================================
# Run SWE-bench with mini-swe-agent
# =============================================================================
echo ""
echo "[2/3] Running SWE-bench_Verified..."

# Set environment for LiteLLM
export LITELLM_MODEL_REGISTRY_PATH="$PROJECT_DIR/configs/litellm_registry.json"
export OPENAI_API_KEY="not-needed"
export OPENAI_API_BASE="http://localhost:$SERVER_PORT/v1"

# Build mini-swe-agent command
MINI_CMD="mini swebench"
MINI_CMD="$MINI_CMD --model openai/default"
MINI_CMD="$MINI_CMD --sandbox $SANDBOX"
MINI_CMD="$MINI_CMD --output $OUTPUT_DIR/trajectories"
MINI_CMD="$MINI_CMD --max-turns $MAX_TURNS"
MINI_CMD="$MINI_CMD --temperature $TEMPERATURE"

if [[ -n "$MAX_INSTANCES" ]]; then
    MINI_CMD="$MINI_CMD --max-instances $MAX_INSTANCES"
fi

echo "Running: $MINI_CMD"
eval "$MINI_CMD" 2>&1 | tee "$OUTPUT_DIR/logs/agent.log"

# =============================================================================
# Evaluate Predictions
# =============================================================================
if ! $SKIP_EVAL; then
    echo ""
    echo "[3/3] Evaluating predictions..."

    # Copy predictions to standard location
    if [[ -d "$OUTPUT_DIR/trajectories" ]]; then
        find "$OUTPUT_DIR/trajectories" -name "*.patch" -exec cp {} "$OUTPUT_DIR/predictions/" \;
    fi

    # Run SWE-bench evaluation if available
    if command -v swe-bench-evaluate &> /dev/null; then
        swe-bench-evaluate \
            --predictions_path "$OUTPUT_DIR/predictions" \
            --swe_bench_tasks "princeton-nlp/SWE-bench_Verified" \
            --output_dir "$OUTPUT_DIR/results" \
            2>&1 | tee "$OUTPUT_DIR/logs/evaluation.log"
    else
        echo "SWE-bench evaluation tool not found."
        echo "To install: pip install swe-bench"
        echo ""
        echo "Predictions saved to: $OUTPUT_DIR/predictions"
    fi
else
    echo ""
    echo "[3/3] Skipping evaluation (--skip-eval)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "Next steps:"
echo "  1. View trajectories: mini trajectory-browser $OUTPUT_DIR/trajectories"
echo "  2. Manual evaluation: swe-bench-evaluate --predictions_path $OUTPUT_DIR/predictions"
echo "=============================================="
