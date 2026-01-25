#!/bin/bash
#
# Start Mid-Training (Continued Pre-Training) for GigaChat
#
# This script runs pure mid-training with next-token prediction loss
# on the Nemotron-Agentic dataset. NO SFT (no prompt masking).
#
# Usage:
#   ./scripts/start_mid_training.sh                     # Default settings
#   ./scripts/start_mid_training.sh --samples 5000      # Limit samples
#   ./scripts/start_mid_training.sh --epochs 3          # Multiple epochs
#

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME="${MODEL_NAME:-ai-sage/GigaChat3-10B-A1.8B-bf16}"
DATASET_NAME="${DATASET_NAME:-nvidia/Nemotron-Agentic-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/mid-training}"

# Training hyperparameters
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-1536}"

# LoRA
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"

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
        --samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch-size|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate|--lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora-r)
            LORA_R="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Mid-Training (Continued Pre-Training) for GigaChat"
            echo ""
            echo "Options:"
            echo "  --skip-install     Skip package installation"
            echo "  --kill             Kill Python processes before starting"
            echo "  --model NAME       Model name (default: $MODEL_NAME)"
            echo "  --dataset NAME     Dataset name (default: $DATASET_NAME)"
            echo "  --output-dir DIR   Output directory (default: $OUTPUT_DIR)"
            echo "  --samples N        Max dataset samples (default: $MAX_SAMPLES)"
            echo "  --batch-size N     Batch size (default: $BATCH_SIZE)"
            echo "  --epochs N         Number of epochs (default: $EPOCHS)"
            echo "  --learning-rate F  Learning rate (default: $LEARNING_RATE)"
            echo "  --lora-r N         LoRA rank (default: $LORA_R)"
            echo "  --lora-alpha N     LoRA alpha (default: $LORA_ALPHA)"
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
    echo "Installation complete."
}

kill_processes() {
    echo "Killing Python processes..."
    pkill -9 -f python || true
    sleep 3
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo "Done."
}

# =============================================================================
# MAIN
# =============================================================================

echo "============================================================"
echo "  MID-TRAINING (Next Token Prediction)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Output: $OUTPUT_DIR"
echo "  Samples: $MAX_SAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA"
echo ""
echo "Training method:"
echo "  Loss: Causal LM (all tokens)"
echo "  Labels = Input IDs (no prompt masking)"
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

echo "============================================================"
echo "  STARTING MID-TRAINING"
echo "============================================================"
echo ""

# Run mid-training
python "$PROJECT_ROOT/src/training/full_pipeline.py" \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --lora-r "$LORA_R" \
    --lora-alpha "$LORA_ALPHA" \
    --mid-training-only

echo ""
echo "============================================================"
echo "  MID-TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "To continue with RLVR:"
echo "  ./scripts/start_atropos_rlvr.sh --model $OUTPUT_DIR/mid-training-final"
