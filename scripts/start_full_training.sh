#!/bin/bash
# ============================================================================
# GigaChat Agentic Training - Full Setup & Training Script
# ============================================================================
#
# Полный скрипт для запуска на новом сервере:
# 1. Клонирование репозитория
# 2. Установка зависимостей
# 3. Запуск vLLM сервера
# 4. Mid-Training + RLVR Post-Training
#
# Usage:
#   # Полная установка и обучение
#   curl -sSL https://raw.githubusercontent.com/.../start_full_training.sh | bash
#
#   # Или после клонирования
#   ./scripts/start_full_training.sh [--skip-install] [--mid-training-only] [--rlvr-only]
#
# ============================================================================

set -e

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Конфигурация
REPO_URL="https://github.com/YOUR_USERNAME/agentic-mid-training.git"
BRANCH="claude/gigachat-agent-training-setup-EMEbE"
WORK_DIR="${WORK_DIR:-/workspace/agentic-mid-training}"
VENV_DIR="${WORK_DIR}/experiments/gigachat-a100/venv"
MODEL_NAME="ai-sage/GigaChat3-10B-A1.8B-bf16"
VLLM_PORT=8000

# Параметры обучения
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/gigachat-pipeline}"

# Флаги
SKIP_INSTALL=false
MID_TRAINING_ONLY=false
RLVR_ONLY=false
USE_ATROPOS=false

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-install) SKIP_INSTALL=true; shift ;;
        --mid-training-only) MID_TRAINING_ONLY=true; shift ;;
        --rlvr-only) RLVR_ONLY=true; shift ;;
        --atropos) USE_ATROPOS=true; shift ;;
        --samples) MAX_SAMPLES="$2"; shift 2 ;;
        --batch) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done


# ============================================================================
# ФУНКЦИИ
# ============================================================================

log_header() {
    echo ""
    echo -e "${PURPLE}============================================================${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}============================================================${NC}"
    echo ""
}

log_info() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${GREEN}INFO${NC}  | $1"
}

log_warn() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${YELLOW}WARN${NC}  | $1"
}

log_error() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} ${RED}ERROR${NC} | $1"
}

check_gpu() {
    log_header "CHECKING GPU"

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. NVIDIA drivers not installed?"
        exit 1
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader)
    log_info "GPU detected:"
    echo "$GPU_INFO" | while read line; do
        log_info "  $line"
    done

    # Check memory
    FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE_MEM" -lt 40000 ]; then
        log_warn "Less than 40GB GPU memory free. May encounter OOM."
        log_info "Killing existing Python processes..."
        pkill -9 python 2>/dev/null || true
        sleep 2
    fi
}

clone_repo() {
    log_header "CLONING REPOSITORY"

    if [ -d "$WORK_DIR" ]; then
        log_info "Directory exists, updating..."
        cd "$WORK_DIR"
        git fetch origin "$BRANCH"
        git checkout "$BRANCH"
        git pull origin "$BRANCH"
    else
        log_info "Cloning from $REPO_URL..."
        git clone "$REPO_URL" "$WORK_DIR"
        cd "$WORK_DIR"
        git checkout "$BRANCH"
    fi

    log_info "Repository ready at $WORK_DIR"
}

setup_environment() {
    log_header "SETTING UP ENVIRONMENT"

    cd "$WORK_DIR/experiments/gigachat-a100"

    # Create venv if needed
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    source venv/bin/activate
    log_info "Virtual environment activated"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel

    # Install PyTorch
    log_info "Installing PyTorch with CUDA..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install main dependencies
    log_info "Installing training dependencies..."
    pip install \
        transformers>=4.45.0 \
        accelerate>=0.34.0 \
        datasets>=2.20.0 \
        peft>=0.12.0 \
        trl>=0.11.0 \
        bitsandbytes>=0.44.0 \
        safetensors>=0.4.0 \
        sentencepiece \
        protobuf \
        wandb \
        tensorboard \
        scipy \
        ninja \
        packaging \
        pyyaml \
        openai \
        aiohttp

    # Install vLLM
    log_info "Installing vLLM..."
    pip install vllm>=0.6.0

    # Install atroposlib
    log_info "Installing atroposlib..."
    pip install atroposlib || log_warn "atroposlib installation failed (optional)"

    log_info "Environment setup complete"
}

verify_installation() {
    log_header "VERIFYING INSTALLATION"

    python -c "
import torch
import transformers
import peft
import trl

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
"
}

start_vllm_server() {
    log_header "STARTING VLLM SERVER"

    # Check if already running
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        log_info "vLLM server already running on port ${VLLM_PORT}"
        return 0
    fi

    log_info "Starting vLLM server for ${MODEL_NAME}..."

    # Start in background
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_NAME" \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 2048 \
        > "${WORK_DIR}/vllm_server.log" 2>&1 &

    VLLM_PID=$!
    echo $VLLM_PID > "${WORK_DIR}/vllm.pid"

    log_info "vLLM server started with PID $VLLM_PID"
    log_info "Waiting for server to be ready..."

    # Wait for server
    for i in {1..60}; do
        if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
            log_info "vLLM server is ready!"
            return 0
        fi
        echo -n "."
        sleep 5
    done

    log_error "vLLM server failed to start. Check ${WORK_DIR}/vllm_server.log"
    exit 1
}

stop_vllm_server() {
    log_header "STOPPING VLLM SERVER"

    if [ -f "${WORK_DIR}/vllm.pid" ]; then
        PID=$(cat "${WORK_DIR}/vllm.pid")
        if kill -0 $PID 2>/dev/null; then
            log_info "Stopping vLLM server (PID $PID)..."
            kill $PID
            rm "${WORK_DIR}/vllm.pid"
        fi
    fi
}

run_mid_training() {
    log_header "STAGE 1: MID-TRAINING"

    cd "$WORK_DIR/experiments/gigachat-a100"
    source venv/bin/activate

    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_ALLOC_CONF="expandable_segments:True"
    export WANDB_MODE="offline"
    export TOKENIZERS_PARALLELISM="false"

    # Clear GPU
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    log_info "Starting Mid-Training..."
    log_info "  Samples: $MAX_SAMPLES"
    log_info "  Batch size: $BATCH_SIZE"
    log_info "  Epochs: $EPOCHS"

    python ../../src/training/full_pipeline.py \
        --config configs/mid_training.yaml \
        --output-dir "$OUTPUT_DIR" \
        --max-samples "$MAX_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --mid-training-only

    log_info "Mid-Training complete!"
}

run_rlvr_training() {
    log_header "STAGE 2: RLVR POST-TRAINING"

    cd "$WORK_DIR/experiments/gigachat-a100"
    source venv/bin/activate

    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_ALLOC_CONF="expandable_segments:True"
    export WANDB_MODE="offline"

    if [ "$USE_ATROPOS" = true ]; then
        log_info "Using full Atropos with vLLM server..."

        # Start vLLM if needed
        start_vllm_server

        python ../../src/training/atropos_rlvr.py \
            --config configs/post_training_rlvr.yaml \
            --output-dir "${OUTPUT_DIR}/atropos-rlvr" \
            --max-samples "$MAX_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --num-episodes "$EPOCHS" \
            --vllm-url "http://localhost:${VLLM_PORT}/v1"
    else
        log_info "Using simplified RLVR (no vLLM server)..."

        python ../../src/training/full_pipeline.py \
            --config configs/mid_training.yaml \
            --output-dir "$OUTPUT_DIR" \
            --max-samples "$MAX_SAMPLES" \
            --batch-size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --rlvr-only
    fi

    log_info "RLVR Training complete!"
}

run_full_pipeline() {
    log_header "FULL TRAINING PIPELINE"

    if [ "$RLVR_ONLY" = false ]; then
        run_mid_training
    fi

    if [ "$MID_TRAINING_ONLY" = false ]; then
        run_rlvr_training
    fi

    log_header "TRAINING COMPLETE"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Checkpoints:"
    ls -la "$OUTPUT_DIR" 2>/dev/null || true
}

print_summary() {
    log_header "TRAINING SUMMARY"

    echo -e "${GREEN}Configuration:${NC}"
    echo "  Model: $MODEL_NAME"
    echo "  Samples: $MAX_SAMPLES"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Epochs: $EPOCHS"
    echo "  Output: $OUTPUT_DIR"
    echo ""
    echo -e "${GREEN}Training stages:${NC}"
    [ "$RLVR_ONLY" = false ] && echo "  ✓ Mid-Training (SFT)"
    [ "$MID_TRAINING_ONLY" = false ] && echo "  ✓ RLVR Post-Training"
    [ "$USE_ATROPOS" = true ] && echo "    └─ Using full Atropos with vLLM"
    echo ""
}


# ============================================================================
# MAIN
# ============================================================================

main() {
    log_header "GIGACHAT AGENTIC TRAINING"
    echo -e "${CYAN}Start time: $(date)${NC}"
    echo ""

    # Check GPU
    check_gpu

    # Setup
    if [ "$SKIP_INSTALL" = false ]; then
        clone_repo
        setup_environment
        verify_installation
    else
        log_info "Skipping installation (--skip-install)"
        cd "$WORK_DIR/experiments/gigachat-a100"
        source venv/bin/activate
    fi

    # Print summary
    print_summary

    # Run training
    run_full_pipeline

    # Cleanup
    if [ "$USE_ATROPOS" = true ]; then
        stop_vllm_server
    fi

    echo ""
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${GREEN}  ALL DONE!${NC}"
    echo -e "${GREEN}============================================================${NC}"
    echo -e "${CYAN}End time: $(date)${NC}"
}

# Run
main "$@"
