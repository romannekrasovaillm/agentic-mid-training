#!/bin/bash
# Скрипт запуска Post-Training (DPO) для GigaChat
# Usage: ./run_post_training_dpo.sh [config_path]

set -e

CONFIG_PATH="${1:-configs/post_training_dpo.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "GigaChat Post-Training (DPO)"
echo "=========================================="
echo "Config: ${CONFIG_PATH}"

cd "$PROJECT_ROOT"

if [ -d "./venv" ]; then
    source ./venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export WANDB_PROJECT="gigachat-agentic-posttraining"

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "Запуск DPO Training..."

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    ../../../src/training/dpo_training.py \
    --config "${CONFIG_PATH}"

echo "=========================================="
echo "DPO Training завершен!"
echo "=========================================="
