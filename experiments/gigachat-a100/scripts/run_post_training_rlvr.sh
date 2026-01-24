#!/bin/bash
# Скрипт запуска Post-Training (RLVR) для GigaChat
# RLVR = Reinforcement Learning with Verifiable Rewards
# Based on Atropos framework with Interleaved Reasoning
# Usage: ./run_post_training_rlvr.sh [config_path]

set -e

CONFIG_PATH="${1:-configs/post_training_rlvr.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "GigaChat Post-Training (RLVR)"
echo "Reinforcement Learning with Verifiable Rewards"
echo "=========================================="
echo "Config: ${CONFIG_PATH}"

cd "$PROJECT_ROOT"

if [ -d "./venv" ]; then
    source ./venv/bin/activate
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_PROJECT="gigachat-rlvr"

# Очистка GPU кэша
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

echo "Запуск RLVR Training..."

# Запуск напрямую (без accelerate для упрощения)
python ../../src/training/rlvr_training.py \
    --config "${CONFIG_PATH}"

echo "=========================================="
echo "RLVR Training завершен!"
echo "=========================================="
