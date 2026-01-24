#!/bin/bash
# Скрипт запуска Mid-Training для GigaChat
# Usage: ./run_mid_training.sh [config_path]

set -e

# Конфигурация по умолчанию
CONFIG_PATH="${1:-configs/mid_training.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "GigaChat Agentic Mid-Training"
echo "=========================================="
echo "Config: ${CONFIG_PATH}"
echo "Project root: ${PROJECT_ROOT}"

# Переход в директорию проекта
cd "$PROJECT_ROOT"

# Активация окружения
if [ -d "./venv" ]; then
    source ./venv/bin/activate
fi

# Настройка CUDA
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF="max_split_size_mb:512"
export TRANSFORMERS_VERBOSITY=info

# Настройка Wandb (опционально)
export WANDB_PROJECT="gigachat-agentic-midtraining"

# Очистка кэша перед запуском
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Запуск обучения
echo "Запуск Mid-Training..."

accelerate launch \
    --mixed_precision bf16 \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    ../../src/training/mid_training.py \
    --config "${CONFIG_PATH}"

echo "=========================================="
echo "Mid-Training завершен!"
echo "=========================================="
