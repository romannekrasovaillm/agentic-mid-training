#!/bin/bash
# Полный пайплайн обучения GigaChat
# Mid-Training + RLVR Post-Training
# С детальным логированием

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "GigaChat Full Training Pipeline"
echo "Mid-Training + RLVR Post-Training"
echo "=========================================="
echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"

cd "$PROJECT_ROOT"

# Активация окружения
if [ -d "./venv" ]; then
    source ./venv/bin/activate
    echo "Virtual environment activated"
fi

# Настройки GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="offline"
export TOKENIZERS_PARALLELISM="false"

# Очистка GPU кэша
echo "Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache(); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')" 2>/dev/null || true

# Параметры обучения (можно переопределить через аргументы)
MAX_SAMPLES="${1:-100}"
BATCH_SIZE="${2:-2}"
EPOCHS="${3:-1}"
OUTPUT_DIR="${4:-./outputs/gigachat-pipeline}"

echo ""
echo "Training parameters:"
echo "  Max samples: ${MAX_SAMPLES}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

# Создание директории для выходных данных
mkdir -p "${OUTPUT_DIR}"

# Запуск пайплайна
echo "Starting training pipeline..."
echo ""

python ../../src/training/full_pipeline.py \
    --config configs/mid_training.yaml \
    --output-dir "${OUTPUT_DIR}" \
    --max-samples "${MAX_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}"

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "End time: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
