#!/bin/bash
# Скрипт оценки модели
# Usage: ./evaluate_model.sh [model_path] [tasks]

set -e

MODEL_PATH="${1:-./outputs/gigachat-mid-training/final}"
TASKS="${2:-hellaswag,arc_easy,mmlu}"
OUTPUT_DIR="./evaluation_results"

echo "=========================================="
echo "GigaChat Model Evaluation"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Tasks: ${TASKS}"

mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=0

# Оценка с помощью lm-eval
lm_eval \
    --model hf \
    --model_args pretrained="${MODEL_PATH}",dtype=bfloat16,trust_remote_code=True \
    --tasks "${TASKS}" \
    --batch_size auto \
    --output_path "${OUTPUT_DIR}" \
    --log_samples

# Агентная оценка (кастомные метрики)
echo "Запуск агентной оценки..."

python ../../src/evaluation/agent_eval.py \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}/agent_eval"

echo "=========================================="
echo "Оценка завершена!"
echo "Результаты: ${OUTPUT_DIR}"
echo "=========================================="
