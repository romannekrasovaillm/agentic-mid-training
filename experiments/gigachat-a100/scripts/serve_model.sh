#!/bin/bash
# Скрипт запуска модели для инференса через vLLM
# Usage: ./serve_model.sh [model_path] [port]

set -e

MODEL_PATH="${1:-ai-sage/GigaChat3-10B-A1.8B}"
PORT="${2:-8000}"

echo "=========================================="
echo "GigaChat Model Server (vLLM)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"

# Настройка CUDA
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_DEEP_GEMM=0

# Запуск vLLM сервера с MTP (Multi-Token Prediction) для ускорения
vllm serve "${MODEL_PATH}" \
    --dtype auto \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "disable_padded_drafter_batch": false}' \
    --enable-prefix-caching \
    --trust-remote-code

# Пример запроса после запуска:
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "ai-sage/GigaChat3-10B-A1.8B",
#     "messages": [{"role": "user", "content": "Привет!"}],
#     "max_tokens": 100
#   }'
