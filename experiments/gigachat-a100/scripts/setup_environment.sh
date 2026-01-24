#!/bin/bash
# Скрипт установки окружения для GigaChat агентного обучения на A100
# Usage: ./setup_environment.sh

set -e

echo "=========================================="
echo "GigaChat Agentic Training - Environment Setup"
echo "=========================================="

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Проверка GPU
check_gpu() {
    echo -e "${YELLOW}Проверка GPU...${NC}"
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}nvidia-smi не найден. Убедитесь, что NVIDIA драйверы установлены.${NC}"
        exit 1
    fi

    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    echo -e "${GREEN}Обнаружен GPU: ${GPU_INFO}${NC}"

    # Проверка что это A100
    if [[ $GPU_INFO == *"A100"* ]]; then
        echo -e "${GREEN}A100 подтвержден!${NC}"
    else
        echo -e "${YELLOW}Предупреждение: GPU не является A100. Конфигурации могут требовать настройки.${NC}"
    fi
}

# Создание виртуального окружения
setup_venv() {
    echo -e "${YELLOW}Создание виртуального окружения...${NC}"

    VENV_PATH="./venv"

    if [ -d "$VENV_PATH" ]; then
        echo -e "${YELLOW}Виртуальное окружение уже существует. Пропускаем создание.${NC}"
    else
        python3 -m venv $VENV_PATH
        echo -e "${GREEN}Виртуальное окружение создано: ${VENV_PATH}${NC}"
    fi

    source $VENV_PATH/bin/activate
    pip install --upgrade pip setuptools wheel
}

# Установка PyTorch с CUDA
install_pytorch() {
    echo -e "${YELLOW}Установка PyTorch с поддержкой CUDA...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# Установка основных зависимостей
install_dependencies() {
    echo -e "${YELLOW}Установка основных зависимостей...${NC}"

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
        packaging

    # DeepSpeed
    pip install deepspeed>=0.15.0

    # Flash Attention 2 - установка pre-built wheel
    echo -e "${YELLOW}Установка Flash Attention 2...${NC}"

    # Определяем версии Python и CUDA для выбора правильного wheel
    PYTHON_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:3])" 2>/dev/null || echo "121")

    echo "Python: ${PYTHON_VERSION}, CUDA: ${CUDA_VERSION}"

    # Пробуем установить pre-built wheel, если не получится - пропускаем
    if pip install flash-attn --no-build-isolation 2>/dev/null; then
        echo -e "${GREEN}Flash Attention 2 установлен!${NC}"
    else
        echo -e "${YELLOW}Не удалось установить flash-attn, пробуем альтернативный метод...${NC}"
        # Установка через pip с принудительной пересборкой
        pip install packaging wheel setuptools ninja
        if MAX_JOBS=4 FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE pip install flash-attn --no-build-isolation --force-reinstall 2>/dev/null; then
            echo -e "${GREEN}Flash Attention 2 установлен!${NC}"
        else
            echo -e "${YELLOW}Flash Attention 2 не установлен. Будет использоваться PyTorch SDPA.${NC}"
            echo -e "${YELLOW}Это не критично - SDPA обеспечивает хорошую производительность на A100.${NC}"
        fi
    fi

    echo -e "${GREEN}Основные зависимости установлены!${NC}"
}

# Установка vLLM для инференса
install_vllm() {
    echo -e "${YELLOW}Установка vLLM для быстрого инференса...${NC}"
    pip install vllm>=0.6.0
    echo -e "${GREEN}vLLM установлен!${NC}"
}

# Установка инструментов для оценки
install_evaluation_tools() {
    echo -e "${YELLOW}Установка инструментов оценки...${NC}"
    pip install lm-eval[api]==0.4.9.1
    echo -e "${GREEN}Инструменты оценки установлены!${NC}"
}

# Загрузка модели
download_model() {
    echo -e "${YELLOW}Загрузка модели GigaChat3-10B-A1.8B...${NC}"

    MODEL_PATH="./models/GigaChat3-10B-A1.8B"

    if [ -d "$MODEL_PATH" ]; then
        echo -e "${YELLOW}Модель уже загружена. Пропускаем.${NC}"
    else
        mkdir -p ./models
        python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ai-sage/GigaChat3-10B-A1.8B',
    local_dir='${MODEL_PATH}',
    local_dir_use_symlinks=False
)
print('Модель загружена успешно!')
"
    fi

    echo -e "${GREEN}Модель готова: ${MODEL_PATH}${NC}"
}

# Проверка установки
verify_installation() {
    echo -e "${YELLOW}Проверка установки...${NC}"

    python -c "
import torch
import transformers
import accelerate
import peft
import trl
import deepspeed

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Transformers version: {transformers.__version__}')
print(f'Accelerate version: {accelerate.__version__}')
print(f'PEFT version: {peft.__version__}')
print(f'TRL version: {trl.__version__}')
print(f'DeepSpeed version: {deepspeed.__version__}')
"

    echo -e "${GREEN}Проверка завершена!${NC}"
}

# Основной процесс
main() {
    check_gpu
    setup_venv
    install_pytorch
    install_dependencies
    install_vllm
    install_evaluation_tools

    # Опционально: загрузка модели
    read -p "Загрузить модель GigaChat3-10B-A1.8B? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        download_model
    fi

    verify_installation

    echo ""
    echo -e "${GREEN}=========================================="
    echo "Установка завершена!"
    echo "=========================================="
    echo -e "Активируйте окружение: source ./venv/bin/activate${NC}"
}

# Запуск
main "$@"
