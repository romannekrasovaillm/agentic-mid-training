# SWE-bench Evaluation для LoRA-адаптированных моделей

Инструменты для оценки LoRA-адаптированных языковых моделей на бенчмарке SWE-bench_Verified с использованием [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) от Princeton.

## Обзор

```
swe_bench_eval/
├── models/
│   ├── lora_server.py        # OpenAI-совместимый сервер (Transformers)
│   └── vllm_lora_server.py   # vLLM сервер с поддержкой LoRA
├── configs/
│   ├── swe_verified.yaml     # Конфигурация бенчмарка
│   └── litellm_registry.json # Регистрация моделей для LiteLLM
├── scripts/
│   ├── run_swe_verified.sh   # Запуск полного бенчмарка
│   └── run_comparison.sh     # A/B сравнение моделей
└── requirements.txt
```

## Быстрый старт

### 1. Установка зависимостей

```bash
# Основные зависимости
pip install -r swe_bench_eval/requirements.txt

# mini-swe-agent
pip install mini-swe-agent
# или из git:
pip install git+https://github.com/SWE-agent/mini-swe-agent.git
```

### 2. Подготовка модели

Убедитесь, что у вас есть:
- **Базовая модель** (например, `ai-sage/GigaChat3-10B-A1.8B-bf16`)
- **LoRA адаптер** (директория с `adapter_config.json` и весами)

### 3. Запуск бенчмарка

```bash
cd swe_bench_eval

# Полный бенчмарк SWE-Verified
./scripts/run_swe_verified.sh \
    --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
    --lora-path /path/to/your/lora-adapter \
    --output-dir ./outputs/my-experiment

# Быстрый тест (10 инстансов)
./scripts/run_swe_verified.sh \
    --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
    --lora-path /path/to/your/lora-adapter \
    --max-instances 10
```

### 4. A/B сравнение

Сравнение baseline LoRA vs RL-обученной версии:

```bash
./scripts/run_comparison.sh \
    --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
    --lora-baseline /path/to/baseline-lora \
    --lora-trained /path/to/rl-trained-lora \
    --output-dir ./outputs/comparison \
    --max-instances 50
```

## Архитектура

### mini-swe-agent + LiteLLM

mini-swe-agent использует LiteLLM для абстракции моделей. Наш сервер предоставляет OpenAI-совместимый API:

```
[mini-swe-agent] --> [LiteLLM] --> [lora_server.py/vllm_lora_server.py] --> [Model + LoRA]
```

### Режимы сервера

1. **vLLM (рекомендуется)** - высокопроизводительный inference
   ```bash
   python models/vllm_lora_server.py \
       --base-model <model> \
       --lora-path <path>
   ```

2. **Transformers** - для отладки или когда vLLM недоступен
   ```bash
   python models/lora_server.py \
       --base-model <model> \
       --lora-path <path>
   ```

### Multi-LoRA (A/B тестирование)

vLLM сервер поддерживает несколько LoRA адаптеров одновременно:

```bash
python models/vllm_lora_server.py \
    --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
    --lora-modules "baseline:/path/to/baseline,trained:/path/to/trained"
```

Затем используйте разные модели через API:
```bash
curl http://localhost:8080/v1/chat/completions \
    -d '{"model": "baseline", "messages": [...]}'

curl http://localhost:8080/v1/chat/completions \
    -d '{"model": "trained", "messages": [...]}'
```

## Конфигурация

### swe_verified.yaml

```yaml
model:
  base_model: "ai-sage/GigaChat3-10B-A1.8B-bf16"
  lora_path: "./checkpoints/swe-agent-lora"
  serving_mode: "vllm"  # или "transformers"

benchmark:
  dataset: "princeton-nlp/SWE-bench_Verified"
  max_instances: null  # null = полный бенчмарк

agent:
  max_turns: 30
  temperature: 0.0
  sandbox: "docker"
```

### Квантизация (для GPU < 80GB)

```bash
# 4-bit quantization
python models/lora_server.py \
    --base-model <model> \
    --lora-path <path> \
    --load-in-4bit

# 8-bit quantization
python models/lora_server.py \
    --base-model <model> \
    --lora-path <path> \
    --load-in-8bit
```

## Результаты

После запуска бенчмарка результаты сохраняются в:

```
outputs/
└── my-experiment/
    ├── trajectories/     # Траектории агента (JSON)
    ├── predictions/      # Патчи (diff файлы)
    ├── logs/
    │   ├── server.log    # Логи сервера модели
    │   ├── agent.log     # Логи mini-swe-agent
    │   └── evaluation.log
    └── results/          # Метрики оценки
```

### Просмотр траекторий

```bash
mini trajectory-browser ./outputs/my-experiment/trajectories
```

### Ручная оценка

```bash
swe-bench-evaluate \
    --predictions_path ./outputs/my-experiment/predictions \
    --swe_bench_tasks princeton-nlp/SWE-bench_Verified
```

## Требования

- **GPU**: NVIDIA A100 80GB (рекомендуется) или A100 40GB с квантизацией
- **Docker**: Для sandbox-окружения
- **Python**: 3.10+

## Troubleshooting

### OOM при загрузке модели

```bash
# Используйте 4-bit квантизацию
./scripts/run_swe_verified.sh \
    --base-model <model> \
    --lora-path <path> \
    --no-vllm  # Transformers с quantization

# Или уменьшите GPU memory utilization
./scripts/run_swe_verified.sh \
    --gpu-memory 0.7
```

### Сервер не запускается

1. Проверьте логи: `cat outputs/*/logs/server.log`
2. Убедитесь, что порт 8080 свободен: `lsof -i :8080`
3. Проверьте CUDA: `nvidia-smi`

### mini-swe-agent не находит модель

```bash
# Установите переменные окружения
export OPENAI_API_BASE="http://localhost:8080/v1"
export OPENAI_API_KEY="not-needed"
```

## Связь с тренировочным пайплайном

Этот модуль интегрируется с тренировочным пайплайном из ветки `gigachat-agent-training-setup`:

1. **Mid-training** → базовые агентные способности
2. **GRPO/RLVR post-training** → оптимизация через RL
3. **SWE-bench evaluation** (этот модуль) → проверка на реальных задачах

```
[Base Model] + [LoRA from training] --> [SWE-bench Eval] --> [Metrics]
```
