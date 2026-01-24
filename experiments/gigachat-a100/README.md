# GigaChat Agentic Training Experiment

Эксперимент по агентному мид-трейнингу и пост-трейнингу модели GigaChat3-10B-A1.8B на NVIDIA A100.

## Модель

**GigaChat3-10B-A1.8B** — диалоговая модель на архитектуре Mixture-of-Experts (MoE):
- 10B общих параметров
- 1.8B активных параметров
- Multi-head Latent Attention (MLA)
- Multi-Token Prediction (MTP) — ускорение генерации до 40%
- Обучена на 20T токенов

## Структура эксперимента

```
gigachat-a100/
├── configs/
│   ├── mid_training.yaml         # Конфигурация mid-training
│   ├── post_training_dpo.yaml    # DPO post-training
│   ├── post_training_grpo.yaml   # GRPO post-training (RL)
│   ├── deepspeed_zero2.json      # DeepSpeed ZeRO-2
│   └── deepspeed_zero3.json      # DeepSpeed ZeRO-3
├── scripts/
│   ├── setup_environment.sh      # Установка окружения
│   ├── run_mid_training.sh       # Запуск mid-training
│   ├── run_post_training_dpo.sh  # Запуск DPO
│   ├── run_post_training_grpo.sh # Запуск GRPO
│   ├── serve_model.sh            # Запуск inference сервера
│   └── evaluate_model.sh         # Оценка модели
├── data/
│   └── samples/                  # Примеры данных
└── README.md
```

## Быстрый старт

### 0. Клонирование репозитория

```bash
# Клонирование репозитория
git clone https://github.com/romannekrasovaillm/agentic-mid-training.git
cd agentic-mid-training

# Переключение на ветку эксперимента (опционально)
git checkout claude/gigachat-agent-training-setup-EMEbE

# Или получение последних изменений (если репозиторий уже склонирован)
git pull origin main
```

### 1. Установка окружения

```bash
cd experiments/gigachat-a100
chmod +x scripts/*.sh
./scripts/setup_environment.sh
```

### 2. Подготовка данных

Поместите данные в формате JSONL в директорию `data/`:

**Для mid-training** (`data/train.jsonl`):
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Для DPO** (`data/preferences_train.jsonl`):
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

**Для GRPO** (`data/grpo_prompts.jsonl`):
```json
{"prompt": "..."}
```

### 3. Запуск обучения

#### Mid-Training (Continued Pre-Training)
```bash
./scripts/run_mid_training.sh
```

#### Post-Training (DPO)
```bash
./scripts/run_post_training_dpo.sh
```

#### Post-Training (GRPO)
```bash
./scripts/run_post_training_grpo.sh
```

### 4. Inference

```bash
./scripts/serve_model.sh ./outputs/gigachat-mid-training/final 8000
```

Пример запроса:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gigachat",
    "messages": [{"role": "user", "content": "Привет!"}],
    "max_tokens": 100
  }'
```

### 5. Оценка

```bash
./scripts/evaluate_model.sh ./outputs/gigachat-mid-training/final
```

## Фазы обучения

### Mid-Training

Цель: кристаллизация агентных навыков через траектории рассуждений.

**Data Mixing:**
- 45% — агентные трассы (tool use, reasoning chains)
- 25% — программный код
- 15% — математика и логика
- 10% — качественный текст
- 5% — function calling

**Параметры (A100 80GB):**
- Batch size: 4 × 8 (gradient accumulation) = 32
- Learning rate: 2e-5
- LoRA: r=64, alpha=128
- Sequence length: 8192

### Post-Training (DPO)

Цель: выравнивание модели на предпочтительное агентное поведение.

**Критерии "хорошего" ответа:**
- Правильный выбор инструмента
- Эффективные рассуждения
- Восстановление после ошибок
- Соответствие JSON формату

**Параметры:**
- Beta: 0.1
- Learning rate: 5e-7
- LoRA: r=32

### Post-Training (GRPO)

Цель: RL-оптимизация с автоматическими reward функциями.

**Reward функции:**
- Format compliance (JSON валидность)
- Safety (отсутствие опасных команд)
- Efficiency (оптимальная длина ответа)

## Требования

- NVIDIA A100 80GB
- CUDA 12.1+
- Python 3.10+
- ~200GB дискового пространства для модели и чекпоинтов

## Мониторинг

Логи обучения доступны через Wandb:
```bash
wandb login
# Проект: gigachat-agentic-midtraining
```

Или через TensorBoard:
```bash
tensorboard --logdir ./outputs/gigachat-mid-training/runs
```

## Troubleshooting

### OOM (Out of Memory)

1. Уменьшите `per_device_train_batch_size`
2. Увеличьте `gradient_accumulation_steps`
3. Включите DeepSpeed ZeRO-3 с offload

### Медленное обучение

1. Проверьте Flash Attention 2
2. Используйте `tf32=true`
3. Включите `gradient_checkpointing`

## Ссылки

- [GigaChat3-10B-A1.8B на HuggingFace](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B)
- [Статья на Habr](https://habr.com/en/companies/sberdevices/articles/968904/)
- [TRL Documentation](https://huggingface.co/docs/trl)
