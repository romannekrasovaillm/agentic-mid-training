#!/usr/bin/env python3
"""
GigaChat Agentic Mid-Training Script
Continued Pre-Training с фокусом на агентные способности
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def is_flash_attn_available():
    """Проверка доступности flash attention"""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


@dataclass
class ModelArguments:
    """Аргументы модели"""
    model_name_or_path: str = "ai-sage/GigaChat3-10B-A1.8B"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"  # sdpa, flash_attention_2, eager


@dataclass
class DataArguments:
    """Аргументы данных"""
    # HuggingFace Hub датасет
    dataset_name: Optional[str] = None  # e.g. "nvidia/Nemotron-Agentic-v1"
    dataset_config: Optional[str] = None  # e.g. "tool_calling" или "interactive_agent"
    # Локальные файлы (если dataset_name не указан)
    train_file: Optional[str] = "data/train.jsonl"
    validation_file: Optional[str] = None
    # Общие параметры
    max_seq_length: int = 4096
    preprocessing_num_workers: int = 16
    max_train_samples: Optional[int] = None  # Ограничение выборки для тестов


@dataclass
class LoraArguments:
    """Аргументы LoRA"""
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )


def load_config(config_path: str) -> dict:
    """Загрузка конфигурации из YAML файла"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Преобразование строки в torch.dtype"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def setup_model_and_tokenizer(model_args: ModelArguments, lora_args: LoraArguments):
    """Инициализация модели и токенизатора"""
    logger.info(f"Загрузка модели: {model_args.model_name_or_path}")

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Модель
    torch_dtype = get_torch_dtype(model_args.torch_dtype)

    # Выбор реализации attention
    attn_impl = model_args.attn_implementation
    if attn_impl == "flash_attention_2" and not is_flash_attn_available():
        logger.warning("Flash Attention 2 недоступен, используем SDPA")
        attn_impl = "sdpa"

    logger.info(f"Используем attention implementation: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=attn_impl,
        device_map="auto",
    )

    # LoRA настройка
    if lora_args.use_lora:
        logger.info("Применение LoRA конфигурации...")

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def format_nemotron_messages(messages: list) -> list:
    """Преобразование Nemotron формата в стандартный chat формат"""
    formatted = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Добавляем reasoning если есть
        reasoning = msg.get("reasoning_content", "")
        if reasoning:
            content = f"<think>\n{reasoning}\n</think>\n{content}"

        # Добавляем tool_calls если есть
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            import json
            for tc in tool_calls:
                if isinstance(tc, dict) and "function" in tc:
                    func = tc["function"]
                    tool_str = json.dumps({
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", {})
                    }, ensure_ascii=False)
                    content = f"{content}\n{tool_str}" if content else tool_str

        formatted.append({"role": role, "content": content})

    return formatted


def prepare_dataset(data_args: DataArguments, tokenizer):
    """Подготовка датасета для обучения"""
    logger.info("Загрузка и подготовка датасета...")

    # Загрузка из HuggingFace Hub или локальных файлов
    if data_args.dataset_name:
        logger.info(f"Загрузка датасета из HuggingFace Hub: {data_args.dataset_name}")
        if data_args.dataset_config:
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config,
            )
        else:
            raw_datasets = load_dataset(data_args.dataset_name)

        # Создаём validation split если его нет
        if "validation" not in raw_datasets and "train" in raw_datasets:
            split = raw_datasets["train"].train_test_split(test_size=0.02, seed=42)
            raw_datasets = {
                "train": split["train"],
                "validation": split["test"]
            }
    else:
        # Локальные файлы
        data_files = {"train": data_args.train_file}
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file

        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
        )

    # Ограничение выборки для тестов
    if data_args.max_train_samples:
        logger.info(f"Ограничение train выборки до {data_args.max_train_samples} примеров")
        raw_datasets["train"] = raw_datasets["train"].select(
            range(min(data_args.max_train_samples, len(raw_datasets["train"])))
        )

    logger.info(f"Train samples: {len(raw_datasets['train'])}")
    if "validation" in raw_datasets:
        logger.info(f"Validation samples: {len(raw_datasets['validation'])}")

    def tokenize_function(examples):
        """Токенизация примеров"""
        texts = []

        for i in range(len(examples["messages"])):
            messages = examples["messages"][i]

            # Преобразуем Nemotron формат если нужно
            if messages and isinstance(messages[0], dict):
                if "tool_calls" in messages[0] or "reasoning_content" in messages[0]:
                    messages = format_nemotron_messages(messages)

            # Применяем chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                # Fallback: простая конкатенация
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        return tokenized

    # Получаем колонки для удаления
    columns_to_remove = raw_datasets["train"].column_names

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=columns_to_remove,
        desc="Токенизация датасета",
    )

    return tokenized_datasets


def main():
    parser = argparse.ArgumentParser(description="GigaChat Mid-Training")
    parser.add_argument("--config", type=str, required=True, help="Путь к конфигурации")
    args = parser.parse_args()

    # Загрузка конфигурации
    config = load_config(args.config)

    # Установка seed
    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Создание аргументов
    model_args = ModelArguments(
        model_name_or_path=config["model"]["name"],
        torch_dtype=config["model"].get("dtype", "bfloat16"),
        trust_remote_code=config["model"].get("trust_remote_code", True),
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
    )

    data_args = DataArguments(
        dataset_name=config["data"].get("dataset_name"),
        dataset_config=config["data"].get("dataset_config"),
        train_file=config["data"].get("train_file"),
        validation_file=config["data"].get("validation_file"),
        max_seq_length=config["data"].get("max_seq_length", 4096),
        preprocessing_num_workers=config["data"].get("preprocessing_num_workers", 16),
        max_train_samples=config["data"].get("max_train_samples"),
    )

    lora_config = config.get("lora", {})
    lora_args = LoraArguments(
        use_lora=lora_config.get("enabled", True),
        lora_r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        lora_target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]),
    )

    # Инициализация модели
    model, tokenizer = setup_model_and_tokenizer(model_args, lora_args)

    # Подготовка данных
    tokenized_datasets = prepare_dataset(data_args, tokenizer)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        optim=training_config.get("optim", "adamw_torch_fused"),
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.1),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        tf32=training_config.get("tf32", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        logging_steps=training_config.get("logging_steps", 10),
        logging_first_step=training_config.get("logging_first_step", True),
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=training_config.get("save_total_limit", 3),
        eval_strategy=training_config.get("evaluation_strategy", "steps"),
        eval_steps=training_config.get("eval_steps", 500),
        deepspeed=training_config.get("deepspeed"),
        report_to=["wandb"] if config.get("wandb") else ["tensorboard"],
        run_name=config.get("wandb", {}).get("run_name", "gigachat-mid-training"),
    )

    # Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Обучение
    logger.info("Запуск обучения...")
    train_result = trainer.train()

    # Сохранение
    logger.info("Сохранение модели...")
    trainer.save_model()
    trainer.save_state()

    # Метрики
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Обучение завершено!")


if __name__ == "__main__":
    main()
