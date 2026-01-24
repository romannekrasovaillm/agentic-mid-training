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
    train_file: str = "data/train.jsonl"
    validation_file: Optional[str] = "data/validation.jsonl"
    max_seq_length: int = 8192
    preprocessing_num_workers: int = 16


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


def prepare_dataset(data_args: DataArguments, tokenizer):
    """Подготовка датасета для обучения"""
    logger.info("Загрузка и подготовка датасета...")

    # Загрузка данных
    data_files = {"train": data_args.train_file}
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
    )

    def tokenize_function(examples):
        """Токенизация примеров"""
        # Поддержка разных форматов данных
        if "messages" in examples:
            # Chat format
            texts = []
            for messages in examples["messages"]:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(text)
        elif "text" in examples:
            texts = examples["text"]
        else:
            raise ValueError("Данные должны содержать 'messages' или 'text'")

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        return tokenized

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
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
        train_file=config["data"]["train_file"],
        validation_file=config["data"].get("validation_file"),
        max_seq_length=config["data"].get("max_seq_length", 8192),
        preprocessing_num_workers=config["data"].get("preprocessing_num_workers", 16),
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
