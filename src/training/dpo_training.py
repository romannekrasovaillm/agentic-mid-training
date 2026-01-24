#!/usr/bin/env python3
"""
GigaChat DPO Post-Training Script
Direct Preference Optimization для агентного поведения
"""

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Загрузка конфигурации из YAML"""
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


def load_preference_dataset(config: dict, tokenizer):
    """Загрузка датасета с предпочтениями"""
    logger.info("Загрузка preference датасета...")

    data_config = config["data"]
    data_files = {"train": data_config["train_file"]}
    if data_config.get("validation_file"):
        data_files["validation"] = data_config["validation_file"]

    dataset = load_dataset("json", data_files=data_files)

    def format_example(example):
        """Форматирование примера для DPO"""
        # Ожидаемый формат: {"prompt": ..., "chosen": ..., "rejected": ...}

        if "messages" in example:
            # Если данные в chat формате
            prompt_messages = example["messages"][:-1]  # Все кроме последнего
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            chosen = example.get("chosen", example["messages"][-1]["content"])
            rejected = example.get("rejected", "")
        else:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    formatted_dataset = dataset.map(
        format_example,
        remove_columns=dataset["train"].column_names,
        desc="Форматирование примеров",
    )

    return formatted_dataset


def main():
    parser = argparse.ArgumentParser(description="GigaChat DPO Training")
    parser.add_argument("--config", type=str, required=True, help="Путь к конфигурации")
    args = parser.parse_args()

    # Загрузка конфигурации
    config = load_config(args.config)
    set_seed(42)

    # Загрузка модели
    model_config = config["model"]
    torch_dtype = get_torch_dtype(model_config.get("dtype", "bfloat16"))

    logger.info(f"Загрузка модели: {model_config['name']}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", True),
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # Загрузка reference модели
    ref_model_config = config.get("ref_model", {})
    if ref_model_config.get("name"):
        logger.info(f"Загрузка reference модели: {ref_model_config['name']}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_config["name"],
            torch_dtype=get_torch_dtype(ref_model_config.get("dtype", "bfloat16")),
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        ref_model = None

    # Токенизатор
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA конфигурация
    lora_config = config.get("lora", {})
    peft_config = None
    if lora_config.get("enabled", True):
        peft_config = LoraConfig(
            r=lora_config.get("r", 32),
            lora_alpha=lora_config.get("lora_alpha", 64),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj"
            ]),
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Загрузка датасета
    dataset = load_preference_dataset(config, tokenizer)

    # DPO конфигурация
    dpo_config = config.get("dpo", {})
    training_config = config["training"]

    dpo_training_args = DPOConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config.get("num_train_epochs", 1),
        max_steps=training_config.get("max_steps", 2000),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        optim=training_config.get("optim", "adamw_torch_fused"),
        learning_rate=training_config.get("learning_rate", 5e-7),
        weight_decay=training_config.get("weight_decay", 0.0),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        bf16=training_config.get("bf16", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 200),
        eval_strategy="steps",
        eval_steps=training_config.get("eval_steps", 200),
        deepspeed=training_config.get("deepspeed"),
        # DPO специфичные параметры
        beta=dpo_config.get("beta", 0.1),
        loss_type=dpo_config.get("loss_type", "sigmoid"),
        max_length=config["data"].get("max_length", 4096),
        max_prompt_length=config["data"].get("max_prompt_length", 2048),
        report_to=["wandb"] if config.get("wandb") else ["tensorboard"],
        run_name=config.get("wandb", {}).get("run_name", "gigachat-dpo"),
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Обучение
    logger.info("Запуск DPO обучения...")
    train_result = trainer.train()

    # Сохранение
    logger.info("Сохранение модели...")
    trainer.save_model()
    trainer.save_state()

    # Метрики
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("DPO обучение завершено!")


if __name__ == "__main__":
    main()
