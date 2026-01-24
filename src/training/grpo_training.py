#!/usr/bin/env python3
"""
GigaChat GRPO Post-Training Script
Group Relative Policy Optimization для агентного поведения
"""

import argparse
import logging
import re
from dataclasses import dataclass
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

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


# Reward функции для агентного поведения
class AgentRewardFunctions:
    """Коллекция reward функций для оценки агентного поведения"""

    def __init__(self, config: dict):
        self.config = config
        self.forbidden_patterns = config.get("reward_functions", {}).get(
            "safety", {}
        ).get("forbidden_patterns", [])

    def format_compliance_reward(self, response: str) -> float:
        """Награда за соответствие формату (JSON, tool calls)"""
        reward = 0.0

        # Проверка валидного JSON
        try:
            import json
            # Пытаемся найти JSON в ответе
            json_match = re.search(r'\{[^{}]*\}', response)
            if json_match:
                json.loads(json_match.group())
                reward += 0.2
        except (json.JSONDecodeError, AttributeError):
            pass

        # Проверка наличия tool call структуры
        if '"name"' in response and '"arguments"' in response:
            reward += 0.1

        # Проверка reasoning структуры
        if '<think>' in response or 'Reasoning:' in response:
            reward += 0.1

        return reward

    def safety_reward(self, response: str) -> float:
        """Штраф за небезопасные паттерны"""
        for pattern in self.forbidden_patterns:
            if pattern.lower() in response.lower():
                return -1.0
        return 0.0

    def efficiency_reward(self, response: str, optimal_length: int = 500) -> float:
        """Награда за эффективность (не слишком длинный/короткий ответ)"""
        length = len(response)
        if length < 50:
            return -0.3  # Слишком короткий
        elif length > optimal_length * 3:
            return -0.2  # Слишком длинный
        else:
            return 0.1

    def compute_reward(self, prompts: list, responses: list) -> list:
        """Вычисление общей награды для списка ответов"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            total_reward = 0.0

            # Формат
            total_reward += self.format_compliance_reward(response)

            # Безопасность
            total_reward += self.safety_reward(response)

            # Эффективность
            total_reward += self.efficiency_reward(response)

            rewards.append(total_reward)

        return rewards


def load_prompts_dataset(config: dict):
    """Загрузка датасета с промптами для GRPO"""
    logger.info("Загрузка датасета промптов...")

    data_config = config["data"]
    dataset = load_dataset("json", data_files={"train": data_config["train_file"]})

    def format_prompt(example):
        """Форматирование промпта"""
        if "messages" in example:
            # Chat формат - используем последнее сообщение пользователя
            prompt = example["messages"]
        elif "prompt" in example:
            prompt = example["prompt"]
        else:
            raise ValueError("Данные должны содержать 'messages' или 'prompt'")

        return {"prompt": prompt}

    formatted_dataset = dataset.map(
        format_prompt,
        remove_columns=[c for c in dataset["train"].column_names if c != "prompt"],
        desc="Форматирование промптов",
    )

    return formatted_dataset


def main():
    parser = argparse.ArgumentParser(description="GigaChat GRPO Training")
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
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Загрузка датасета
    dataset = load_prompts_dataset(config)

    # Reward функции
    reward_funcs = AgentRewardFunctions(config)

    # GRPO конфигурация
    grpo_config = config.get("grpo", {})
    training_config = config["training"]

    grpo_training_args = GRPOConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config.get("num_train_epochs", 1),
        max_steps=training_config.get("max_steps", 1000),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        optim=training_config.get("optim", "adamw_torch_fused"),
        learning_rate=training_config.get("learning_rate", 1e-6),
        weight_decay=training_config.get("weight_decay", 0.0),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        bf16=training_config.get("bf16", True),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        logging_steps=training_config.get("logging_steps", 5),
        save_steps=training_config.get("save_steps", 100),
        # GRPO специфичные параметры
        num_generations=grpo_config.get("num_generations", 4),
        max_completion_length=grpo_config.get("max_new_tokens", 2048),
        beta=grpo_config.get("beta", 0.04),
        report_to=["wandb"] if config.get("wandb") else ["tensorboard"],
        run_name=config.get("wandb", {}).get("run_name", "gigachat-grpo"),
    )

    # GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_training_args,
        train_dataset=dataset["train"],
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_funcs.compute_reward,
    )

    # Обучение
    logger.info("Запуск GRPO обучения...")
    train_result = trainer.train()

    # Сохранение
    logger.info("Сохранение модели...")
    trainer.save_model()
    trainer.save_state()

    # Метрики
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("GRPO обучение завершено!")


if __name__ == "__main__":
    main()
