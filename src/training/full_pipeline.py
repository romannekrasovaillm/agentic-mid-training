#!/usr/bin/env python3
"""
GigaChat Full Training Pipeline
Mid-Training + RLVR Post-Training

Полный пайплайн с детальным логированием
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Настройка логирования
def setup_logging(output_dir: str, name: str = "pipeline"):
    """Настройка детального логирования"""
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Формат с timestamp и уровнем
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__), log_file


def log_gpu_memory(logger, prefix: str = ""):
    """Логирование использования GPU памяти"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"{prefix}GPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, max={max_allocated:.2f}GB")


def log_system_info(logger):
    """Логирование системной информации"""
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)

    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f}GB")

    logger.info("=" * 60)


@dataclass
class PipelineConfig:
    """Конфигурация пайплайна"""
    # Model
    model_name: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"
    dtype: str = "bfloat16"

    # Directories
    output_dir: str = "./outputs/gigachat-pipeline"
    cache_dir: str = "./cache"

    # Mid-training
    mid_training_epochs: int = 1
    mid_training_lr: float = 2e-5
    mid_training_batch_size: int = 2
    mid_training_grad_accum: int = 8
    mid_training_max_samples: int = 1000

    # RLVR
    rlvr_epochs: int = 1
    rlvr_lr: float = 5e-7
    rlvr_batch_size: int = 2
    rlvr_num_generations: int = 4
    rlvr_max_samples: int = 500

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Data
    dataset_name: str = "nvidia/Nemotron-Agentic-v1"
    dataset_split: str = "tool_calling"
    max_seq_length: int = 2048

    # Training
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Logging
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 50


class AgenticDataset(Dataset):
    """Dataset для agentic training"""

    def __init__(self, samples: list, tokenizer, max_length: int = 2048):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Форматируем как chat
        if "messages" in sample:
            messages = sample["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)
        elif "messages_json" in sample:
            messages = json.loads(sample["messages_json"])
        else:
            messages = [{"role": "user", "content": str(sample)}]

        # Применяем chat template если есть
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception:
                text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        else:
            text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        # Токенизация
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0).clone()
        }


def load_dataset_samples(config: PipelineConfig, logger, max_samples: int = None) -> list:
    """Загрузка датасета"""
    logger.info(f"Загрузка датасета: {config.dataset_name}")

    from huggingface_hub import hf_hub_download

    file_path = hf_hub_download(
        repo_id=config.dataset_name,
        filename=f"data/{config.dataset_split}.jsonl",
        repo_type="dataset",
        cache_dir=config.cache_dir
    )

    logger.info(f"Файл загружен: {file_path}")

    samples = []
    max_samples = max_samples or config.mid_training_max_samples

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                example = json.loads(line.strip())
                if "messages" in example:
                    samples.append({"messages": example["messages"]})
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга строки {i}: {e}")
                continue

            if (i + 1) % 1000 == 0:
                logger.debug(f"Загружено {i + 1} примеров...")

    logger.info(f"Загружено {len(samples)} примеров из датасета")
    return samples


def load_model_and_tokenizer(config: PipelineConfig, logger):
    """Загрузка модели и токенизатора"""
    logger.info("=" * 60)
    logger.info("ЗАГРУЗКА МОДЕЛИ")
    logger.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Модель: {config.model_name}")
    logger.info(f"Dtype: {config.dtype}")

    # Tokenizer
    logger.info("Загрузка токенизатора...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Установлен pad_token = eos_token: {tokenizer.pad_token}")

    logger.info(f"Токенизатор загружен за {time.time() - start_time:.1f}s")
    logger.info(f"Vocab size: {tokenizer.vocab_size}")

    # Model
    logger.info("Загрузка модели...")
    start_time = time.time()
    log_gpu_memory(logger, "До загрузки модели: ")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
        cache_dir=config.cache_dir
    )

    logger.info(f"Модель загружена за {time.time() - start_time:.1f}s")
    log_gpu_memory(logger, "После загрузки модели: ")

    # Параметры модели
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Всего параметров: {total_params / 1e9:.2f}B")

    return model, tokenizer


def apply_lora(model, config: PipelineConfig, logger):
    """Применение LoRA"""
    logger.info("=" * 60)
    logger.info("ПРИМЕНЕНИЕ LoRA")
    logger.info("=" * 60)

    from peft import LoraConfig, get_peft_model, TaskType

    logger.info(f"LoRA r: {config.lora_r}")
    logger.info(f"LoRA alpha: {config.lora_alpha}")
    logger.info(f"Target modules: {config.lora_target_modules}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    # Статистика
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Trainable params: {trainable_params / 1e6:.2f}M")
    logger.info(f"Total params: {total_params / 1e9:.2f}B")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    log_gpu_memory(logger, "После LoRA: ")

    return model


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    config: PipelineConfig,
    logger,
    epoch: int,
    stage: str = "mid-training"
):
    """Обучение одной эпохи"""
    model.train()

    total_loss = 0.0
    num_steps = 0
    grad_accum_steps = config.mid_training_grad_accum

    optimizer.zero_grad()

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1} [{stage}]",
        leave=True,
        file=sys.stdout
    )

    for step, batch in enumerate(progress_bar):
        # Move to device
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / grad_accum_steps

        # Backward pass
        loss.backward()

        total_loss += outputs.loss.item()
        num_steps += 1

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = total_loss / num_steps
            current_lr = scheduler.get_last_lr()[0]

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # Детальное логирование
            if (step + 1) % (grad_accum_steps * config.logging_steps) == 0:
                logger.info(
                    f"[{stage}] Step {step + 1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                )
                log_gpu_memory(logger, "  ")

    # Final step if needed
    if (step + 1) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_steps, 1)
    logger.info(f"[{stage}] Epoch {epoch + 1} завершена | Avg Loss: {avg_loss:.4f}")

    return avg_loss


def run_mid_training(model, tokenizer, config: PipelineConfig, logger):
    """Запуск Mid-Training"""
    logger.info("=" * 60)
    logger.info("MID-TRAINING")
    logger.info("=" * 60)

    # Загрузка данных
    samples = load_dataset_samples(config, logger, config.mid_training_max_samples)

    dataset = AgenticDataset(samples, tokenizer, config.max_seq_length)
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.mid_training_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
    logger.info(f"Batches per epoch: {len(dataloader)}")

    # Optimizer
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(
        model.parameters(),
        lr=config.mid_training_lr,
        weight_decay=config.weight_decay
    )

    total_steps = len(dataloader) * config.mid_training_epochs // config.mid_training_grad_accum
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logger.info(f"Total optimization steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {config.mid_training_lr}")

    # Training loop
    for epoch in range(config.mid_training_epochs):
        logger.info(f"\n{'='*40}")
        logger.info(f"EPOCH {epoch + 1}/{config.mid_training_epochs}")
        logger.info(f"{'='*40}")

        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler,
            config, logger, epoch, "mid-training"
        )

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            save_path = os.path.join(config.output_dir, f"mid-training-epoch-{epoch + 1}")
            logger.info(f"Сохранение checkpoint: {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    logger.info("Mid-Training завершен!")
    return model


def run_rlvr_training(model, tokenizer, config: PipelineConfig, logger):
    """Запуск RLVR Post-Training"""
    logger.info("=" * 60)
    logger.info("RLVR POST-TRAINING")
    logger.info("=" * 60)

    from .rlvr_training import (
        compute_reward,
        format_tool_use_prompt,
        extract_boxed_answer,
        execute_tools,
        extract_tool_calls
    )

    # Загрузка данных
    samples = load_dataset_samples(config, logger, config.rlvr_max_samples)

    # Подготовка промптов
    rlvr_samples = []
    for sample in samples:
        messages = sample.get("messages", [])
        question = None
        for msg in messages:
            if msg.get("role") == "user":
                question = msg.get("content", "")
                break
        if question:
            prompt = format_tool_use_prompt(question)
            rlvr_samples.append({"prompt": prompt, "question": question})

    logger.info(f"RLVR samples: {len(rlvr_samples)}")

    # Optimizer с низким LR
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=config.rlvr_lr)

    model.train()
    total_reward = 0.0
    num_samples = 0

    progress_bar = tqdm(rlvr_samples, desc="RLVR Training", file=sys.stdout)

    for i, sample in enumerate(progress_bar):
        prompt = sample["prompt"]

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Compute reward
        reward, info = compute_reward(response)
        total_reward += reward
        num_samples += 1

        # Policy gradient update (simplified)
        if reward != 0:
            # Get log probs
            full_text = prompt + response
            full_inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            full_inputs = {k: v.to(model.device) for k, v in full_inputs.items()}

            model_outputs = model(**full_inputs, labels=full_inputs["input_ids"])
            loss = -reward * model_outputs.loss  # Reward-weighted loss

            loss.backward()

            if (i + 1) % config.mid_training_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

        avg_reward = total_reward / num_samples
        progress_bar.set_postfix({
            'reward': f'{reward:.2f}',
            'avg_reward': f'{avg_reward:.3f}',
            'tools': info.get('tool_calls_count', 0)
        })

        if (i + 1) % 10 == 0:
            logger.info(
                f"[RLVR] Step {i + 1}/{len(rlvr_samples)} | "
                f"Reward: {reward:.2f} | Avg: {avg_reward:.3f} | "
                f"Tools: {info.get('tool_calls_count', 0)}"
            )

    # Final optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Save final model
    save_path = os.path.join(config.output_dir, "rlvr-final")
    logger.info(f"Сохранение финальной модели: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    logger.info(f"RLVR Training завершен! Avg Reward: {total_reward / max(num_samples, 1):.3f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="GigaChat Full Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--output-dir", type=str, default="./outputs/gigachat-pipeline")
    parser.add_argument("--mid-training-only", action="store_true", help="Run only mid-training")
    parser.add_argument("--rlvr-only", action="store_true", help="Run only RLVR")
    parser.add_argument("--max-samples", type=int, default=100, help="Max training samples")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    args = parser.parse_args()

    # Setup config
    config = PipelineConfig()
    config.output_dir = args.output_dir
    config.mid_training_max_samples = args.max_samples
    config.rlvr_max_samples = args.max_samples
    config.mid_training_batch_size = args.batch_size
    config.mid_training_epochs = args.epochs

    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            # Override from yaml
            if "model" in yaml_config:
                config.model_name = yaml_config["model"].get("name", config.model_name)
                config.dtype = yaml_config["model"].get("dtype", config.dtype)
            if "training" in yaml_config:
                tc = yaml_config["training"]
                config.mid_training_epochs = tc.get("num_train_epochs", config.mid_training_epochs)
                config.mid_training_lr = tc.get("learning_rate", config.mid_training_lr)
                config.mid_training_batch_size = tc.get("per_device_train_batch_size", config.mid_training_batch_size)
                config.mid_training_grad_accum = tc.get("gradient_accumulation_steps", config.mid_training_grad_accum)
            if "data" in yaml_config:
                dc = yaml_config["data"]
                config.dataset_name = dc.get("dataset_name", config.dataset_name)
                config.mid_training_max_samples = dc.get("max_train_samples", config.mid_training_max_samples)
                config.max_seq_length = dc.get("max_seq_length", config.max_seq_length)

    # Setup logging
    os.makedirs(config.output_dir, exist_ok=True)
    logger, log_file = setup_logging(config.output_dir, "pipeline")

    logger.info("=" * 60)
    logger.info("GIGACHAT TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Output dir: {config.output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Max samples: {config.mid_training_max_samples}")
    logger.info(f"Batch size: {config.mid_training_batch_size}")
    logger.info(f"Epochs: {config.mid_training_epochs}")

    log_system_info(logger)

    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(config, logger)

        # Apply LoRA
        model = apply_lora(model, config, logger)

        # Training stages
        if not args.rlvr_only:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 1: MID-TRAINING")
            logger.info("=" * 60)
            model = run_mid_training(model, tokenizer, config, logger)

        if not args.mid_training_only:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 2: RLVR POST-TRAINING")
            logger.info("=" * 60)
            try:
                model = run_rlvr_training(model, tokenizer, config, logger)
            except ImportError as e:
                logger.warning(f"RLVR import error: {e}")
                logger.warning("Skipping RLVR training")

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Final model saved to: {config.output_dir}")

    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        raise


if __name__ == "__main__":
    main()
