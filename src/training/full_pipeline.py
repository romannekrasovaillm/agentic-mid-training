#!/usr/bin/env python3
"""
GigaChat Full Training Pipeline
Mid-Training + RLVR Post-Training

Полный пайплайн с детальным логированием и метриками CLM
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================================
# TERMINAL COLORS AND FORMATTING
# ============================================================================

class Colors:
    """ANSI color codes for beautiful terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @staticmethod
    def colored(text: str, color: str) -> str:
        return f"{color}{text}{Colors.END}"

    @staticmethod
    def bold(text: str) -> str:
        return f"{Colors.BOLD}{text}{Colors.END}"

    @staticmethod
    def header(text: str) -> str:
        return f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}"

    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.GREEN}{text}{Colors.END}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.YELLOW}{text}{Colors.END}"

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{text}{Colors.END}"

    @staticmethod
    def metric(name: str, value: str) -> str:
        return f"{Colors.DIM}{name}:{Colors.END} {Colors.BOLD}{value}{Colors.END}"


def print_box(title: str, content: list, width: int = 70):
    """Print a beautiful box with title and content"""
    border = "═" * width
    print(f"\n{Colors.CYAN}╔{border}╗{Colors.END}")
    print(f"{Colors.CYAN}║{Colors.END} {Colors.BOLD}{title.center(width - 2)}{Colors.END} {Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}╠{border}╣{Colors.END}")
    for line in content:
        padding = width - 2 - len(line.replace('\033[', '').split('m')[-1] if '\033[' in line else line)
        # Strip ANSI codes for length calculation
        import re
        clean_line = re.sub(r'\033\[[0-9;]*m', '', line)
        padding = width - 2 - len(clean_line)
        print(f"{Colors.CYAN}║{Colors.END} {line}{' ' * max(0, padding)} {Colors.CYAN}║{Colors.END}")
    print(f"{Colors.CYAN}╚{border}╝{Colors.END}\n")


def print_separator(char: str = "─", width: int = 70):
    """Print a separator line"""
    print(f"{Colors.DIM}{char * width}{Colors.END}")


def format_number(num: float, precision: int = 4) -> str:
    """Format number with color based on magnitude"""
    if abs(num) < 0.0001:
        return f"{num:.2e}"
    elif abs(num) < 1:
        return f"{num:.{precision}f}"
    elif abs(num) < 1000:
        return f"{num:.{min(precision, 2)}f}"
    else:
        return f"{num:.2e}"


def format_time(seconds: float) -> str:
    """Format time duration"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


# ============================================================================
# METRICS TRACKER
# ============================================================================

class MetricsTracker:
    """Track and compute training metrics for Causal Language Modeling"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.losses = deque(maxlen=self.window_size)
        self.token_accuracies = deque(maxlen=self.window_size)
        self.gradient_norms = deque(maxlen=self.window_size)
        self.tokens_per_second = deque(maxlen=self.window_size)
        self.batch_sizes = deque(maxlen=self.window_size)

        self.total_tokens = 0
        self.total_correct_tokens = 0
        self.total_loss = 0.0
        self.total_steps = 0
        self.start_time = time.time()

    def update(
        self,
        loss: float,
        num_tokens: int,
        correct_tokens: int = 0,
        gradient_norm: float = 0.0,
        batch_time: float = 0.0,
        batch_size: int = 1
    ):
        """Update metrics with new batch results"""
        self.losses.append(loss)
        self.total_loss += loss
        self.total_steps += 1

        self.total_tokens += num_tokens
        self.total_correct_tokens += correct_tokens

        if correct_tokens > 0:
            self.token_accuracies.append(correct_tokens / max(num_tokens, 1))

        if gradient_norm > 0:
            self.gradient_norms.append(gradient_norm)

        if batch_time > 0:
            self.tokens_per_second.append(num_tokens / batch_time)

        self.batch_sizes.append(batch_size)

    @property
    def avg_loss(self) -> float:
        """Average loss over window"""
        return sum(self.losses) / max(len(self.losses), 1)

    @property
    def perplexity(self) -> float:
        """Perplexity = exp(avg_loss)"""
        return math.exp(min(self.avg_loss, 20))  # Clip to avoid overflow

    @property
    def bits_per_byte(self) -> float:
        """Bits per byte ≈ loss / ln(2)"""
        return self.avg_loss / math.log(2)

    @property
    def token_accuracy(self) -> float:
        """Average token accuracy over window"""
        if not self.token_accuracies:
            return 0.0
        return sum(self.token_accuracies) / len(self.token_accuracies)

    @property
    def global_token_accuracy(self) -> float:
        """Global token accuracy"""
        if self.total_tokens == 0:
            return 0.0
        return self.total_correct_tokens / self.total_tokens

    @property
    def avg_gradient_norm(self) -> float:
        """Average gradient norm over window"""
        if not self.gradient_norms:
            return 0.0
        return sum(self.gradient_norms) / len(self.gradient_norms)

    @property
    def throughput(self) -> float:
        """Tokens per second"""
        if not self.tokens_per_second:
            elapsed = time.time() - self.start_time
            return self.total_tokens / max(elapsed, 1)
        return sum(self.tokens_per_second) / len(self.tokens_per_second)

    @property
    def global_avg_loss(self) -> float:
        """Global average loss"""
        return self.total_loss / max(self.total_steps, 1)

    def get_summary(self) -> Dict[str, float]:
        """Get all metrics as dict"""
        return {
            "loss": self.avg_loss,
            "perplexity": self.perplexity,
            "bits_per_byte": self.bits_per_byte,
            "token_accuracy": self.token_accuracy,
            "gradient_norm": self.avg_gradient_norm,
            "throughput_tps": self.throughput,
            "total_tokens": self.total_tokens,
            "global_loss": self.global_avg_loss,
        }

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


def clear_gpu_memory(logger):
    """Полная очистка GPU памяти между стадиями"""
    import gc

    logger.info("=" * 40)
    logger.info("CLEARING GPU MEMORY")
    logger.info("=" * 40)

    log_gpu_memory(logger, "Before cleanup: ")

    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Force garbage collection
    gc.collect()

    # Additional CUDA synchronization
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    log_gpu_memory(logger, "After cleanup: ")
    logger.info("GPU memory cleared successfully")


def log_system_info(logger):
    """Логирование системной информации"""

    # Collect system info
    system_info = []

    # Python and PyTorch
    system_info.append(f"Python: {Colors.BOLD}{sys.version.split()[0]}{Colors.END}")
    system_info.append(f"PyTorch: {Colors.BOLD}{torch.__version__}{Colors.END}")

    # CUDA
    if torch.cuda.is_available():
        system_info.append(f"CUDA: {Colors.BOLD}{torch.version.cuda}{Colors.END}")
        system_info.append("")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_name = props.name
            gpu_mem = props.total_memory / 1e9

            # Highlight H100
            if "H100" in gpu_name:
                gpu_str = f"{Colors.GREEN}{gpu_name}{Colors.END}"
            else:
                gpu_str = gpu_name

            system_info.append(f"GPU {i}: {gpu_str} ({gpu_mem:.0f}GB)")

        # Compute capability
        cc = torch.cuda.get_device_capability(0)
        system_info.append(f"Compute Capability: {cc[0]}.{cc[1]}")
    else:
        system_info.append(f"{Colors.RED}CUDA: Not available{Colors.END}")

    system_info.append("")

    # Flash Attention
    try:
        import flash_attn
        fa_version = flash_attn.__version__
        system_info.append(f"Flash Attention: {Colors.GREEN}✓ v{fa_version}{Colors.END}")
    except ImportError:
        system_info.append(f"Flash Attention: {Colors.YELLOW}✗ Not installed{Colors.END}")

    # Transformers
    try:
        import transformers
        system_info.append(f"Transformers: {transformers.__version__}")
    except ImportError:
        pass

    # PEFT
    try:
        import peft
        system_info.append(f"PEFT: {peft.__version__}")
    except ImportError:
        pass

    # BitsAndBytes
    try:
        import bitsandbytes
        system_info.append(f"BitsAndBytes: {bitsandbytes.__version__}")
    except ImportError:
        pass

    # Print beautiful box
    print_box("SYSTEM INFORMATION", system_info, width=60)

    # Also log to file (without colors)
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}, {props.total_memory / 1e9:.1f}GB")

    # Flash Attention check
    try:
        import flash_attn
        logger.info(f"Flash Attention: {flash_attn.__version__}")
    except ImportError:
        logger.info("Flash Attention: Not installed")

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

    # Mid-training (оптимизировано для H100 80GB с gradient checkpointing)
    mid_training_epochs: int = 1
    mid_training_lr: float = 2e-5
    mid_training_batch_size: int = 2  # Уменьшено из-за OOM
    mid_training_grad_accum: int = 8  # Компенсируем меньший batch
    mid_training_max_samples: int = 1000
    gradient_checkpointing: bool = True  # Экономит память

    # RLVR
    rlvr_epochs: int = 1
    rlvr_lr: float = 5e-7
    rlvr_batch_size: int = 2
    rlvr_num_generations: int = 4
    rlvr_max_samples: int = 500
    rlvr_max_turns: int = 3  # Max tool-use turns per rollout

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
    max_seq_length: int = 1024  # Уменьшено для экономии памяти
    dataset_split_seed: int = 42  # Seed для разделения датасета
    rlvr_data_fraction: float = 0.5  # Доля данных для RLVR (mid-training берёт остаток)

    # Training
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Logging & Checkpoints
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 50
    save_total_limit: int = 3  # Хранить только последние N чекпойнтов


def cleanup_checkpoints(output_dir: str, prefix: str, keep_last: int, logger):
    """
    Удаление старых чекпойнтов, оставляя только последние keep_last

    Args:
        output_dir: Директория с чекпойнтами
        prefix: Префикс имени чекпойнта (например, 'mid-training-epoch', 'checkpoint-step')
        keep_last: Сколько последних чекпойнтов оставить
        logger: Логгер
    """
    import glob
    import shutil

    # Найти все чекпойнты с данным префиксом
    pattern = os.path.join(output_dir, f"{prefix}*")
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_last:
        return

    # Сортируем по времени модификации (старые первые)
    checkpoints_with_time = []
    for ckpt in checkpoints:
        if os.path.isdir(ckpt):
            mtime = os.path.getmtime(ckpt)
            checkpoints_with_time.append((ckpt, mtime))

    checkpoints_with_time.sort(key=lambda x: x[1])

    # Удаляем старые, оставляя keep_last последних
    to_delete = checkpoints_with_time[:-keep_last] if keep_last > 0 else checkpoints_with_time

    for ckpt_path, _ in to_delete:
        try:
            shutil.rmtree(ckpt_path)
            logger.info(f"Удалён старый чекпойнт: {ckpt_path}")
        except Exception as e:
            logger.warning(f"Не удалось удалить {ckpt_path}: {e}")

    remaining = len(checkpoints) - len(to_delete)
    logger.info(f"Очистка чекпойнтов: удалено {len(to_delete)}, осталось {remaining}")


def get_checkpoint_size(checkpoint_dir: str) -> str:
    """Получить размер чекпойнта в человекочитаемом формате"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(checkpoint_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)

    # Конвертируем в читаемый формат
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024:
            return f"{total_size:.1f}{unit}"
        total_size /= 1024
    return f"{total_size:.1f}TB"


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


def load_dataset_samples(config: PipelineConfig, logger, max_samples: int = None, for_rlvr: bool = False) -> list:
    """
    Загрузка датасета с разделением для mid-training и RLVR.

    Dataset split to avoid memorization:
    - Mid-training: first (1 - rlvr_data_fraction) portion
    - RLVR: last (rlvr_data_fraction) portion

    Args:
        config: Pipeline configuration
        logger: Logger instance
        max_samples: Maximum samples to return
        for_rlvr: If True, return RLVR portion; if False, return mid-training portion
    """
    logger.info(f"Загрузка датасета: {config.dataset_name}")
    logger.info(f"Dataset split seed: {config.dataset_split_seed}")
    logger.info(f"RLVR data fraction: {config.rlvr_data_fraction}")
    logger.info(f"Loading for: {'RLVR' if for_rlvr else 'Mid-Training'}")

    from huggingface_hub import hf_hub_download
    import random

    file_path = hf_hub_download(
        repo_id=config.dataset_name,
        filename=f"data/{config.dataset_split}.jsonl",
        repo_type="dataset",
        cache_dir=config.cache_dir
    )

    logger.info(f"Файл загружен: {file_path}")

    # Load all samples first
    all_samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line.strip())
                if "messages" in example:
                    all_samples.append({"messages": example["messages"]})
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга строки {i}: {e}")
                continue

            if (i + 1) % 10000 == 0:
                logger.debug(f"Загружено {i + 1} примеров...")

    logger.info(f"Всего примеров в датасете: {len(all_samples)}")

    # Shuffle with fixed seed for reproducibility
    random.seed(config.dataset_split_seed)
    random.shuffle(all_samples)

    # Split point
    split_point = int(len(all_samples) * (1 - config.rlvr_data_fraction))

    if for_rlvr:
        # RLVR gets the second portion
        samples = all_samples[split_point:]
        logger.info(f"RLVR: взято {len(samples)} примеров (после индекса {split_point})")
    else:
        # Mid-training gets the first portion
        samples = all_samples[:split_point]
        logger.info(f"Mid-Training: взято {len(samples)} примеров (до индекса {split_point})")

    # Limit to max_samples
    max_samples = max_samples or (config.rlvr_max_samples if for_rlvr else config.mid_training_max_samples)
    if len(samples) > max_samples:
        samples = samples[:max_samples]
        logger.info(f"Ограничено до {max_samples} примеров")

    logger.info(f"Итого загружено: {len(samples)} примеров")
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

    # Выбор реализации attention (Flash Attention 2 для H100)
    attn_impl = "flash_attention_2"
    try:
        # Проверяем доступность Flash Attention 2
        import flash_attn
        logger.info(f"Flash Attention version: {flash_attn.__version__}")
        logger.info(Colors.success("✓ Flash Attention 2 доступен"))
    except ImportError:
        logger.warning(Colors.warning("⚠ Flash Attention 2 не установлен, используем SDPA"))
        attn_impl = "sdpa"

    # Принудительно загружаем всё на GPU (H100 80GB)
    # device_map="auto" может выгружать на CPU, что замедляет обучение
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map={"": 0},  # Всё на GPU 0
        cache_dir=config.cache_dir,
        low_cpu_mem_usage=True,
    )

    logger.info(f"Attention implementation: {Colors.bold(attn_impl)}")

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

    # Gradient checkpointing для экономии памяти
    if config.gradient_checkpointing:
        logger.info("Включение gradient checkpointing...")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Статистика
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Trainable params: {trainable_params / 1e6:.2f}M")
    logger.info(f"Total params: {total_params / 1e9:.2f}B")
    logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    log_gpu_memory(logger, "После LoRA: ")

    return model


def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    """
    Compute token-level accuracy (top-1).

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target labels [batch, seq_len]

    Returns:
        (correct_tokens, total_tokens)
    """
    # Shift for causal LM: predict next token
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Get predictions
    predictions = shift_logits.argmax(dim=-1)

    # Mask padding tokens (usually -100)
    mask = shift_labels != -100

    correct = ((predictions == shift_labels) & mask).sum().item()
    total = mask.sum().item()

    return correct, total


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
    """
    Обучение одной эпохи с расширенными метриками CLM.

    Метрики:
    - Loss: Cross-entropy loss
    - Perplexity: exp(loss)
    - BPB: Bits per byte
    - Token Accuracy: % правильно предсказанных токенов
    - Gradient Norm: Норма градиентов
    - Throughput: Токенов в секунду
    """
    model.train()

    # Initialize metrics tracker
    metrics = MetricsTracker(window_size=100)
    grad_accum_steps = config.mid_training_grad_accum

    optimizer.zero_grad()

    # Beautiful progress bar
    progress_bar = tqdm(
        dataloader,
        desc=f"{Colors.CYAN}Epoch {epoch + 1}{Colors.END} [{stage}]",
        leave=True,
        file=sys.stdout,
        bar_format="{l_bar}{bar:30}{r_bar}",
        ncols=120
    )

    epoch_start_time = time.time()
    batch_start_time = time.time()

    for step, batch in enumerate(progress_bar):
        # Move to device
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Count tokens (excluding padding)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            num_tokens = attention_mask.sum().item()
        else:
            num_tokens = batch["input_ids"].numel()

        # Forward pass with autocast for mixed precision
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps

        # Compute token accuracy
        with torch.no_grad():
            correct_tokens, total_tokens = compute_token_accuracy(
                outputs.logits, batch["labels"]
            )

        # Backward pass
        loss.backward()

        # Calculate batch time
        batch_time = time.time() - batch_start_time
        batch_start_time = time.time()

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0:
            # Compute gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            ).item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update metrics
            metrics.update(
                loss=outputs.loss.item(),
                num_tokens=num_tokens,
                correct_tokens=correct_tokens,
                gradient_norm=grad_norm,
                batch_time=batch_time * grad_accum_steps,
                batch_size=batch["input_ids"].shape[0]
            )

            current_lr = scheduler.get_last_lr()[0]

            # Update progress bar with metrics
            progress_bar.set_postfix_str(
                f"loss={Colors.BOLD}{metrics.avg_loss:.4f}{Colors.END} | "
                f"ppl={Colors.CYAN}{metrics.perplexity:.2f}{Colors.END} | "
                f"acc={Colors.GREEN}{metrics.token_accuracy*100:.1f}%{Colors.END} | "
                f"∇={metrics.avg_gradient_norm:.2f} | "
                f"lr={current_lr:.2e} | "
                f"{metrics.throughput:.0f} tok/s"
            )

            # Detailed logging at intervals
            if (step + 1) % (grad_accum_steps * config.logging_steps) == 0:
                opt_step = (step + 1) // grad_accum_steps
                total_opt_steps = len(dataloader) // grad_accum_steps

                # Calculate ETA
                elapsed = time.time() - epoch_start_time
                steps_done = step + 1
                steps_remaining = len(dataloader) - steps_done
                eta = (elapsed / steps_done) * steps_remaining if steps_done > 0 else 0

                # Print detailed metrics
                print()  # New line for clean output
                print_separator("─", 80)
                print(
                    f"  {Colors.BOLD}Step {opt_step}/{total_opt_steps}{Colors.END} "
                    f"({Colors.DIM}{(step + 1) / len(dataloader) * 100:.1f}%{Colors.END}) "
                    f"│ ETA: {Colors.CYAN}{format_time(eta)}{Colors.END}"
                )
                print()

                # Metrics table
                metrics_str = [
                    f"  {Colors.metric('Loss', format_number(metrics.avg_loss))}",
                    f"  {Colors.metric('Perplexity', format_number(metrics.perplexity, 2))}",
                    f"  {Colors.metric('Bits/Byte', format_number(metrics.bits_per_byte, 3))}",
                    f"  {Colors.metric('Token Acc', f'{metrics.token_accuracy*100:.2f}%')}",
                    f"  {Colors.metric('Grad Norm', format_number(metrics.avg_gradient_norm, 3))}",
                    f"  {Colors.metric('Learning Rate', f'{current_lr:.2e}')}",
                    f"  {Colors.metric('Throughput', f'{metrics.throughput:.0f} tokens/sec')}",
                    f"  {Colors.metric('Total Tokens', f'{metrics.total_tokens:,}')}",
                ]
                for m in metrics_str:
                    print(m)

                # GPU memory
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    max_alloc = torch.cuda.max_memory_allocated() / 1e9
                    print()
                    print(
                        f"  {Colors.DIM}GPU:{Colors.END} "
                        f"{allocated:.1f}GB allocated / "
                        f"{reserved:.1f}GB reserved / "
                        f"{max_alloc:.1f}GB peak"
                    )

                print_separator("─", 80)
                print()

                # Also log to file
                logger.info(
                    f"[{stage}] Step {opt_step}/{total_opt_steps} | "
                    f"Loss: {metrics.avg_loss:.4f} | "
                    f"PPL: {metrics.perplexity:.2f} | "
                    f"Acc: {metrics.token_accuracy*100:.2f}% | "
                    f"∇: {metrics.avg_gradient_norm:.3f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Throughput: {metrics.throughput:.0f} tok/s"
                )
        else:
            # Still update metrics for non-optimization steps
            metrics.update(
                loss=outputs.loss.item(),
                num_tokens=num_tokens,
                correct_tokens=correct_tokens,
                batch_time=batch_time
            )

    # Final step if needed
    if (step + 1) % grad_accum_steps != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        ).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Epoch summary
    epoch_time = time.time() - epoch_start_time
    summary = metrics.get_summary()

    print()
    print_box(
        f"EPOCH {epoch + 1} COMPLETE",
        [
            f"Duration: {Colors.CYAN}{format_time(epoch_time)}{Colors.END}",
            f"",
            f"Final Metrics:",
            f"  Loss:        {Colors.BOLD}{summary['global_loss']:.4f}{Colors.END}",
            f"  Perplexity:  {Colors.CYAN}{math.exp(min(summary['global_loss'], 20)):.2f}{Colors.END}",
            f"  Bits/Byte:   {summary['global_loss'] / math.log(2):.3f}",
            f"  Token Acc:   {Colors.GREEN}{summary['token_accuracy']*100:.2f}%{Colors.END}",
            f"",
            f"Throughput: {Colors.BOLD}{summary['throughput_tps']:.0f}{Colors.END} tokens/sec",
            f"Total Tokens: {summary['total_tokens']:,}",
        ],
        width=60
    )

    logger.info(
        f"[{stage}] Epoch {epoch + 1} завершена | "
        f"Loss: {summary['global_loss']:.4f} | "
        f"PPL: {math.exp(min(summary['global_loss'], 20)):.2f} | "
        f"Acc: {summary['token_accuracy']*100:.2f}% | "
        f"Time: {format_time(epoch_time)}"
    )

    return summary['global_loss']


def run_mid_training(model, tokenizer, config: PipelineConfig, logger):
    """Запуск Mid-Training (Continued Pre-Training with Next Token Prediction)"""

    # Beautiful header
    print_box(
        "MID-TRAINING (Next Token Prediction)",
        [
            f"Method: {Colors.CYAN}Causal Language Modeling{Colors.END}",
            f"Loss: Cross-entropy on ALL tokens",
            f"Objective: Predict next token",
            "",
            f"Dataset: {Colors.BOLD}{config.dataset_name}{Colors.END}",
            f"Model: {Colors.BOLD}{config.model_name}{Colors.END}",
        ],
        width=70
    )

    logger.info("=" * 60)
    logger.info("MID-TRAINING (Next Token Prediction)")
    logger.info("=" * 60)

    # Загрузка данных (mid-training берёт первую часть датасета)
    samples = load_dataset_samples(config, logger, config.mid_training_max_samples, for_rlvr=False)

    dataset = AgenticDataset(samples, tokenizer, config.max_seq_length)

    dataloader = DataLoader(
        dataset,
        batch_size=config.mid_training_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )

    # Optimizer with fused AdamW for better performance
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    # Try to use fused AdamW for better performance on GPU
    try:
        optimizer = AdamW(
            model.parameters(),
            lr=config.mid_training_lr,
            weight_decay=config.weight_decay,
            fused=True  # Fused implementation for GPU
        )
        logger.info(Colors.success("✓ Using fused AdamW optimizer"))
    except TypeError:
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

    # Training configuration summary
    effective_batch = config.mid_training_batch_size * config.mid_training_grad_accum
    tokens_per_batch = effective_batch * config.max_seq_length

    print_box(
        "TRAINING CONFIGURATION",
        [
            f"Dataset size: {Colors.BOLD}{len(dataset):,}{Colors.END} samples",
            f"Batches per epoch: {Colors.BOLD}{len(dataloader):,}{Colors.END}",
            "",
            f"Batch size: {config.mid_training_batch_size}",
            f"Gradient accumulation: {config.mid_training_grad_accum}",
            f"Effective batch: {Colors.CYAN}{effective_batch}{Colors.END}",
            f"Tokens per batch: ~{tokens_per_batch:,}",
            "",
            f"Epochs: {Colors.BOLD}{config.mid_training_epochs}{Colors.END}",
            f"Total optimization steps: {Colors.BOLD}{total_steps:,}{Colors.END}",
            f"Warmup steps: {warmup_steps:,} ({config.warmup_ratio*100:.0f}%)",
            "",
            f"Learning rate: {Colors.CYAN}{config.mid_training_lr}{Colors.END}",
            f"Weight decay: {config.weight_decay}",
            f"Max gradient norm: {config.max_grad_norm}",
        ],
        width=70
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batches per epoch: {len(dataloader)}")
    logger.info(f"Total optimization steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    logger.info(f"Learning rate: {config.mid_training_lr}")

    # Training loop
    training_start = time.time()

    for epoch in range(config.mid_training_epochs):
        print()
        print_box(
            f"EPOCH {epoch + 1} / {config.mid_training_epochs}",
            [
                f"Progress: {epoch}/{config.mid_training_epochs} complete",
                f"Remaining: {config.mid_training_epochs - epoch} epochs",
            ],
            width=50
        )

        avg_loss = train_epoch(
            model, dataloader, optimizer, scheduler,
            config, logger, epoch, "mid-training"
        )

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            save_path = os.path.join(config.output_dir, f"mid-training-epoch-{epoch + 1}")
            os.makedirs(save_path, exist_ok=True)
            logger.info(f"Сохранение checkpoint: {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

            # Логируем размер чекпойнта
            ckpt_size = get_checkpoint_size(save_path)
            logger.info(f"Размер чекпойнта: {ckpt_size}")

            # Очистка старых чекпойнтов
            cleanup_checkpoints(
                config.output_dir,
                "mid-training-epoch-",
                config.save_total_limit,
                logger
            )

    # Training complete summary
    total_time = time.time() - training_start

    print()
    print_box(
        "MID-TRAINING COMPLETE",
        [
            f"{Colors.GREEN}✓ Training finished successfully{Colors.END}",
            "",
            f"Total time: {Colors.BOLD}{format_time(total_time)}{Colors.END}",
            f"Epochs completed: {config.mid_training_epochs}",
            "",
            f"Final checkpoint: {config.output_dir}/mid-training-epoch-{config.mid_training_epochs}",
        ],
        width=70
    )

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

    # Загрузка данных (RLVR берёт вторую часть датасета - не пересекается с mid-training)
    samples = load_dataset_samples(config, logger, config.rlvr_max_samples, for_rlvr=True)

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

        # Промежуточное сохранение каждые 100 шагов
        if (i + 1) % 100 == 0:
            step_save_path = os.path.join(config.output_dir, f"rlvr-step-{i + 1}")
            os.makedirs(step_save_path, exist_ok=True)
            model.save_pretrained(step_save_path)
            tokenizer.save_pretrained(step_save_path)
            logger.info(f"Промежуточный чекпойнт: {step_save_path}")

            # Очистка старых RLVR чекпойнтов
            cleanup_checkpoints(
                config.output_dir,
                "rlvr-step-",
                config.save_total_limit,
                logger
            )

    # Final optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Save final model
    save_path = os.path.join(config.output_dir, "rlvr-final")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Сохранение финальной модели: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    ckpt_size = get_checkpoint_size(save_path)
    logger.info(f"Размер финальной модели: {ckpt_size}")
    logger.info(f"RLVR Training завершен! Avg Reward: {total_reward / max(num_samples, 1):.3f}")
    return model


def start_vllm_server(model_name: str, port: int, logger) -> Optional[int]:
    """Запуск vLLM сервера для RLVR фазы"""
    import subprocess
    import requests

    logger.info("=" * 40)
    logger.info("STARTING VLLM SERVER")
    logger.info("=" * 40)

    # Проверяем, не запущен ли уже сервер
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            logger.info(f"vLLM server already running on port {port}")
            return None
    except:
        pass

    logger.info(f"Starting vLLM server: {model_name} on port {port}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--max-model-len", "1536",  # Ограничение модели GigaChat3
        "--gpu-memory-utilization", "0.7",  # Оставить память для training модели
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Ждём запуска сервера
    logger.info("Waiting for vLLM server to start...")
    import time
    for i in range(60):
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200:
                logger.info(f"vLLM server ready on port {port}")
                return process.pid
        except:
            pass
        time.sleep(5)
        if i % 6 == 0:
            logger.info(f"Still waiting... ({i*5}s)")

    logger.error("vLLM server failed to start")
    process.kill()
    return None


def stop_vllm_server(pid: Optional[int], logger):
    """Остановка vLLM сервера"""
    if pid is None:
        return

    import signal
    logger.info(f"Stopping vLLM server (PID {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("vLLM server stopped")
    except ProcessLookupError:
        logger.info("vLLM server already stopped")


def main():
    parser = argparse.ArgumentParser(description="GigaChat Full Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--output-dir", type=str, default="./outputs/gigachat-pipeline")
    parser.add_argument("--mid-training-only", action="store_true", help="Run only mid-training")
    parser.add_argument("--rlvr-only", action="store_true", help="Run only RLVR")
    parser.add_argument("--use-atropos", action="store_true", help="Use full Atropos with vLLM")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--max-samples", type=int, default=100, help="Max training samples")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max-turns", type=int, default=3, help="Max tool-use turns per rollout")
    args = parser.parse_args()

    # Setup config
    config = PipelineConfig()
    config.output_dir = args.output_dir
    config.mid_training_max_samples = args.max_samples
    config.rlvr_max_samples = args.max_samples
    config.mid_training_batch_size = args.batch_size
    config.mid_training_epochs = args.epochs
    config.rlvr_max_turns = args.max_turns

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
    logger.info(f"Max turns (RLVR): {config.rlvr_max_turns}")

    log_system_info(logger)

    vllm_pid = None

    try:
        # Load model
        model, tokenizer = load_model_and_tokenizer(config, logger)

        # Apply LoRA
        model = apply_lora(model, config, logger)

        # ================================================================
        # STAGE 1: MID-TRAINING
        # ================================================================
        if not args.rlvr_only:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 1: MID-TRAINING (Next Token Prediction)")
            logger.info("=" * 60)
            model = run_mid_training(model, tokenizer, config, logger)

            # Сохраняем checkpoint после mid-training
            mid_checkpoint = os.path.join(config.output_dir, "mid-training-final")
            os.makedirs(mid_checkpoint, exist_ok=True)
            model.save_pretrained(mid_checkpoint)
            tokenizer.save_pretrained(mid_checkpoint)
            logger.info(f"Mid-training checkpoint saved: {mid_checkpoint}")

        # ================================================================
        # ОЧИСТКА ПАМЯТИ МЕЖДУ СТАДИЯМИ
        # ================================================================
        if not args.mid_training_only and not args.rlvr_only:
            logger.info("\n")
            clear_gpu_memory(logger)

        # ================================================================
        # STAGE 2: RLVR POST-TRAINING
        # ================================================================
        if not args.mid_training_only:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 2: RLVR POST-TRAINING")
            logger.info("=" * 60)

            if args.use_atropos:
                # Используем полный Atropos с vLLM сервером
                logger.info("Using full Atropos RLVR with vLLM server")

                # Выгружаем модель из памяти для vLLM
                logger.info("Unloading model from GPU for vLLM...")
                del model
                clear_gpu_memory(logger)

                # Запускаем vLLM сервер
                vllm_pid = start_vllm_server(config.model_name, args.vllm_port, logger)

                if vllm_pid is not None or True:  # Сервер может быть уже запущен
                    # Запускаем Atropos RLVR
                    import asyncio
                    from .atropos_rlvr import AtroposTrainer, AtroposConfig

                    atropos_config = AtroposConfig(
                        model_name=config.model_name,
                        vllm_base_url=f"http://localhost:{args.vllm_port}/v1",
                        output_dir=os.path.join(config.output_dir, "atropos-rlvr"),
                        num_episodes=config.rlvr_epochs,
                        batch_size=config.mid_training_batch_size,
                        max_samples=config.rlvr_max_samples,
                        max_concurrent_requests=32,  # Высокая параллельность
                        num_rollouts_per_prompt=8,
                        max_turns=config.rlvr_max_turns,  # Tool-use turns
                    )

                    trainer = AtroposTrainer(atropos_config)
                    asyncio.run(trainer.train())
                else:
                    logger.error("Failed to start vLLM server, skipping Atropos RLVR")
            else:
                # Используем простой RLVR без vLLM
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
    finally:
        # Останавливаем vLLM сервер если запускали
        if vllm_pid is not None:
            stop_vllm_server(vllm_pid, logger)


if __name__ == "__main__":
    main()
