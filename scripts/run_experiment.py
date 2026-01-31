#!/usr/bin/env python3
"""Run a single entropy-reward experiment from a config file.

Usage:
    python scripts/run_experiment.py --config configs/base.yaml
    python scripts/run_experiment.py --config configs/adaptive_entropy.yaml --base-config configs/base.yaml
    python scripts/run_experiment.py --config configs/base.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.utils.config_loader import (
    ExperimentConfig,
    load_config,
    _merge_dict,
    _dict_to_dataclass,
)
from entropy_reward.utils.logging_utils import setup_logger
from entropy_reward.trainer import GRPOTrainer


TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run entropy-reward experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", type=str, default=None, help="JSON dict of config overrides")
    parser.add_argument("--base-config", type=str, default=None, help="Base config to merge with")
    parser.add_argument("--dry-run", action="store_true", help="Only print config without training")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint dir to resume from")
    return parser.parse_args()


def load_nemotron_data(config: ExperimentConfig):
    """Load nvidia/Nemotron-Agentic-v1 dataset.

    Returns:
        (train_samples, eval_samples) — lists of AgenticSample
    """
    from entropy_reward.data import load_nemotron_splits

    dcfg = config.dataset
    train_samples, eval_samples = load_nemotron_splits(
        max_train=dcfg.max_train_samples,
        max_eval=dcfg.max_eval_samples,
        multiturn=dcfg.multiturn,
        seed=config.training.seed,
        cache_dir=dcfg.cache_dir or None,
    )
    return train_samples, eval_samples


def load_model_and_tokenizer(config: ExperimentConfig):
    """Load model and tokenizer using config.model settings.

    Handles GigaChat3-10B-A1.8B (MoE with MLA + MTP) and standard HF models.
    """
    mcfg = config.model
    model_name = mcfg.name
    dtype = TORCH_DTYPE_MAP.get(mcfg.torch_dtype, torch.bfloat16)

    log = logging.getLogger("experiment")
    log.info(f"Loading model: {model_name}")
    log.info(
        f"  dtype={mcfg.torch_dtype}, attn={mcfg.attn_implementation}, "
        f"device_map={mcfg.device_map}, trust_remote_code={mcfg.trust_remote_code}"
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=mcfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model kwargs
    model_kwargs = dict(
        torch_dtype=dtype,
        device_map=mcfg.device_map,
        trust_remote_code=mcfg.trust_remote_code,
    )

    # Attention implementation (flash_attention_2, sdpa, eager)
    if mcfg.attn_implementation:
        model_kwargs["attn_implementation"] = mcfg.attn_implementation

    # Memory constraints
    if mcfg.max_memory:
        model_kwargs["max_memory"] = mcfg.max_memory

    # Policy model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Apply HF GenerationConfig from model hub (GigaChat3 ships one)
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_name)
    except Exception:
        pass

    # Reference model (frozen copy for KL anchor)
    ref_model_name = mcfg.ref_model_name or model_name
    log.info(f"Loading reference model: {ref_model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, **model_kwargs)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    return model, ref_model, tokenizer


def main():
    args = parse_args()

    # Load config
    overrides = json.loads(args.override) if args.override else None

    if args.base_config:
        # Load base, then overlay experiment config on top
        base_config = load_config(args.base_config)
        base_dict = asdict(base_config)
        with open(args.config) as f:
            exp_raw = yaml.safe_load(f) or {}
        merged = _merge_dict(base_dict, exp_raw)
        if overrides:
            merged = _merge_dict(merged, overrides)
        config = _dict_to_dataclass(ExperimentConfig, merged)
    else:
        config = load_config(args.config, overrides)

    # Setup logging
    log = setup_logger("experiment", log_dir=str(Path(config.training.output_dir) / "logs"))
    log.info(f"Experiment: {config.name}")
    log.info(f"Description: {config.description}")
    log.info(f"Model: {config.model.name}")
    log.info(f"Config: {config}")

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Name:     {config.name}")
        print(f"Model:    {config.model.name}")
        print(f"  dtype:  {config.model.torch_dtype}")
        print(f"  attn:   {config.model.attn_implementation}")
        print(f"  chat:   {config.model.use_chat_template}")
        print(f"Dataset:  {config.dataset.name}")
        print(f"  train:  {config.dataset.train_split}")
        print(f"  eval:   {config.dataset.eval_split}")
        print(f"Entropy:  {config.entropy.strategy}")
        print(f"KL:       {config.kl.schedule} ({config.kl.initial_coeff} -> {config.kl.final_coeff})")
        print(f"Reward:   format={config.reward.format_mode}, baseline={config.reward.baseline}")
        print(f"Steps:    {config.training.max_steps}")
        print(f"Seq len:  {config.training.max_seq_len}")
        print(f"Batch:    {config.training.batch_size} x {config.training.gradient_accumulation_steps} accum")
        return

    # Load dataset (nvidia/Nemotron-Agentic-v1)
    log.info(f"Loading dataset: {config.dataset.name}")
    train_samples, eval_samples = load_nemotron_data(config)
    log.info(f"Train: {len(train_samples)} samples, Eval: {len(eval_samples)} samples")

    # Collect unique tool names across all training samples
    all_tool_names = set()
    for s in train_samples:
        all_tool_names.update(s.tool_names)
    log.info(f"Unique tools in training data: {len(all_tool_names)} — {sorted(all_tool_names)[:20]}...")

    # Load model & tokenizer
    model, ref_model, tokenizer = load_model_and_tokenizer(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create trainer — pass available tools for reward scoring
    trainer = GRPOTrainer(config)
    trainer.reward_fn.available_tools = sorted(all_tool_names)
    trainer.setup_model(model, ref_model, tokenizer, optimizer)

    # Init wandb
    if config.training.wandb_project:
        trainer.reward_logger.init_wandb(
            project=config.training.wandb_project,
            run_name=config.training.wandb_run_name or config.name,
            config=asdict(config),
        )

    # Accuracy scorer
    from entropy_reward.data.accuracy import compute_accuracy

    # Training loop
    log.info(f"Starting training for {config.training.max_steps} steps")
    batch_size = config.training.batch_size

    for step in range(1, config.training.max_steps + 1):
        # Sample batch from Nemotron data
        indices = torch.randint(0, len(train_samples), (batch_size,))
        batch_samples = [train_samples[i] for i in indices]
        batch_prompts = [s.prompt for s in batch_samples]

        # Accuracy will be computed after generation inside trainer,
        # for now pass 0 (trainer.train_step handles R_acc externally)
        batch_acc = [0.0] * batch_size

        # Train step
        metrics = trainer.train_step(batch_prompts, batch_acc, step)

        # Log
        if step % config.metrics.log_interval == 0:
            log.info(
                f"[Step {step}] loss={metrics['loss']:.4f} "
                f"H_tok={metrics['token_entropy']:.4f} "
                f"H_act={metrics['action_entropy']:.4f} "
                f"BLEU={metrics['self_bleu']:.4f} "
                f"R_fmt={metrics['r_format']:.3f} "
                f"R_tool={metrics['r_tool']:.3f} "
                f"R_acc={metrics['r_acc']:.3f}"
            )

        # Eval
        if step % config.metrics.eval_interval == 0:
            eval_prompts = [s.prompt for s in eval_samples[:50]]
            eval_results = trainer.run_eval(eval_prompts, step)
            log.info(f"[Step {step}] Eval: {eval_results}")

        # Stop condition
        if metrics["should_stop"]:
            log.warning(f"STOPPING at step {step}: {metrics['stop_reason']}")
            trainer.save_checkpoint(step)
            break

        # Periodic checkpoint
        if step % config.training.checkpoint_interval == 0:
            trainer.save_checkpoint(step)

    # Final save
    trainer.save_checkpoint(step)
    trainer.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
