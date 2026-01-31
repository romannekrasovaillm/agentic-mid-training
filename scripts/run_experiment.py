#!/usr/bin/env python3
"""Run a single entropy-reward experiment from a config file.

Usage:
    python scripts/run_experiment.py --config configs/base.yaml
    python scripts/run_experiment.py --config configs/adaptive_entropy.yaml --override '{"training.max_steps": 1000}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.utils.config_loader import load_config
from entropy_reward.utils.logging_utils import setup_logger
from entropy_reward.trainer import GRPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run entropy-reward experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", type=str, default=None, help="JSON dict of config overrides")
    parser.add_argument("--base-config", type=str, default=None, help="Base config to merge with")
    parser.add_argument("--dry-run", action="store_true", help="Only print config without training")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint dir to resume from")
    return parser.parse_args()


def load_dataset_placeholder() -> tuple[list[str], list[float], list[str]]:
    """Placeholder dataset loader. Replace with actual data loading.

    Returns:
        (train_prompts, train_accuracy_scores, eval_prompts)
    """
    train_prompts = [
        "Solve: What is 15 * 23? Use the calculate tool to find the answer.",
        "Search for the capital of France and provide the answer.",
        "Write a Python function to check if a number is prime.",
        "Analyze the sentiment of: 'This product is amazing and worth every penny'",
        "Calculate the area of a circle with radius 7. Show your work.",
        "Find information about the tallest building in the world.",
        "Explain the difference between a stack and a queue data structure.",
        "What is the square root of 144? Use tools to verify.",
    ]
    train_acc = [0.0] * len(train_prompts)  # to be filled by actual accuracy checker
    eval_prompts = train_prompts[:4]  # subset for eval
    return train_prompts, train_acc, eval_prompts


def main():
    args = parse_args()

    # Load config
    overrides = json.loads(args.override) if args.override else None

    if args.base_config:
        config = load_config(args.base_config)
        config_override = load_config(args.config, overrides)
        # Merge — use override config values
        config = load_config(args.base_config, overrides)
    else:
        config = load_config(args.config, overrides)

    # Setup logging
    log = setup_logger("experiment", log_dir=str(Path(config.training.output_dir) / "logs"))
    log.info(f"Experiment: {config.name}")
    log.info(f"Description: {config.description}")
    log.info(f"Config: {config}")

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Name: {config.name}")
        print(f"Entropy: {config.entropy.strategy}")
        print(f"KL: {config.kl.schedule} ({config.kl.initial_coeff} -> {config.kl.final_coeff})")
        print(f"Reward: format={config.reward.format_mode}, baseline={config.reward.baseline}")
        print(f"Steps: {config.training.max_steps}")
        return

    # Load model & tokenizer
    log.info(f"Loading model: {config.training.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.training.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.training.model_name,
        torch_dtype=torch.bfloat16 if config.training.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    ref_model_name = config.training.ref_model_name or config.training.model_name
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        torch_dtype=torch.bfloat16 if config.training.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=0.01,
    )

    # Create trainer
    trainer = GRPOTrainer(config)
    trainer.setup_model(model, ref_model, tokenizer, optimizer)

    # Init wandb
    if config.training.wandb_project:
        from dataclasses import asdict
        trainer.reward_logger.init_wandb(
            project=config.training.wandb_project,
            run_name=config.training.wandb_run_name or config.name,
            config=asdict(config),
        )

    # Load data
    train_prompts, train_acc, eval_prompts = load_dataset_placeholder()

    # Training loop
    log.info(f"Starting training for {config.training.max_steps} steps")
    batch_size = config.training.batch_size

    for step in range(1, config.training.max_steps + 1):
        # Sample batch
        indices = torch.randint(0, len(train_prompts), (batch_size,))
        batch_prompts = [train_prompts[i] for i in indices]
        batch_acc = [train_acc[i] for i in indices]

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
            eval_results = trainer.run_eval(eval_prompts, step)
            log.info(f"[Step {step}] Eval: {eval_results}")

        # Stop condition
        if metrics["should_stop"]:
            log.warning(f"STOPPING at step {step}: {metrics['stop_reason']}")
            trainer.save_checkpoint(step)
            break

        # Periodic checkpoint
        if step % 500 == 0:
            trainer.save_checkpoint(step)

    # Final save
    trainer.save_checkpoint(step)
    trainer.close()
    log.info("Training complete.")


if __name__ == "__main__":
    main()
