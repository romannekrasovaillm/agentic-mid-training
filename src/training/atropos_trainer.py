#!/usr/bin/env python3
"""
Atropos Trainer for GigaChat

Connects to Atropos Trajectory API and performs policy gradient training
on collected rollouts with rewards.

Based on NousResearch/atropos framework.

Usage:
    python atropos_trainer.py \
        --model ai-sage/GigaChat3-10B-A1.8B-bf16 \
        --trajectory-api http://localhost:8000 \
        --output-dir ./outputs/atropos-training
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


@dataclass
class AtroposTrainerConfig:
    """Configuration for Atropos-based training"""

    # Model
    model_name: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"
    output_dir: str = "./outputs/atropos-training"

    # Trajectory API
    trajectory_api_url: str = "http://localhost:8000"
    batch_size: int = 8
    poll_interval: float = 5.0  # seconds

    # Training
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 4
    total_steps: int = 2000
    save_steps: int = 100
    log_steps: int = 10

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Quantization
    use_4bit: bool = True

    # Device
    device: str = "cuda"

    # Policy gradient
    baseline_type: str = "mean"  # mean, running_mean, none
    entropy_coef: float = 0.01
    reward_scale: float = 1.0


class AtroposTrajectoryClient:
    """Client for Atropos Trajectory API"""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()

    def get_batch(self, batch_size: int) -> Optional[List[Dict]]:
        """Get a batch of trajectories from the API"""
        try:
            response = self.session.get(
                f"{self.api_url}/trajectories/batch",
                params={"batch_size": batch_size},
                timeout=30,
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 204:
                # No data available
                return None
            else:
                print(f"Error getting batch: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return None

    def report_training_step(self, step: int, metrics: Dict[str, float]):
        """Report training metrics to API"""
        try:
            self.session.post(
                f"{self.api_url}/training/metrics",
                json={"step": step, "metrics": metrics},
                timeout=10,
            )
        except requests.RequestException:
            pass  # Non-critical

    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


class AtroposTrainer:
    """
    Trainer that pulls trajectories from Atropos API and performs
    policy gradient training.
    """

    def __init__(self, config: AtroposTrainerConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Initialize trajectory client
        self.trajectory_client = AtroposTrajectoryClient(config.trajectory_api_url)

        # Load model and tokenizer
        self._setup_model()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # Metrics
        self.global_step = 0
        self.total_reward = 0.0
        self.num_samples = 0
        self.running_baseline = 0.0

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

    def _setup_model(self):
        """Setup model with optional quantization and LoRA"""
        print(f"Loading model: {self.config.model_name}")

        # Quantization config
        bnb_config = None
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA
        if self.config.use_lora:
            if self.config.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.model.train()

    def compute_policy_gradient_loss(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
    ) -> torch.Tensor:
        """
        Compute policy gradient loss: -E[R * log p(response|prompt)]
        """
        total_loss = torch.tensor(0.0, device=self.device)
        num_valid = 0

        # Compute baseline
        if self.config.baseline_type == "mean":
            baseline = sum(rewards) / len(rewards) if rewards else 0.0
        elif self.config.baseline_type == "running_mean":
            baseline = self.running_baseline
        else:
            baseline = 0.0

        for prompt, response, reward in zip(prompts, responses, rewards):
            # Tokenize
            full_text = prompt + response
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1536,
                padding=False,
            ).to(self.device)

            prompt_encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            )
            prompt_len = prompt_encoded["input_ids"].shape[1]

            # Forward pass
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )

                # Get log probabilities for response tokens
                logits = outputs.logits[:, prompt_len - 1:-1, :]
                target_ids = encoded["input_ids"][:, prompt_len:]

                if target_ids.shape[1] == 0:
                    continue

                # Compute log probs
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

                # Sum log probs over response tokens
                response_log_prob = token_log_probs.sum()

                # Policy gradient: -advantage * log_prob
                advantage = (reward - baseline) * self.config.reward_scale
                loss = -advantage * response_log_prob

                total_loss = total_loss + loss
                num_valid += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"OOM, skipping sample")
                    continue
                raise

        if num_valid > 0:
            total_loss = total_loss / num_valid

        # Update running baseline
        if rewards:
            self.running_baseline = 0.9 * self.running_baseline + 0.1 * (sum(rewards) / len(rewards))

        return total_loss

    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """Perform one training step on a batch of trajectories"""
        prompts = []
        responses = []
        rewards = []

        for trajectory in batch:
            # Extract data from trajectory
            prompt = trajectory.get("prompt", "")
            response = trajectory.get("response", "")
            reward = trajectory.get("reward", 0.0)

            if prompt and response:
                prompts.append(prompt)
                responses.append(response)
                rewards.append(reward)

        if not prompts:
            return {"loss": 0.0, "reward": 0.0}

        # Compute loss
        loss = self.compute_policy_gradient_loss(prompts, responses, rewards)

        # Backward pass
        loss.backward()

        # Update metrics
        self.total_reward += sum(rewards)
        self.num_samples += len(rewards)

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "num_samples": len(rewards),
        }

    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save LoRA weights
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "step": step,
            "total_reward": self.total_reward,
            "num_samples": self.num_samples,
            "running_baseline": self.running_baseline,
        }
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
            json.dump(state, f)

        print(f"Saved checkpoint to {checkpoint_dir}")

    def train(self):
        """Main training loop"""
        print("=" * 60)
        print("ATROPOS TRAINER")
        print("=" * 60)
        print(f"Model: {self.config.model_name}")
        print(f"Trajectory API: {self.config.trajectory_api_url}")
        print(f"Total steps: {self.config.total_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print("=" * 60)

        # Check API health
        if not self.trajectory_client.health_check():
            print("WARNING: Trajectory API not available. Will poll until ready...")

        progress_bar = tqdm(
            range(self.config.total_steps),
            desc="Training",
            file=sys.stdout,
        )

        self.optimizer.zero_grad()

        for step in progress_bar:
            self.global_step = step

            # Get batch from trajectory API
            batch = None
            while batch is None:
                batch = self.trajectory_client.get_batch(self.config.batch_size)
                if batch is None:
                    time.sleep(self.config.poll_interval)

            # Train step
            metrics = self.train_step(batch)

            # Update progress bar
            avg_reward = self.total_reward / self.num_samples if self.num_samples > 0 else 0.0
            progress_bar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "reward": f"{metrics['reward']:.3f}",
                "avg_reward": f"{avg_reward:.3f}",
            })

            # Log
            if (step + 1) % self.config.log_steps == 0:
                log_metrics = {
                    "loss": metrics["loss"],
                    "reward": metrics["reward"],
                    "avg_reward": avg_reward,
                    "samples": self.num_samples,
                }
                self.trajectory_client.report_training_step(step, log_metrics)
                print(f"\n[Step {step + 1}] Loss: {metrics['loss']:.4f}, "
                      f"Reward: {metrics['reward']:.3f}, Avg: {avg_reward:.3f}")

            # Save checkpoint
            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(step + 1)

        # Final save
        self.save_checkpoint(self.config.total_steps)
        print("\nTraining complete!")
        print(f"Total samples: {self.num_samples}")
        print(f"Average reward: {self.total_reward / self.num_samples if self.num_samples > 0 else 0:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Atropos Trainer for GigaChat")

    # Model
    parser.add_argument("--model", type=str, default="ai-sage/GigaChat3-10B-A1.8B-bf16",
                        help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="./outputs/atropos-training",
                        help="Output directory")

    # Trajectory API
    parser.add_argument("--trajectory-api", type=str, default="http://localhost:8000",
                        help="Atropos Trajectory API URL")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Polling interval in seconds")

    # Training
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--total-steps", type=int, default=2000,
                        help="Total training steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save checkpoint every N steps")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")

    # Quantization
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    # Policy gradient
    parser.add_argument("--baseline-type", type=str, default="mean",
                        choices=["mean", "running_mean", "none"],
                        help="Baseline type for advantage computation")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Reward scaling factor")

    args = parser.parse_args()

    # Create config
    config = AtroposTrainerConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        trajectory_api_url=args.trajectory_api,
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        learning_rate=args.learning_rate,
        total_steps=args.total_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_lora=not args.no_lora,
        use_4bit=not args.no_4bit,
        baseline_type=args.baseline_type,
        reward_scale=args.reward_scale,
    )

    # Create trainer and run
    trainer = AtroposTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
