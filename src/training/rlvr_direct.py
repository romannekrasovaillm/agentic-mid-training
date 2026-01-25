#!/usr/bin/env python3
"""
Direct RLVR Training for GigaChat

Simplified RLVR trainer that works directly with vLLM server,
without needing the full Atropos microservice architecture.

Features:
- Generates rollouts directly from vLLM
- Computes rewards inline
- Policy gradient training with comprehensive metrics
- Beautiful terminal logging

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model ai-sage/GigaChat3-10B-A1.8B-bf16 \
        --port 8000 --dtype bfloat16 --trust-remote-code \
        --max-model-len 1536 --gpu-memory-utilization 0.6

    # Then run training:
    python rlvr_direct.py \
        --vllm-url http://localhost:8000/v1 \
        --output-dir ./outputs/rlvr
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from openai import OpenAI
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


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
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'

    @staticmethod
    def reward(value: float) -> str:
        if value > 0.5:
            return f"{Colors.GREEN}{value:+.3f}{Colors.END}"
        elif value > 0:
            return f"{Colors.YELLOW}{value:+.3f}{Colors.END}"
        elif value > -0.5:
            return f"{Colors.YELLOW}{value:+.3f}{Colors.END}"
        else:
            return f"{Colors.RED}{value:+.3f}{Colors.END}"

    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.GREEN}{text}{Colors.END}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.YELLOW}{text}{Colors.END}"


def print_box(title: str, content: list, width: int = 70, color: str = None):
    """Print a beautiful box with title and content"""
    if color is None:
        color = Colors.CYAN

    border = "═" * width
    print(f"\n{color}╔{border}╗{Colors.END}")
    print(f"{color}║{Colors.END} {Colors.BOLD}{title.center(width - 2)}{Colors.END} {color}║{Colors.END}")
    print(f"{color}╠{border}╣{Colors.END}")

    for line in content:
        clean_line = re.sub(r'\033\[[0-9;]*m', '', str(line))
        padding = width - 2 - len(clean_line)
        print(f"{color}║{Colors.END} {line}{' ' * max(0, padding)} {color}║{Colors.END}")

    print(f"{color}╚{border}╝{Colors.END}\n")


def print_separator(char: str = "─", width: int = 70):
    print(f"{Colors.DIM}{char * width}{Colors.END}")


def format_number(num: float, precision: int = 4) -> str:
    if abs(num) < 0.0001:
        return f"{num:.2e}"
    elif abs(num) < 1:
        return f"{num:.{precision}f}"
    elif abs(num) < 1000:
        return f"{num:.{min(precision, 2)}f}"
    else:
        return f"{num:,.0f}"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def truncate_text(text: str, max_len: int = 50) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ============================================================================
# METRICS TRACKER
# ============================================================================

class RLVRMetrics:
    """Track RLVR training metrics"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.rewards = deque(maxlen=self.window_size)
        self.cumulative_reward = 0.0
        self.advantages = deque(maxlen=self.window_size)
        self.policy_losses = deque(maxlen=self.window_size)
        self.entropy_losses = deque(maxlen=self.window_size)
        self.token_entropies = deque(maxlen=self.window_size)
        self.total_tool_calls = 0
        self.correct_tool_calls = 0
        self.tasks_attempted = 0
        self.tasks_solved = 0
        self.gradient_norms = deque(maxlen=self.window_size)
        self.response_lengths = deque(maxlen=self.window_size)
        self.total_episodes = 0
        self.start_time = time.time()
        self.best_examples = []
        self.worst_examples = []
        self.recent_examples = []

    def update(self, reward, advantage=None, policy_loss=None, entropy=None,
               tool_calls=0, correct_tools=0, solved=False, response_len=0,
               grad_norm=None, prompt="", response=""):
        self.rewards.append(reward)
        self.cumulative_reward += reward
        self.total_episodes += 1

        if advantage is not None:
            self.advantages.append(advantage)
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if entropy is not None:
            self.token_entropies.append(entropy)
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)

        self.response_lengths.append(response_len)
        self.total_tool_calls += tool_calls
        self.correct_tool_calls += correct_tools
        self.tasks_attempted += 1
        if solved:
            self.tasks_solved += 1

        # Track examples
        example = {"prompt": prompt[:200], "response": response[:300], "reward": reward}
        self.recent_examples.append(example)
        if len(self.recent_examples) > 5:
            self.recent_examples.pop(0)

        if len(self.best_examples) < 3 or reward > min(e["reward"] for e in self.best_examples):
            self.best_examples.append(example)
            self.best_examples = sorted(self.best_examples, key=lambda x: x["reward"], reverse=True)[:3]

        if len(self.worst_examples) < 3 or reward < max(e["reward"] for e in self.worst_examples):
            self.worst_examples.append(example)
            self.worst_examples = sorted(self.worst_examples, key=lambda x: x["reward"])[:3]

    @property
    def mean_reward(self):
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def std_reward(self):
        if len(self.rewards) < 2:
            return 0.0
        mean = self.mean_reward
        return math.sqrt(sum((r - mean) ** 2 for r in self.rewards) / len(self.rewards))

    @property
    def mean_policy_loss(self):
        return sum(self.policy_losses) / len(self.policy_losses) if self.policy_losses else 0.0

    @property
    def mean_entropy(self):
        return sum(self.token_entropies) / len(self.token_entropies) if self.token_entropies else 0.0

    @property
    def mean_advantage(self):
        return sum(self.advantages) / len(self.advantages) if self.advantages else 0.0

    @property
    def tool_accuracy(self):
        return self.correct_tool_calls / self.total_tool_calls if self.total_tool_calls > 0 else 0.0

    @property
    def task_accuracy(self):
        return self.tasks_solved / self.tasks_attempted if self.tasks_attempted > 0 else 0.0

    @property
    def mean_grad_norm(self):
        return sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0.0

    @property
    def throughput(self):
        elapsed = time.time() - self.start_time
        return self.total_episodes / elapsed if elapsed > 0 else 0.0

    def get_summary(self):
        return {
            "reward_mean": self.mean_reward,
            "reward_std": self.std_reward,
            "reward_cumulative": self.cumulative_reward,
            "advantage_mean": self.mean_advantage,
            "policy_loss": self.mean_policy_loss,
            "entropy": self.mean_entropy,
            "tool_accuracy": self.tool_accuracy,
            "task_accuracy": self.task_accuracy,
            "gradient_norm": self.mean_grad_norm,
            "total_episodes": self.total_episodes,
            "throughput": self.throughput,
        }


# ============================================================================
# REWARD FUNCTION
# ============================================================================

class RewardComputer:
    """Compute rewards for tool-calling responses"""

    TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)

    @classmethod
    def extract_tool_calls(cls, text: str) -> List[Dict]:
        """Extract tool calls from response"""
        tool_calls = []
        matches = cls.TOOL_CALL_PATTERN.findall(text)
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                pass
        return tool_calls

    @classmethod
    def compute_reward(cls, response: str, expected_tools: List[Dict] = None) -> Tuple[float, Dict]:
        """
        Compute reward for a response.

        Reward components:
        - Tool call format: +0.3 if valid JSON tool call
        - Tool name match: +0.3 if matches expected
        - Arguments match: +0.4 if arguments correct
        - Penalty for no tool call when expected: -0.5
        """
        info = {
            "has_tool_call": False,
            "valid_json": False,
            "name_match": False,
            "args_match": False,
            "tool_calls": [],
        }

        actual_tools = cls.extract_tool_calls(response)
        info["tool_calls"] = actual_tools
        info["has_tool_call"] = len(actual_tools) > 0

        if not actual_tools:
            # No tool call - penalize if we expected one
            if expected_tools:
                return -0.5, info
            return 0.0, info

        info["valid_json"] = True
        reward = 0.3  # Base reward for valid tool call

        if expected_tools:
            # Check name match
            expected_names = {t.get("name", "") for t in expected_tools}
            actual_names = {t.get("name", "") for t in actual_tools}

            matches = expected_names & actual_names
            if matches:
                info["name_match"] = True
                reward += 0.3

                # Check arguments (simplified)
                for exp in expected_tools:
                    for act in actual_tools:
                        if exp.get("name") == act.get("name"):
                            exp_args = exp.get("arguments", {})
                            act_args = act.get("arguments", {})
                            if exp_args == act_args:
                                info["args_match"] = True
                                reward += 0.4
                                break
        else:
            # No expected tools to compare, give partial reward
            reward += 0.2

        return min(reward, 1.0), info


# ============================================================================
# DIRECT RLVR TRAINER
# ============================================================================

@dataclass
class DirectRLVRConfig:
    """Configuration for direct RLVR training"""
    model_name: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"
    vllm_url: str = "http://localhost:8000/v1"
    output_dir: str = "./outputs/rlvr-direct"
    dataset_name: str = "nvidia/Nemotron-Agentic-v1"

    # Training
    total_steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    gradient_accumulation: int = 4

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # Policy gradient
    entropy_coef: float = 0.01
    reward_scale: float = 1.0
    baseline_momentum: float = 0.99

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128

    # Logging
    log_steps: int = 10
    save_steps: int = 100
    example_interval: int = 50

    # Data
    max_samples: int = 5000
    max_seq_length: int = 1536


class DirectRLVRTrainer:
    """
    RLVR Trainer that works directly with vLLM.

    Flow:
    1. Load prompts from dataset
    2. Generate responses using vLLM
    3. Compute rewards
    4. Update policy with policy gradient
    """

    def __init__(self, config: DirectRLVRConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Metrics
        self.metrics = RLVRMetrics()

        # Running baseline
        self.running_baseline = 0.0

        # Setup
        self._load_dataset()
        self._setup_vllm_client()
        self._setup_model()

        os.makedirs(config.output_dir, exist_ok=True)

    def _load_dataset(self):
        """Load and prepare dataset"""
        print(f"Loading dataset: {self.config.dataset_name}")

        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split="train",
                trust_remote_code=True
            )
            self.samples = list(dataset)[:self.config.max_samples]
            random.shuffle(self.samples)
            print(f"Loaded {len(self.samples)} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.samples = []

    def _setup_vllm_client(self):
        """Setup OpenAI client for vLLM"""
        print(f"Connecting to vLLM: {self.config.vllm_url}")
        self.vllm_client = OpenAI(
            base_url=self.config.vllm_url,
            api_key="EMPTY",
        )

        # Test connection
        try:
            models = self.vllm_client.models.list()
            print(Colors.success(f"✓ Connected to vLLM"))
            if models.data:
                print(f"  Model: {models.data[0].id}")
        except Exception as e:
            print(Colors.warning(f"⚠ vLLM connection failed: {e}"))

    def _setup_model(self):
        """Setup training model with LoRA"""
        print(f"Loading training model: {self.config.model_name}")

        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply LoRA
        if self.config.use_lora:
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.model.train()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    def _format_prompt(self, sample: Dict) -> Tuple[str, List[Dict]]:
        """Format sample into prompt and extract expected tools"""
        messages = sample.get("messages", [])
        if isinstance(messages, str):
            messages = json.loads(messages)

        prompt_parts = []
        expected_tools = []

        system_prompt = """You are a helpful AI assistant with access to tools. When you need to use a tool, respond with:

<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

Think step by step before making tool calls."""

        prompt_parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                # Extract expected tool calls from assistant message
                tools = RewardComputer.extract_tool_calls(content)
                expected_tools.extend(tools)
                # Don't include assistant response in prompt
                break
            elif role == "tool":
                pass  # Skip tool responses for now

        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts), expected_tools

    def generate_response(self, prompt: str) -> str:
        """Generate response using vLLM"""
        try:
            response = self.vllm_client.completions.create(
                model=self.config.model_name,
                prompt=prompt,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            return response.choices[0].text
        except Exception as e:
            print(f"Generation error: {e}")
            return ""

    def compute_policy_loss(
        self,
        prompt: str,
        response: str,
        reward: float,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute policy gradient loss"""
        # Tokenize
        full_text = prompt + response
        try:
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.device)
        except Exception:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0

        prompt_encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_encoded["input_ids"].shape[1]
        response_len = encoded["input_ids"].shape[1] - prompt_len

        if response_len <= 0:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0

        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

        # Response logits
        logits = outputs.logits[:, prompt_len - 1:-1, :]
        target_ids = encoded["input_ids"][:, prompt_len:]

        # Log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        response_log_prob = token_log_probs.sum()

        # Entropy
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Advantage
        advantage = (reward - self.running_baseline) * self.config.reward_scale

        # Policy loss
        policy_loss = -advantage * response_log_prob
        entropy_loss = -self.config.entropy_coef * entropy

        total_loss = policy_loss + entropy_loss

        return total_loss, entropy.item(), advantage

    def train_step(self, samples: List[Dict]) -> Dict[str, float]:
        """Perform one training step"""
        step_start = time.time()

        total_loss = torch.tensor(0.0, device=self.device)
        batch_rewards = []
        num_valid = 0

        for sample in samples:
            # Format prompt
            prompt, expected_tools = self._format_prompt(sample)

            # Generate response
            response = self.generate_response(prompt)
            if not response:
                continue

            # Compute reward
            reward, reward_info = RewardComputer.compute_reward(response, expected_tools)
            batch_rewards.append(reward)

            # Compute loss
            loss, entropy, advantage = self.compute_policy_loss(prompt, response, reward)

            if loss.requires_grad:
                total_loss = total_loss + loss
                num_valid += 1

            # Update metrics
            tool_calls = reward_info.get("tool_calls", [])
            self.metrics.update(
                reward=reward,
                advantage=advantage,
                policy_loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
                entropy=entropy,
                tool_calls=len(tool_calls),
                correct_tools=1 if reward_info.get("name_match") else 0,
                solved=reward > 0.5,
                response_len=len(response),
                prompt=prompt,
                response=response,
            )

        if num_valid > 0:
            total_loss = total_loss / num_valid
            total_loss.backward()

        # Update running baseline
        if batch_rewards:
            batch_mean = sum(batch_rewards) / len(batch_rewards)
            self.running_baseline = (
                self.config.baseline_momentum * self.running_baseline +
                (1 - self.config.baseline_momentum) * batch_mean
            )

        return {
            "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            "reward": sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0,
            "num_valid": num_valid,
            "step_time": time.time() - step_start,
        }

    def log_detailed_metrics(self, step: int, eta: float):
        """Log detailed metrics"""
        summary = self.metrics.get_summary()

        print()
        print_separator("═", 80)
        print(
            f"  {Colors.BOLD}Step {step}/{self.config.total_steps}{Colors.END} "
            f"({step / self.config.total_steps * 100:.1f}%) "
            f"│ ETA: {Colors.CYAN}{format_time(eta)}{Colors.END}"
        )
        print_separator("─", 80)

        print(f"\n  {Colors.BOLD}{Colors.GREEN}📊 REWARDS{Colors.END}")
        print(f"    Mean:       {Colors.reward(summary['reward_mean'])}")
        print(f"    Std:        {format_number(summary['reward_std'])}")
        print(f"    Cumulative: {Colors.BOLD}{summary['reward_cumulative']:.2f}{Colors.END}")

        print(f"\n  {Colors.BOLD}{Colors.BLUE}📉 TRAINING{Colors.END}")
        print(f"    Policy Loss: {format_number(summary['policy_loss'])}")
        print(f"    Entropy:     {format_number(summary['entropy'])}")
        print(f"    Advantage:   {format_number(summary['advantage_mean'])}")
        print(f"    Grad Norm:   {format_number(summary['gradient_norm'])}")

        print(f"\n  {Colors.BOLD}{Colors.YELLOW}🔧 TOOL CALLS{Colors.END}")
        acc_color = Colors.GREEN if summary['tool_accuracy'] > 0.5 else Colors.YELLOW
        print(f"    Accuracy: {acc_color}{summary['tool_accuracy']*100:.1f}%{Colors.END}")

        print(f"\n  {Colors.BOLD}{Colors.CYAN}✓ TASKS{Colors.END}")
        task_color = Colors.GREEN if summary['task_accuracy'] > 0.5 else Colors.YELLOW
        print(f"    Accuracy: {task_color}{summary['task_accuracy']*100:.1f}%{Colors.END}")
        print(f"    Episodes: {summary['total_episodes']:,}")
        print(f"    Throughput: {summary['throughput']:.2f} ep/s")

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            print(f"\n  {Colors.DIM}GPU: {alloc:.1f}GB{Colors.END}")

        print_separator("═", 80)

    def log_examples(self, step: int):
        """Log example responses"""
        if self.metrics.recent_examples:
            print_box(
                "📝 RECENT EXAMPLE",
                [
                    f"Prompt: {truncate_text(self.metrics.recent_examples[-1]['prompt'], 55)}",
                    f"Response: {truncate_text(self.metrics.recent_examples[-1]['response'], 55)}",
                    f"Reward: {Colors.reward(self.metrics.recent_examples[-1]['reward'])}",
                ],
                width=70,
                color=Colors.DIM
            )

    def save_checkpoint(self, step: int):
        """Save checkpoint"""
        ckpt_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)

        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        state = {
            "step": step,
            "running_baseline": self.running_baseline,
            "metrics": self.metrics.get_summary(),
        }
        with open(os.path.join(ckpt_dir, "trainer_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        print(Colors.success(f"✓ Saved checkpoint: {ckpt_dir}"))

    def train(self):
        """Main training loop"""
        print_box(
            "DIRECT RLVR TRAINER",
            [
                f"Model: {Colors.BOLD}{self.config.model_name}{Colors.END}",
                f"vLLM: {self.config.vllm_url}",
                "",
                f"Steps: {Colors.CYAN}{self.config.total_steps}{Colors.END}",
                f"Batch: {self.config.batch_size}",
                f"LR: {self.config.learning_rate}",
                "",
                f"Samples: {len(self.samples)}",
            ],
            width=70
        )

        self.optimizer.zero_grad()
        training_start = time.time()

        progress = tqdm(
            range(self.config.total_steps),
            desc=f"{Colors.CYAN}RLVR{Colors.END}",
            ncols=100,
        )

        sample_idx = 0

        for step in progress:
            # Get batch
            batch = []
            for _ in range(self.config.batch_size):
                if sample_idx >= len(self.samples):
                    sample_idx = 0
                    random.shuffle(self.samples)
                batch.append(self.samples[sample_idx])
                sample_idx += 1

            # Train step
            step_metrics = self.train_step(batch)

            # Gradient step
            if (step + 1) % self.config.gradient_accumulation == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                ).item()
                self.metrics.gradient_norms.append(grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update progress
            summary = self.metrics.get_summary()
            progress.set_postfix_str(
                f"R={Colors.reward(summary['reward_mean'])} | "
                f"Tool={summary['tool_accuracy']*100:.0f}% | "
                f"Task={summary['task_accuracy']*100:.0f}%"
            )

            # Logging
            if (step + 1) % self.config.log_steps == 0:
                elapsed = time.time() - training_start
                eta = (elapsed / (step + 1)) * (self.config.total_steps - step - 1)
                self.log_detailed_metrics(step + 1, eta)

            if (step + 1) % self.config.example_interval == 0:
                self.log_examples(step + 1)

            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(step + 1)

        # Final save
        self.save_checkpoint(self.config.total_steps)

        # Summary
        total_time = time.time() - training_start
        final = self.metrics.get_summary()

        print_box(
            "🎉 TRAINING COMPLETE",
            [
                f"Time: {Colors.BOLD}{format_time(total_time)}{Colors.END}",
                f"Episodes: {final['total_episodes']:,}",
                "",
                f"Mean Reward: {Colors.reward(final['reward_mean'])}",
                f"Tool Accuracy: {final['tool_accuracy']*100:.1f}%",
                f"Task Accuracy: {final['task_accuracy']*100:.1f}%",
            ],
            width=70,
            color=Colors.GREEN
        )


def main():
    parser = argparse.ArgumentParser(description="Direct RLVR Training")

    parser.add_argument("--model", default="ai-sage/GigaChat3-10B-A1.8B-bf16")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--output-dir", default="./outputs/rlvr-direct")
    parser.add_argument("--dataset", default="nvidia/Nemotron-Agentic-v1")

    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=5000)

    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)

    args = parser.parse_args()

    config = DirectRLVRConfig(
        model_name=args.model,
        vllm_url=args.vllm_url,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
    )

    trainer = DirectRLVRTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
