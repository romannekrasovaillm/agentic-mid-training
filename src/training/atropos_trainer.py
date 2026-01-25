#!/usr/bin/env python3
"""
Atropos Trainer for GigaChat

Connects to Atropos Trajectory API and performs policy gradient training
on collected rollouts with rewards.

Based on NousResearch/atropos framework.

Features:
- Comprehensive RLVR metrics tracking
- Beautiful terminal logging
- Tool call accuracy tracking
- Entropy monitoring (per-token and per-trace)
- Advantage estimation
- Task examples logging

Usage:
    python atropos_trainer.py \
        --model ai-sage/GigaChat3-10B-A1.8B-bf16 \
        --trajectory-api http://localhost:8000 \
        --output-dir ./outputs/atropos-training
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
import torch.nn.functional as F
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
    UNDERLINE = '\033[4m'
    END = '\033[0m'

    @staticmethod
    def colored(text: str, color: str) -> str:
        return f"{color}{text}{Colors.END}"

    @staticmethod
    def bold(text: str) -> str:
        return f"{Colors.BOLD}{text}{Colors.END}"

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
    def metric(name: str, value: str) -> str:
        return f"{Colors.DIM}{name}:{Colors.END} {Colors.BOLD}{value}{Colors.END}"


def print_box(title: str, content: list, width: int = 70, color: str = None):
    """Print a beautiful box with title and content"""
    if color is None:
        color = Colors.CYAN

    border = "═" * width
    print(f"\n{color}╔{border}╗{Colors.END}")
    print(f"{color}║{Colors.END} {Colors.BOLD}{title.center(width - 2)}{Colors.END} {color}║{Colors.END}")
    print(f"{color}╠{border}╣{Colors.END}")

    for line in content:
        # Strip ANSI codes for length calculation
        clean_line = re.sub(r'\033\[[0-9;]*m', '', str(line))
        padding = width - 2 - len(clean_line)
        print(f"{color}║{Colors.END} {line}{' ' * max(0, padding)} {color}║{Colors.END}")

    print(f"{color}╚{border}╝{Colors.END}\n")


def print_separator(char: str = "─", width: int = 70):
    """Print a separator line"""
    print(f"{Colors.DIM}{char * width}{Colors.END}")


def format_number(num: float, precision: int = 4) -> str:
    """Format number with appropriate precision"""
    if abs(num) < 0.0001:
        return f"{num:.2e}"
    elif abs(num) < 1:
        return f"{num:.{precision}f}"
    elif abs(num) < 1000:
        return f"{num:.{min(precision, 2)}f}"
    else:
        return f"{num:,.0f}"


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


def truncate_text(text: str, max_len: int = 50) -> str:
    """Truncate text with ellipsis"""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


# ============================================================================
# RLVR METRICS TRACKER
# ============================================================================

class RLVRMetricsTracker:
    """
    Comprehensive metrics tracker for RLVR training.

    Tracks:
    - Rewards (cumulative, mean, std, min, max)
    - Advantages
    - Policy loss and entropy
    - Tool call statistics
    - Task solving accuracy
    - Token-level and trace-level metrics
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Reset all metrics"""
        # Reward metrics
        self.rewards = deque(maxlen=self.window_size)
        self.cumulative_reward = 0.0
        self.total_episodes = 0

        # Advantage metrics
        self.advantages = deque(maxlen=self.window_size)

        # Loss metrics
        self.policy_losses = deque(maxlen=self.window_size)
        self.entropy_losses = deque(maxlen=self.window_size)
        self.total_losses = deque(maxlen=self.window_size)

        # Entropy metrics (per-token and per-trace)
        self.token_entropies = deque(maxlen=self.window_size)
        self.trace_entropies = deque(maxlen=self.window_size)

        # Tool call metrics
        self.total_tool_calls = 0
        self.correct_tool_calls = 0
        self.tool_call_attempts = deque(maxlen=self.window_size)
        self.tool_call_successes = deque(maxlen=self.window_size)

        # Task metrics
        self.tasks_attempted = 0
        self.tasks_solved = 0
        self.task_accuracies = deque(maxlen=self.window_size)

        # Gradient metrics
        self.gradient_norms = deque(maxlen=self.window_size)

        # Response metrics
        self.response_lengths = deque(maxlen=self.window_size)
        self.num_turns = deque(maxlen=self.window_size)

        # Timing
        self.start_time = time.time()
        self.step_times = deque(maxlen=self.window_size)

        # Examples for logging
        self.best_examples: List[Dict] = []
        self.worst_examples: List[Dict] = []
        self.recent_examples: List[Dict] = []

    def update_reward(self, reward: float, advantage: float = None):
        """Update reward metrics"""
        self.rewards.append(reward)
        self.cumulative_reward += reward
        self.total_episodes += 1

        if advantage is not None:
            self.advantages.append(advantage)

    def update_loss(self, policy_loss: float, entropy_loss: float = 0.0, total_loss: float = None):
        """Update loss metrics"""
        self.policy_losses.append(policy_loss)
        self.entropy_losses.append(entropy_loss)
        if total_loss is not None:
            self.total_losses.append(total_loss)
        else:
            self.total_losses.append(policy_loss + entropy_loss)

    def update_entropy(self, token_entropy: float = None, trace_entropy: float = None):
        """Update entropy metrics"""
        if token_entropy is not None:
            self.token_entropies.append(token_entropy)
        if trace_entropy is not None:
            self.trace_entropies.append(trace_entropy)

    def update_tool_calls(self, num_calls: int, num_correct: int):
        """Update tool call metrics"""
        self.total_tool_calls += num_calls
        self.correct_tool_calls += num_correct
        self.tool_call_attempts.append(num_calls)
        self.tool_call_successes.append(num_correct)

    def update_task(self, solved: bool):
        """Update task metrics"""
        self.tasks_attempted += 1
        if solved:
            self.tasks_solved += 1
        self.task_accuracies.append(1.0 if solved else 0.0)

    def update_gradient_norm(self, grad_norm: float):
        """Update gradient norm"""
        self.gradient_norms.append(grad_norm)

    def update_response(self, length: int, turns: int = 1):
        """Update response metrics"""
        self.response_lengths.append(length)
        self.num_turns.append(turns)

    def update_step_time(self, step_time: float):
        """Update step timing"""
        self.step_times.append(step_time)

    def add_example(self, prompt: str, response: str, reward: float, tool_calls: List[str] = None):
        """Add example for logging"""
        example = {
            "prompt": prompt[:200],
            "response": response[:300],
            "reward": reward,
            "tool_calls": tool_calls or [],
            "timestamp": time.time(),
        }

        self.recent_examples.append(example)
        if len(self.recent_examples) > 5:
            self.recent_examples.pop(0)

        # Track best/worst
        if len(self.best_examples) < 3 or reward > min(e["reward"] for e in self.best_examples):
            self.best_examples.append(example)
            self.best_examples = sorted(self.best_examples, key=lambda x: x["reward"], reverse=True)[:3]

        if len(self.worst_examples) < 3 or reward < max(e["reward"] for e in self.worst_examples):
            self.worst_examples.append(example)
            self.worst_examples = sorted(self.worst_examples, key=lambda x: x["reward"])[:3]

    # ==================== Properties ====================

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def std_reward(self) -> float:
        if len(self.rewards) < 2:
            return 0.0
        mean = self.mean_reward
        return math.sqrt(sum((r - mean) ** 2 for r in self.rewards) / len(self.rewards))

    @property
    def min_reward(self) -> float:
        return min(self.rewards) if self.rewards else 0.0

    @property
    def max_reward(self) -> float:
        return max(self.rewards) if self.rewards else 0.0

    @property
    def mean_advantage(self) -> float:
        return sum(self.advantages) / len(self.advantages) if self.advantages else 0.0

    @property
    def mean_policy_loss(self) -> float:
        return sum(self.policy_losses) / len(self.policy_losses) if self.policy_losses else 0.0

    @property
    def mean_entropy_loss(self) -> float:
        return sum(self.entropy_losses) / len(self.entropy_losses) if self.entropy_losses else 0.0

    @property
    def mean_total_loss(self) -> float:
        return sum(self.total_losses) / len(self.total_losses) if self.total_losses else 0.0

    @property
    def mean_token_entropy(self) -> float:
        return sum(self.token_entropies) / len(self.token_entropies) if self.token_entropies else 0.0

    @property
    def mean_trace_entropy(self) -> float:
        return sum(self.trace_entropies) / len(self.trace_entropies) if self.trace_entropies else 0.0

    @property
    def tool_call_accuracy(self) -> float:
        if self.total_tool_calls == 0:
            return 0.0
        return self.correct_tool_calls / self.total_tool_calls

    @property
    def recent_tool_accuracy(self) -> float:
        attempts = sum(self.tool_call_attempts)
        successes = sum(self.tool_call_successes)
        return successes / attempts if attempts > 0 else 0.0

    @property
    def task_accuracy(self) -> float:
        if self.tasks_attempted == 0:
            return 0.0
        return self.tasks_solved / self.tasks_attempted

    @property
    def recent_task_accuracy(self) -> float:
        return sum(self.task_accuracies) / len(self.task_accuracies) if self.task_accuracies else 0.0

    @property
    def mean_gradient_norm(self) -> float:
        return sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0.0

    @property
    def mean_response_length(self) -> float:
        return sum(self.response_lengths) / len(self.response_lengths) if self.response_lengths else 0.0

    @property
    def mean_turns(self) -> float:
        return sum(self.num_turns) / len(self.num_turns) if self.num_turns else 0.0

    @property
    def throughput(self) -> float:
        """Episodes per second"""
        elapsed = time.time() - self.start_time
        return self.total_episodes / elapsed if elapsed > 0 else 0.0

    @property
    def mean_step_time(self) -> float:
        return sum(self.step_times) / len(self.step_times) if self.step_times else 0.0

    def get_summary(self) -> Dict[str, float]:
        """Get all metrics as dict"""
        return {
            # Rewards
            "reward_mean": self.mean_reward,
            "reward_std": self.std_reward,
            "reward_min": self.min_reward,
            "reward_max": self.max_reward,
            "reward_cumulative": self.cumulative_reward,

            # Advantages
            "advantage_mean": self.mean_advantage,

            # Losses
            "policy_loss": self.mean_policy_loss,
            "entropy_loss": self.mean_entropy_loss,
            "total_loss": self.mean_total_loss,

            # Entropy
            "token_entropy": self.mean_token_entropy,
            "trace_entropy": self.mean_trace_entropy,

            # Tool calls
            "tool_accuracy": self.tool_call_accuracy,
            "tool_accuracy_recent": self.recent_tool_accuracy,
            "total_tool_calls": self.total_tool_calls,
            "correct_tool_calls": self.correct_tool_calls,

            # Tasks
            "task_accuracy": self.task_accuracy,
            "task_accuracy_recent": self.recent_task_accuracy,
            "tasks_attempted": self.tasks_attempted,
            "tasks_solved": self.tasks_solved,

            # Training
            "gradient_norm": self.mean_gradient_norm,
            "response_length": self.mean_response_length,
            "mean_turns": self.mean_turns,
            "total_episodes": self.total_episodes,
            "throughput": self.throughput,
        }


# ============================================================================
# TOOL CALL PARSER
# ============================================================================

class ToolCallParser:
    """Parse and validate tool calls from model responses"""

    TOOL_CALL_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    FUNCTION_CALL_PATTERN = re.compile(r'"function_call"\s*:\s*\{([^}]+)\}', re.DOTALL)

    @classmethod
    def extract_tool_calls(cls, text: str) -> List[Dict]:
        """Extract tool calls from text"""
        tool_calls = []

        # Try <tool_call> format
        matches = cls.TOOL_CALL_PATTERN.findall(text)
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                pass

        # Try function_call format
        if not tool_calls:
            matches = cls.FUNCTION_CALL_PATTERN.findall(text)
            for match in matches:
                try:
                    tool_call = json.loads("{" + match + "}")
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    pass

        return tool_calls

    @classmethod
    def compare_tool_calls(cls, expected: List[Dict], actual: List[Dict]) -> Tuple[int, int]:
        """
        Compare expected and actual tool calls.
        Returns (num_correct, total_expected)
        """
        if not expected:
            return (0, 0)

        correct = 0
        for exp in expected:
            exp_name = exp.get("name", exp.get("function", {}).get("name", ""))
            for act in actual:
                act_name = act.get("name", act.get("function", {}).get("name", ""))
                if exp_name == act_name:
                    correct += 1
                    break

        return (correct, len(expected))


# ============================================================================
# CONFIGURATION
# ============================================================================

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
    baseline_type: str = "running_mean"  # mean, running_mean, none
    entropy_coef: float = 0.01
    reward_scale: float = 1.0
    gamma: float = 0.99  # Discount factor

    # Logging
    show_examples: bool = True
    example_log_interval: int = 50


# ============================================================================
# TRAJECTORY CLIENT
# ============================================================================

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
                return None
            else:
                return None
        except requests.RequestException:
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
            pass

    def health_check(self) -> bool:
        """Check if API is available"""
        try:
            response = self.session.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False


# ============================================================================
# ATROPOS TRAINER
# ============================================================================

class AtroposTrainer:
    """
    Trainer that pulls trajectories from Atropos API and performs
    policy gradient training with comprehensive metrics tracking.
    """

    def __init__(self, config: AtroposTrainerConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Initialize trajectory client
        self.trajectory_client = AtroposTrajectoryClient(config.trajectory_api_url)

        # Metrics tracker
        self.metrics = RLVRMetricsTracker(window_size=100)

        # Load model and tokenizer
        self._setup_model()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # Running baseline for advantage estimation
        self.running_baseline = 0.0
        self.baseline_momentum = 0.99

        # Global step counter
        self.global_step = 0

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Logging setup
        self.log_file = open(
            os.path.join(config.output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            "w"
        )

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

    def compute_entropy(self, logits: torch.Tensor) -> Tuple[float, float]:
        """
        Compute token-level and trace-level entropy.

        Returns:
            (mean_token_entropy, trace_entropy)
        """
        # Token-level entropy: H = -sum(p * log(p))
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        token_entropy = -(probs * log_probs).sum(dim=-1)
        mean_token_entropy = token_entropy.mean().item()

        # Trace entropy: entropy of the whole sequence
        trace_entropy = token_entropy.sum().item()

        return mean_token_entropy, trace_entropy

    def compute_policy_gradient_loss(
        self,
        prompts: List[str],
        responses: List[str],
        rewards: List[float],
        expected_tool_calls: List[List[Dict]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute policy gradient loss with detailed metrics.

        Returns:
            (loss, metrics_dict)
        """
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_entropy_loss = torch.tensor(0.0, device=self.device)
        num_valid = 0

        batch_metrics = {
            "token_entropies": [],
            "trace_entropies": [],
            "advantages": [],
            "response_lengths": [],
        }

        # Compute baseline
        if self.config.baseline_type == "mean":
            baseline = sum(rewards) / len(rewards) if rewards else 0.0
        elif self.config.baseline_type == "running_mean":
            baseline = self.running_baseline
        else:
            baseline = 0.0

        for i, (prompt, response, reward) in enumerate(zip(prompts, responses, rewards)):
            # Tokenize
            full_text = prompt + response
            try:
                encoded = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1536,
                    padding=False,
                ).to(self.device)
            except Exception:
                continue

            prompt_encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=False,
            )
            prompt_len = prompt_encoded["input_ids"].shape[1]
            response_len = encoded["input_ids"].shape[1] - prompt_len

            if response_len <= 0:
                continue

            # Forward pass
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )

                # Get logits for response tokens
                logits = outputs.logits[:, prompt_len - 1:-1, :]
                target_ids = encoded["input_ids"][:, prompt_len:]

                if target_ids.shape[1] == 0:
                    continue

                # Compute log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

                # Sum log probs for policy gradient
                response_log_prob = token_log_probs.sum()

                # Compute advantage
                advantage = (reward - baseline) * self.config.reward_scale
                batch_metrics["advantages"].append(advantage)

                # Policy gradient loss: -advantage * log_prob
                policy_loss = -advantage * response_log_prob

                # Compute entropy bonus
                token_entropy, trace_entropy = self.compute_entropy(logits)
                entropy_loss = -self.config.entropy_coef * token_entropy

                batch_metrics["token_entropies"].append(token_entropy)
                batch_metrics["trace_entropies"].append(trace_entropy)
                batch_metrics["response_lengths"].append(response_len)

                total_policy_loss = total_policy_loss + policy_loss
                total_entropy_loss = total_entropy_loss + entropy_loss
                num_valid += 1

                # Update metrics
                self.metrics.update_reward(reward, advantage)
                self.metrics.update_entropy(token_entropy, trace_entropy)
                self.metrics.update_response(response_len)

                # Parse and track tool calls
                actual_tool_calls = ToolCallParser.extract_tool_calls(response)
                expected = expected_tool_calls[i] if expected_tool_calls and i < len(expected_tool_calls) else []
                correct, total = ToolCallParser.compare_tool_calls(expected, actual_tool_calls)
                self.metrics.update_tool_calls(len(actual_tool_calls), correct)

                # Track task success (reward > 0 means solved)
                self.metrics.update_task(reward > 0)

                # Add example
                self.metrics.add_example(
                    prompt=prompt,
                    response=response,
                    reward=reward,
                    tool_calls=[tc.get("name", "unknown") for tc in actual_tool_calls]
                )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

        if num_valid > 0:
            total_policy_loss = total_policy_loss / num_valid
            total_entropy_loss = total_entropy_loss / num_valid

        # Update running baseline
        if rewards:
            batch_mean = sum(rewards) / len(rewards)
            self.running_baseline = (
                self.baseline_momentum * self.running_baseline +
                (1 - self.baseline_momentum) * batch_mean
            )

        # Aggregate batch metrics
        metrics = {
            "policy_loss": total_policy_loss.item() if isinstance(total_policy_loss, torch.Tensor) else total_policy_loss,
            "entropy_loss": total_entropy_loss.item() if isinstance(total_entropy_loss, torch.Tensor) else total_entropy_loss,
            "mean_advantage": sum(batch_metrics["advantages"]) / len(batch_metrics["advantages"]) if batch_metrics["advantages"] else 0.0,
            "mean_token_entropy": sum(batch_metrics["token_entropies"]) / len(batch_metrics["token_entropies"]) if batch_metrics["token_entropies"] else 0.0,
            "mean_response_length": sum(batch_metrics["response_lengths"]) / len(batch_metrics["response_lengths"]) if batch_metrics["response_lengths"] else 0.0,
            "num_valid": num_valid,
        }

        total_loss = total_policy_loss + total_entropy_loss
        self.metrics.update_loss(
            policy_loss=metrics["policy_loss"],
            entropy_loss=metrics["entropy_loss"],
            total_loss=total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        )

        return total_loss, metrics

    def train_step(self, batch: List[Dict]) -> Dict[str, float]:
        """Perform one training step on a batch of trajectories"""
        step_start = time.time()

        prompts = []
        responses = []
        rewards = []
        expected_tool_calls = []

        for trajectory in batch:
            prompt = trajectory.get("prompt", "")
            response = trajectory.get("response", "")
            reward = trajectory.get("reward", 0.0)
            expected = trajectory.get("expected_tool_calls", [])

            if prompt and response:
                prompts.append(prompt)
                responses.append(response)
                rewards.append(reward)
                expected_tool_calls.append(expected)

        if not prompts:
            return {"loss": 0.0, "reward": 0.0}

        # Compute loss
        loss, batch_metrics = self.compute_policy_gradient_loss(
            prompts, responses, rewards, expected_tool_calls
        )

        # Backward pass
        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()

        # Gradient accumulation step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            ).item()
            self.metrics.update_gradient_norm(grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

        # Update timing
        step_time = time.time() - step_start
        self.metrics.update_step_time(step_time)

        return {
            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "num_samples": len(rewards),
            **batch_metrics,
        }

    def log_detailed_metrics(self, step: int, eta_seconds: float):
        """Log detailed metrics to console"""
        summary = self.metrics.get_summary()

        print()
        print_separator("═", 80)
        print(
            f"  {Colors.BOLD}Step {step}/{self.config.total_steps}{Colors.END} "
            f"({Colors.DIM}{step / self.config.total_steps * 100:.1f}%{Colors.END}) "
            f"│ ETA: {Colors.CYAN}{format_time(eta_seconds)}{Colors.END} "
            f"│ Elapsed: {format_time(time.time() - self.metrics.start_time)}"
        )
        print_separator("─", 80)

        # Rewards section
        print(f"\n  {Colors.BOLD}{Colors.GREEN}📊 REWARDS{Colors.END}")
        print(f"    Mean:       {Colors.reward(summary['reward_mean'])}")
        print(f"    Std:        {format_number(summary['reward_std'])}")
        print(f"    Min/Max:    {Colors.reward(summary['reward_min'])} / {Colors.reward(summary['reward_max'])}")
        print(f"    Cumulative: {Colors.BOLD}{summary['reward_cumulative']:.2f}{Colors.END}")

        # Losses section
        print(f"\n  {Colors.BOLD}{Colors.BLUE}📉 LOSSES{Colors.END}")
        print(f"    Policy Loss:  {format_number(summary['policy_loss'])}")
        print(f"    Entropy Loss: {format_number(summary['entropy_loss'])}")
        print(f"    Total Loss:   {Colors.BOLD}{format_number(summary['total_loss'])}{Colors.END}")

        # Entropy section
        print(f"\n  {Colors.BOLD}{Colors.MAGENTA}🎲 ENTROPY{Colors.END}")
        print(f"    Token Entropy: {format_number(summary['token_entropy'])}")
        print(f"    Trace Entropy: {format_number(summary['trace_entropy'])}")
        print(f"    Advantage:     {format_number(summary['advantage_mean'])}")

        # Tool calls section
        print(f"\n  {Colors.BOLD}{Colors.YELLOW}🔧 TOOL CALLS{Colors.END}")
        tool_acc_color = Colors.GREEN if summary['tool_accuracy'] > 0.5 else Colors.YELLOW if summary['tool_accuracy'] > 0.2 else Colors.RED
        print(f"    Accuracy (all):    {tool_acc_color}{summary['tool_accuracy']*100:.1f}%{Colors.END} ({summary['correct_tool_calls']}/{summary['total_tool_calls']})")
        print(f"    Accuracy (recent): {summary['tool_accuracy_recent']*100:.1f}%")

        # Task accuracy section
        print(f"\n  {Colors.BOLD}{Colors.CYAN}✓ TASK ACCURACY{Colors.END}")
        task_acc_color = Colors.GREEN if summary['task_accuracy'] > 0.5 else Colors.YELLOW if summary['task_accuracy'] > 0.2 else Colors.RED
        print(f"    Accuracy (all):    {task_acc_color}{summary['task_accuracy']*100:.1f}%{Colors.END} ({summary['tasks_solved']}/{summary['tasks_attempted']})")
        print(f"    Accuracy (recent): {summary['task_accuracy_recent']*100:.1f}%")

        # Training stats
        print(f"\n  {Colors.BOLD}⚡ TRAINING{Colors.END}")
        print(f"    Gradient Norm:   {format_number(summary['gradient_norm'])}")
        print(f"    Response Length: {summary['response_length']:.0f} tokens")
        print(f"    Mean Turns:      {summary['mean_turns']:.1f}")
        print(f"    Throughput:      {Colors.BOLD}{summary['throughput']:.2f}{Colors.END} episodes/sec")
        print(f"    Total Episodes:  {summary['total_episodes']:,}")

        # GPU memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"\n  {Colors.DIM}GPU: {allocated:.1f}GB allocated / {reserved:.1f}GB reserved{Colors.END}")

        print_separator("═", 80)

    def log_examples(self, step: int):
        """Log example prompts and responses"""
        if not self.config.show_examples:
            return

        if self.metrics.recent_examples:
            print_box(
                "📝 RECENT EXAMPLE",
                [
                    f"{Colors.DIM}Prompt:{Colors.END}",
                    f"  {truncate_text(self.metrics.recent_examples[-1]['prompt'], 60)}",
                    "",
                    f"{Colors.DIM}Response:{Colors.END}",
                    f"  {truncate_text(self.metrics.recent_examples[-1]['response'], 60)}",
                    "",
                    f"Reward: {Colors.reward(self.metrics.recent_examples[-1]['reward'])}",
                    f"Tools: {', '.join(self.metrics.recent_examples[-1]['tool_calls']) or 'none'}",
                ],
                width=70,
                color=Colors.DIM
            )

        if self.metrics.best_examples:
            print_box(
                f"🏆 BEST EXAMPLE (reward: {self.metrics.best_examples[0]['reward']:.3f})",
                [
                    f"{Colors.DIM}Prompt:{Colors.END} {truncate_text(self.metrics.best_examples[0]['prompt'], 55)}",
                    f"{Colors.DIM}Response:{Colors.END} {truncate_text(self.metrics.best_examples[0]['response'], 55)}",
                ],
                width=70,
                color=Colors.GREEN
            )

        if self.metrics.worst_examples:
            print_box(
                f"💀 WORST EXAMPLE (reward: {self.metrics.worst_examples[0]['reward']:.3f})",
                [
                    f"{Colors.DIM}Prompt:{Colors.END} {truncate_text(self.metrics.worst_examples[0]['prompt'], 55)}",
                    f"{Colors.DIM}Response:{Colors.END} {truncate_text(self.metrics.worst_examples[0]['response'], 55)}",
                ],
                width=70,
                color=Colors.RED
            )

    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save LoRA weights
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state and metrics
        state = {
            "step": step,
            "running_baseline": self.running_baseline,
            "metrics": self.metrics.get_summary(),
        }
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        print(Colors.success(f"✓ Saved checkpoint to {checkpoint_dir}"))

    def train(self):
        """Main training loop"""
        # Print header
        print_box(
            "ATROPOS RLVR TRAINER",
            [
                f"Model: {Colors.BOLD}{self.config.model_name}{Colors.END}",
                f"Trajectory API: {self.config.trajectory_api_url}",
                "",
                f"Total steps: {Colors.CYAN}{self.config.total_steps:,}{Colors.END}",
                f"Batch size: {self.config.batch_size}",
                f"Learning rate: {self.config.learning_rate}",
                f"Entropy coef: {self.config.entropy_coef}",
                "",
                f"Baseline: {self.config.baseline_type}",
                f"Reward scale: {self.config.reward_scale}",
            ],
            width=70
        )

        # Check API health
        if not self.trajectory_client.health_check():
            print(Colors.warning("⚠ Trajectory API not available. Waiting..."))

        progress_bar = tqdm(
            range(self.config.total_steps),
            desc=f"{Colors.CYAN}RLVR Training{Colors.END}",
            file=sys.stdout,
            bar_format="{l_bar}{bar:30}{r_bar}",
            ncols=100
        )

        self.optimizer.zero_grad()
        training_start = time.time()

        for step in progress_bar:
            self.global_step = step

            # Get batch from trajectory API
            batch = None
            while batch is None:
                batch = self.trajectory_client.get_batch(self.config.batch_size)
                if batch is None:
                    time.sleep(self.config.poll_interval)

            # Train step
            step_metrics = self.train_step(batch)

            # Update progress bar
            summary = self.metrics.get_summary()
            progress_bar.set_postfix_str(
                f"R={Colors.reward(summary['reward_mean'])} | "
                f"L={format_number(summary['total_loss'])} | "
                f"Tool={summary['tool_accuracy']*100:.0f}% | "
                f"Task={summary['task_accuracy']*100:.0f}% | "
                f"H={format_number(summary['token_entropy'])}"
            )

            # Detailed logging
            if (step + 1) % self.config.log_steps == 0:
                elapsed = time.time() - training_start
                steps_remaining = self.config.total_steps - step - 1
                eta = (elapsed / (step + 1)) * steps_remaining

                self.log_detailed_metrics(step + 1, eta)

                # Report to API
                self.trajectory_client.report_training_step(step + 1, summary)

            # Log examples
            if (step + 1) % self.config.example_log_interval == 0:
                self.log_examples(step + 1)

            # Save checkpoint
            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(step + 1)

        # Final save
        self.save_checkpoint(self.config.total_steps)

        # Print final summary
        total_time = time.time() - training_start
        final_summary = self.metrics.get_summary()

        print_box(
            "🎉 TRAINING COMPLETE",
            [
                f"Total time: {Colors.BOLD}{format_time(total_time)}{Colors.END}",
                f"Total episodes: {final_summary['total_episodes']:,}",
                "",
                f"{Colors.BOLD}Final Metrics:{Colors.END}",
                f"  Mean Reward:    {Colors.reward(final_summary['reward_mean'])}",
                f"  Cumulative:     {final_summary['reward_cumulative']:.2f}",
                f"  Tool Accuracy:  {final_summary['tool_accuracy']*100:.1f}%",
                f"  Task Accuracy:  {final_summary['task_accuracy']*100:.1f}%",
                "",
                f"Checkpoint: {self.config.output_dir}/checkpoint-{self.config.total_steps}",
            ],
            width=70,
            color=Colors.GREEN
        )

        self.log_file.close()


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
    parser.add_argument("--log-steps", type=int, default=10,
                        help="Log metrics every N steps")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")

    # Quantization
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")

    # Policy gradient
    parser.add_argument("--baseline-type", type=str, default="running_mean",
                        choices=["mean", "running_mean", "none"],
                        help="Baseline type for advantage computation")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Reward scaling factor")

    # Logging
    parser.add_argument("--no-examples", action="store_true",
                        help="Disable example logging")
    parser.add_argument("--example-interval", type=int, default=50,
                        help="Log examples every N steps")

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
        log_steps=args.log_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_lora=not args.no_lora,
        use_4bit=not args.no_4bit,
        baseline_type=args.baseline_type,
        entropy_coef=args.entropy_coef,
        reward_scale=args.reward_scale,
        show_examples=not args.no_examples,
        example_log_interval=args.example_interval,
    )

    # Create trainer and run
    trainer = AtroposTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
