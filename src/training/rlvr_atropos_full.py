#!/usr/bin/env python3
"""
Full Atropos RLVR Pipeline for GigaChat

Complete RLVR training pipeline following NousResearch/atropos architecture:
- Trajectory API server (manages rollout queue)
- Environment (generates rollouts, computes rewards, pushes to API)
- Trainer (pulls batches, does policy gradient training)
- vLLM (inference server for rollout generation)

Usage:
    # Start all components:
    python rlvr_atropos_full.py \
        --model ai-sage/GigaChat3-10B-A1.8B-bf16 \
        --output-dir ./outputs/rlvr-full \
        --vllm-port 8001 \
        --api-port 8000
"""

import argparse
import asyncio
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

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

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{text}{Colors.END}"


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
# RLVR METRICS TRACKER
# ============================================================================

class RLVRMetricsTracker:
    """Comprehensive metrics tracker for RLVR training"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reset()

    def reset(self):
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

        # Entropy metrics
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

        # Timing
        self.start_time = time.time()
        self.step_times = deque(maxlen=self.window_size)

        # Examples
        self.best_examples: List[Dict] = []
        self.worst_examples: List[Dict] = []
        self.recent_examples: List[Dict] = []

    def update(self, reward: float, advantage: float = 0.0, policy_loss: float = 0.0,
               entropy: float = 0.0, tool_calls: int = 0, correct_tools: int = 0,
               solved: bool = False, response_len: int = 0, grad_norm: float = 0.0,
               prompt: str = "", response: str = ""):
        self.rewards.append(reward)
        self.cumulative_reward += reward
        self.total_episodes += 1

        self.advantages.append(advantage)
        self.policy_losses.append(policy_loss)
        self.token_entropies.append(entropy)
        self.response_lengths.append(response_len)

        if grad_norm > 0:
            self.gradient_norms.append(grad_norm)

        self.total_tool_calls += tool_calls
        self.correct_tool_calls += correct_tools
        self.tool_call_attempts.append(tool_calls)
        self.tool_call_successes.append(correct_tools)

        self.tasks_attempted += 1
        if solved:
            self.tasks_solved += 1
        self.task_accuracies.append(1.0 if solved else 0.0)

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
    def min_reward(self):
        return min(self.rewards) if self.rewards else 0.0

    @property
    def max_reward(self):
        return max(self.rewards) if self.rewards else 0.0

    @property
    def mean_advantage(self):
        return sum(self.advantages) / len(self.advantages) if self.advantages else 0.0

    @property
    def mean_policy_loss(self):
        return sum(self.policy_losses) / len(self.policy_losses) if self.policy_losses else 0.0

    @property
    def mean_entropy(self):
        return sum(self.token_entropies) / len(self.token_entropies) if self.token_entropies else 0.0

    @property
    def tool_accuracy(self):
        return self.correct_tool_calls / self.total_tool_calls if self.total_tool_calls > 0 else 0.0

    @property
    def recent_tool_accuracy(self):
        attempts = sum(self.tool_call_attempts)
        successes = sum(self.tool_call_successes)
        return successes / attempts if attempts > 0 else 0.0

    @property
    def task_accuracy(self):
        return self.tasks_solved / self.tasks_attempted if self.tasks_attempted > 0 else 0.0

    @property
    def recent_task_accuracy(self):
        return sum(self.task_accuracies) / len(self.task_accuracies) if self.task_accuracies else 0.0

    @property
    def mean_gradient_norm(self):
        return sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0.0

    @property
    def mean_response_length(self):
        return sum(self.response_lengths) / len(self.response_lengths) if self.response_lengths else 0.0

    @property
    def throughput(self):
        elapsed = time.time() - self.start_time
        return self.total_episodes / elapsed if elapsed > 0 else 0.0

    def get_summary(self):
        return {
            "reward_mean": self.mean_reward,
            "reward_std": self.std_reward,
            "reward_min": self.min_reward,
            "reward_max": self.max_reward,
            "reward_cumulative": self.cumulative_reward,
            "advantage_mean": self.mean_advantage,
            "policy_loss": self.mean_policy_loss,
            "entropy": self.mean_entropy,
            "tool_accuracy": self.tool_accuracy,
            "tool_accuracy_recent": self.recent_tool_accuracy,
            "total_tool_calls": self.total_tool_calls,
            "correct_tool_calls": self.correct_tool_calls,
            "task_accuracy": self.task_accuracy,
            "task_accuracy_recent": self.recent_task_accuracy,
            "tasks_solved": self.tasks_solved,
            "tasks_attempted": self.tasks_attempted,
            "gradient_norm": self.mean_gradient_norm,
            "response_length": self.mean_response_length,
            "total_episodes": self.total_episodes,
            "throughput": self.throughput,
        }


# ============================================================================
# TRAJECTORY API SERVER
# ============================================================================

class TrajectoryQueue:
    """Thread-safe queue for trajectories"""

    def __init__(self, maxsize: int = 10000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        self.stats = {
            "total_pushed": 0,
            "total_pulled": 0,
        }

    def push(self, trajectories: List[Dict]):
        with self.lock:
            for traj in trajectories:
                try:
                    self.queue.put_nowait(traj)
                    self.stats["total_pushed"] += 1
                except queue.Full:
                    # Drop oldest if full
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(traj)
                        self.stats["total_pushed"] += 1
                    except:
                        pass

    def pull(self, batch_size: int) -> List[Dict]:
        batch = []
        with self.lock:
            for _ in range(batch_size):
                try:
                    traj = self.queue.get_nowait()
                    batch.append(traj)
                    self.stats["total_pulled"] += 1
                except queue.Empty:
                    break
        return batch

    def size(self) -> int:
        return self.queue.qsize()


# Global trajectory queue
TRAJECTORY_QUEUE = TrajectoryQueue()


class TrajectoryAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for Trajectory API"""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json({"status": "ok", "queue_size": TRAJECTORY_QUEUE.size()})

        elif path == "/trajectories/batch":
            params = parse_qs(parsed.query)
            batch_size = int(params.get("batch_size", [8])[0])
            batch = TRAJECTORY_QUEUE.pull(batch_size)
            if batch:
                self._send_json(batch)
            else:
                self.send_response(204)  # No Content
                self.end_headers()

        elif path == "/stats":
            self._send_json({
                "queue_size": TRAJECTORY_QUEUE.size(),
                **TRAJECTORY_QUEUE.stats,
            })

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        if path == "/trajectories":
            try:
                data = json.loads(body)
                trajectories = data if isinstance(data, list) else [data]
                TRAJECTORY_QUEUE.push(trajectories)
                self._send_json({"status": "ok", "pushed": len(trajectories)})
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, 400)

        elif path == "/training/metrics":
            # Just acknowledge training metrics
            self._send_json({"status": "ok"})

        else:
            self.send_response(404)
            self.end_headers()


def run_trajectory_api(port: int = 8000):
    """Run the Trajectory API server"""
    server = HTTPServer(('0.0.0.0', port), TrajectoryAPIHandler)
    print(f"  {Colors.success('✓')} Trajectory API running on port {port}")
    server.serve_forever()


# ============================================================================
# REWARD COMPUTATION
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
    def compute_reward(
        cls,
        response: str,
        expected_tools: List[Dict] = None,
        expected_answer: str = None
    ) -> Tuple[float, Dict]:
        """
        Compute reward for a response.

        Reward components:
        - Tool call format: +0.2 if valid JSON tool call
        - Tool name match: +0.3 if matches expected
        - Arguments match: +0.3 if arguments correct
        - Content similarity: +0.2 if response similar to expected
        - Penalty for no tool call when expected: -0.5
        """
        info = {
            "has_tool_call": False,
            "valid_json": False,
            "name_match": False,
            "args_match": False,
            "content_match": False,
            "tool_calls": [],
            "num_tool_calls": 0,
            "num_correct": 0,
        }

        actual_tools = cls.extract_tool_calls(response)
        info["tool_calls"] = actual_tools
        info["has_tool_call"] = len(actual_tools) > 0
        info["num_tool_calls"] = len(actual_tools)

        reward = 0.0

        # Check tool calls
        if actual_tools:
            info["valid_json"] = True
            reward += 0.2  # Valid tool call format

            if expected_tools:
                expected_names = {t.get("name", "") for t in expected_tools}
                actual_names = {t.get("name", "") for t in actual_tools}

                matches = expected_names & actual_names
                if matches:
                    info["name_match"] = True
                    info["num_correct"] = len(matches)
                    reward += 0.3  # Tool name match

                    # Check arguments
                    for exp in expected_tools:
                        for act in actual_tools:
                            if exp.get("name") == act.get("name"):
                                exp_args = exp.get("arguments", {})
                                act_args = act.get("arguments", {})
                                if exp_args == act_args:
                                    info["args_match"] = True
                                    reward += 0.3  # Arguments match
                                    break
                        if info["args_match"]:
                            break
            else:
                reward += 0.1  # Tool call without expected (exploratory)
        else:
            # No tool call in response
            if expected_tools:
                reward = -0.5  # Penalty for missing tool call
            else:
                reward = 0.0

        # Check content similarity with expected answer
        if expected_answer and len(expected_answer) > 10:
            # Simple similarity: check if key phrases match
            response_lower = response.lower()
            expected_lower = expected_answer.lower()

            # Extract key words from expected answer
            expected_words = set(expected_lower.split())
            response_words = set(response_lower.split())

            # Calculate overlap
            if expected_words:
                overlap = len(expected_words & response_words) / len(expected_words)
                if overlap > 0.5:
                    info["content_match"] = True
                    reward += 0.2 * overlap  # Up to +0.2 for content match

        return min(max(reward, -1.0), 1.0), info


# ============================================================================
# ENVIRONMENT (Rollout Generator)
# ============================================================================

class RolloutEnvironment:
    """
    Environment that generates rollouts using vLLM and pushes to Trajectory API.
    """

    def __init__(
        self,
        vllm_url: str,
        api_url: str,
        dataset_name: str,
        model_name: str,
        max_samples: int = 5000,
        group_size: int = 4,  # Number of rollouts per prompt
        max_new_tokens: int = 384,  # Reduced for memory
        temperature: float = 0.7,
        max_queue_size: int = 64,  # Throttling: pause when queue exceeds this
        generation_pause_time: float = 2.0,  # Seconds to pause
    ):
        self.vllm_url = vllm_url
        self.api_url = api_url
        self.model_name = model_name
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_queue_size = max_queue_size
        self.generation_pause_time = generation_pause_time

        # Setup vLLM client
        self.vllm_client = OpenAI(base_url=vllm_url, api_key="EMPTY")

        # Load dataset
        self.samples = []
        self._load_dataset(dataset_name, max_samples)

        # API session
        import requests
        self.session = requests.Session()

        self.running = False
        self.stats = {"rollouts_generated": 0, "rollouts_pushed": 0}

    def _load_dataset(self, dataset_name: str, max_samples: int):
        """Load dataset"""
        print(f"  Loading dataset: {dataset_name}")
        self.samples = []

        try:
            # Try loading with streaming to handle schema issues
            try:
                print(f"    Trying streaming mode...")
                from huggingface_hub import hf_hub_download
                import json as json_module

                # Download the JSONL file directly
                file_path = hf_hub_download(
                    repo_id=dataset_name,
                    filename="data/tool_calling.jsonl",
                    repo_type="dataset"
                )

                print(f"    Loading from: {file_path}")
                loaded = 0
                errors = 0

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if loaded >= max_samples:
                            break
                        try:
                            sample = json_module.loads(line.strip())
                            # Normalize messages - ensure content is string
                            if "messages" in sample:
                                normalized_messages = []
                                for msg in sample["messages"]:
                                    content = msg.get("content", "")
                                    if isinstance(content, dict):
                                        content = json_module.dumps(content)
                                    elif isinstance(content, list):
                                        content = " ".join(str(c) for c in content)
                                    normalized_messages.append({
                                        "role": msg.get("role", "user"),
                                        "content": str(content)
                                    })
                                sample["messages"] = normalized_messages
                            self.samples.append(sample)
                            loaded += 1
                        except Exception:
                            errors += 1
                            continue

                print(f"    {Colors.success('✓')} Loaded {loaded} samples (skipped {errors} errors)")

            except Exception as e:
                print(f"    {Colors.warning('⚠')} Streaming failed: {type(e).__name__}: {str(e)[:100]}")

            # Try interactive_agent file
            if len(self.samples) < max_samples:
                try:
                    from huggingface_hub import hf_hub_download
                    import json as json_module

                    file_path = hf_hub_download(
                        repo_id=dataset_name,
                        filename="data/interactive_agent.jsonl",
                        repo_type="dataset"
                    )

                    remaining = max_samples - len(self.samples)
                    loaded = 0

                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if loaded >= remaining:
                                break
                            try:
                                sample = json_module.loads(line.strip())
                                if "messages" in sample:
                                    normalized_messages = []
                                    for msg in sample["messages"]:
                                        content = msg.get("content", "")
                                        if isinstance(content, dict):
                                            content = json_module.dumps(content)
                                        elif isinstance(content, list):
                                            content = " ".join(str(c) for c in content)
                                        normalized_messages.append({
                                            "role": msg.get("role", "user"),
                                            "content": str(content)
                                        })
                                    sample["messages"] = normalized_messages
                                self.samples.append(sample)
                                loaded += 1
                            except Exception:
                                continue

                    print(f"    {Colors.success('✓')} Added {loaded} from interactive_agent")

                except Exception as e:
                    print(f"    {Colors.warning('⚠')} interactive_agent failed: {str(e)[:50]}")

            # Final fallback: generate synthetic
            if not self.samples:
                print(f"  {Colors.warning('⚠')} Could not load dataset, using synthetic data")
                self.samples = self._generate_synthetic_samples(max_samples)

            random.shuffle(self.samples)
            print(f"  {Colors.success('✓')} Loaded {len(self.samples)} samples total")

        except Exception as e:
            print(f"  {Colors.error('✗')} Error loading dataset: {type(e).__name__}: {e}")
            print(f"  {Colors.warning('⚠')} Using synthetic data")
            self.samples = self._generate_synthetic_samples(max_samples)

    def _generate_synthetic_samples(self, num_samples: int) -> List[Dict]:
        """Generate synthetic tool-calling samples for testing"""
        tools = [
            {"name": "get_weather", "args": {"city": "Moscow"}},
            {"name": "search_web", "args": {"query": "Python tutorials"}},
            {"name": "calculate", "args": {"expression": "2 + 2"}},
            {"name": "get_time", "args": {"timezone": "UTC"}},
            {"name": "translate", "args": {"text": "Hello", "to": "ru"}},
        ]

        prompts = [
            "What's the weather like in Moscow?",
            "Search for Python tutorials",
            "Calculate 2 + 2",
            "What time is it in UTC?",
            "Translate 'Hello' to Russian",
        ]

        samples = []
        for i in range(num_samples):
            idx = i % len(tools)
            tool = tools[idx]
            samples.append({
                "messages": [
                    {"role": "user", "content": prompts[idx]},
                    {"role": "assistant", "content": f'<tool_call>\n{json.dumps(tool)}\n</tool_call>'}
                ]
            })

        return samples

    def _format_prompt(self, sample: Dict) -> Tuple[str, List[Dict], str]:
        """Format sample into prompt, extract expected tools, and expected answer"""
        messages = sample.get("messages", [])
        if isinstance(messages, str):
            messages = json.loads(messages)

        prompt_parts = []
        expected_tools = []
        expected_answer = ""

        system_prompt = """You are a helpful AI assistant with access to tools. When you need to use a tool, respond with:

<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

Think step by step before making tool calls."""

        prompt_parts.append(f"System: {system_prompt}\n")

        # Process messages and extract expected answer from assistant
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle content that might be dict or list
            if isinstance(content, dict):
                content = json.dumps(content)
            elif isinstance(content, list):
                content = " ".join(str(c) for c in content)

            if role == "system":
                # Use the actual system prompt if provided
                prompt_parts = [f"System: {content}\n"]
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                # Extract expected answer from the last assistant message
                expected_answer = content

                # Extract tool calls from assistant's response
                tools = RewardComputer.extract_tool_calls(content)
                expected_tools.extend(tools)

                # Also check for tool_calls field in the message
                if "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tool_info = {
                                    "name": tc.get("function", {}).get("name", tc.get("name", "")),
                                    "arguments": tc.get("function", {}).get("arguments", tc.get("arguments", {}))
                                }
                                # Parse arguments if string
                                if isinstance(tool_info["arguments"], str):
                                    try:
                                        tool_info["arguments"] = json.loads(tool_info["arguments"])
                                    except:
                                        pass
                                if tool_info["name"]:
                                    expected_tools.append(tool_info)
                break  # Only use first assistant response as target

        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts), expected_tools, expected_answer

    def generate_rollout(self, prompt: str) -> str:
        """Generate a single rollout using vLLM"""
        try:
            response = self.vllm_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                n=1,
            )
            return response.choices[0].text
        except Exception as e:
            return ""

    def push_trajectories(self, trajectories: List[Dict]):
        """Push trajectories to API"""
        try:
            self.session.post(
                f"{self.api_url}/trajectories",
                json=trajectories,
                timeout=10,
            )
            self.stats["rollouts_pushed"] += len(trajectories)
        except:
            # Fallback: push directly to queue
            TRAJECTORY_QUEUE.push(trajectories)
            self.stats["rollouts_pushed"] += len(trajectories)

    def run(self, num_rollouts: int = None):
        """Generate rollouts continuously with throttling"""
        self.running = True
        sample_idx = 0
        throttle_warned = False

        # Check if we have samples
        if not self.samples:
            print(f"  {Colors.error('✗')} No samples available, cannot generate rollouts")
            return

        while self.running:
            if num_rollouts and self.stats["rollouts_generated"] >= num_rollouts:
                break

            # THROTTLING: Pause if queue is too full
            queue_size = TRAJECTORY_QUEUE.size()
            if queue_size > self.max_queue_size:
                if not throttle_warned:
                    print(f"  {Colors.warning('⏸')} Queue full ({queue_size}), throttling generation...")
                    throttle_warned = True
                time.sleep(self.generation_pause_time)
                continue
            elif throttle_warned and queue_size < self.max_queue_size // 2:
                print(f"  {Colors.success('▶')} Queue drained ({queue_size}), resuming generation")
                throttle_warned = False

            # Get sample
            if sample_idx >= len(self.samples):
                sample_idx = 0
                random.shuffle(self.samples)

            sample = self.samples[sample_idx]
            sample_idx += 1

            # Format prompt and extract expected answer
            prompt, expected_tools, expected_answer = self._format_prompt(sample)

            # Generate multiple rollouts per prompt
            trajectories = []
            for _ in range(self.group_size):
                response = self.generate_rollout(prompt)
                if not response:
                    continue

                # Compute reward with expected answer
                reward, info = RewardComputer.compute_reward(
                    response,
                    expected_tools,
                    expected_answer=expected_answer
                )

                trajectory = {
                    "prompt": prompt,
                    "response": response,
                    "reward": reward,
                    "expected_tool_calls": expected_tools,
                    "expected_answer": expected_answer,
                    "info": info,
                }
                trajectories.append(trajectory)
                self.stats["rollouts_generated"] += 1

            # Push to API
            if trajectories:
                self.push_trajectories(trajectories)

    def stop(self):
        self.running = False


# ============================================================================
# TRAINER
# ============================================================================

@dataclass
class RLVRConfig:
    """Configuration for full RLVR training"""

    # Model
    model_name: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"
    output_dir: str = "./outputs/rlvr-full"

    # vLLM
    vllm_url: str = "http://localhost:8001/v1"

    # Trajectory API
    api_url: str = "http://localhost:8000"

    # Dataset
    dataset_name: str = "nvidia/Nemotron-Agentic-v1"
    max_samples: int = 5000

    # Training
    total_steps: int = 2000
    batch_size: int = 4  # Reduced from 8 to save memory
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    gradient_accumulation: int = 4
    steps_per_rollout: int = 2  # Multiple gradient steps per batch (reuse rollouts)

    # Generation (throttling)
    group_size: int = 4  # Number of rollouts per prompt for GRPO-style training
    max_new_tokens: int = 384  # Reduced from 512
    temperature: float = 0.7
    max_queue_size: int = 64  # Pause generation when queue exceeds this
    generation_pause_time: float = 2.0  # Seconds to pause when queue full

    # Policy gradient
    entropy_coef: float = 0.01
    reward_scale: float = 1.0
    baseline_momentum: float = 0.99

    # LoRA (reduced for memory)
    use_lora: bool = True
    lora_r: int = 32  # Reduced from 64
    lora_alpha: int = 64  # Reduced from 128

    # Logging
    log_steps: int = 10
    save_steps: int = 100
    example_interval: int = 50

    # Sequence
    max_seq_length: int = 1536


class FullRLVRTrainer:
    """
    Full RLVR Trainer following Atropos architecture.

    Components:
    1. Trajectory API (manages rollout queue)
    2. Environment (generates rollouts, computes rewards)
    3. Trainer (policy gradient training)
    """

    def __init__(self, config: RLVRConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Metrics
        self.metrics = RLVRMetricsTracker()

        # Running baseline
        self.running_baseline = 0.0

        # Setup
        self._setup_model()

        os.makedirs(config.output_dir, exist_ok=True)

    def _setup_model(self):
        """Setup training model with LoRA"""
        print(f"  Loading model: {self.config.model_name}")

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

        if self.config.use_lora:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=False  # Disable for MoE models
            )
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

        # Explicitly disable gradient checkpointing (causes dtype issues with MoE)
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

        self.model.train()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

    def compute_policy_loss(
        self,
        prompt: str,
        response: str,
        reward: float,
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute policy gradient loss"""
        full_text = prompt + response
        try:
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.device)
        except:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0

        prompt_encoded = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_encoded["input_ids"].shape[1]
        response_len = encoded["input_ids"].shape[1] - prompt_len

        if response_len <= 0:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0

        try:
            # Forward pass without autocast (MoE models have dtype issues)
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )

            logits = outputs.logits[:, prompt_len - 1:-1, :].float()  # Cast to float32
            target_ids = encoded["input_ids"][:, prompt_len:]

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
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0.0

        return total_loss, entropy.item(), advantage

    def train_on_batch(self, batch: List[Dict]) -> Dict[str, float]:
        """Train on a batch of trajectories"""
        total_loss = torch.tensor(0.0, device=self.device)
        batch_rewards = []
        num_valid = 0

        for traj in batch:
            prompt = traj.get("prompt", "")
            response = traj.get("response", "")
            reward = traj.get("reward", 0.0)
            info = traj.get("info", {})

            if not prompt or not response:
                continue

            loss, entropy, advantage = self.compute_policy_loss(prompt, response, reward)

            if loss.requires_grad:
                total_loss = total_loss + loss
                num_valid += 1

            batch_rewards.append(reward)

            # Update metrics
            self.metrics.update(
                reward=reward,
                advantage=advantage,
                policy_loss=loss.item() if isinstance(loss, torch.Tensor) else loss,
                entropy=entropy,
                tool_calls=info.get("num_tool_calls", 0),
                correct_tools=info.get("num_correct", 0),
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
        print(f"    Min/Max:    {Colors.reward(summary['reward_min'])} / {Colors.reward(summary['reward_max'])}")
        print(f"    Cumulative: {Colors.BOLD}{summary['reward_cumulative']:.2f}{Colors.END}")

        print(f"\n  {Colors.BOLD}{Colors.BLUE}📉 TRAINING{Colors.END}")
        print(f"    Policy Loss: {format_number(summary['policy_loss'])}")
        print(f"    Entropy:     {format_number(summary['entropy'])}")
        print(f"    Advantage:   {format_number(summary['advantage_mean'])}")
        print(f"    Grad Norm:   {format_number(summary['gradient_norm'])}")

        print(f"\n  {Colors.BOLD}{Colors.YELLOW}🔧 TOOL CALLS{Colors.END}")
        acc_color = Colors.GREEN if summary['tool_accuracy'] > 0.5 else Colors.YELLOW
        print(f"    Accuracy (all):    {acc_color}{summary['tool_accuracy']*100:.1f}%{Colors.END}")
        print(f"    Accuracy (recent): {summary['tool_accuracy_recent']*100:.1f}%")
        print(f"    Total: {summary['correct_tool_calls']}/{summary['total_tool_calls']}")

        print(f"\n  {Colors.BOLD}{Colors.CYAN}✓ TASKS{Colors.END}")
        task_color = Colors.GREEN if summary['task_accuracy'] > 0.5 else Colors.YELLOW
        print(f"    Accuracy (all):    {task_color}{summary['task_accuracy']*100:.1f}%{Colors.END}")
        print(f"    Accuracy (recent): {summary['task_accuracy_recent']*100:.1f}%")
        print(f"    Solved: {summary['tasks_solved']}/{summary['tasks_attempted']}")

        print(f"\n  {Colors.BOLD}⚡ THROUGHPUT{Colors.END}")
        print(f"    Episodes: {summary['total_episodes']:,}")
        print(f"    Speed: {summary['throughput']:.2f} ep/s")
        print(f"    Queue: {TRAJECTORY_QUEUE.size()}")

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

        if self.metrics.best_examples:
            print_box(
                f"🏆 BEST (R={self.metrics.best_examples[0]['reward']:.2f})",
                [
                    f"Response: {truncate_text(self.metrics.best_examples[0]['response'], 60)}",
                ],
                width=70,
                color=Colors.GREEN
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
        """Main training loop with memory optimization"""
        print_box(
            "FULL ATROPOS RLVR TRAINER",
            [
                f"Model: {Colors.BOLD}{self.config.model_name}{Colors.END}",
                f"vLLM: {self.config.vllm_url}",
                f"API: {self.config.api_url}",
                "",
                f"Steps: {Colors.CYAN}{self.config.total_steps}{Colors.END}",
                f"Batch: {self.config.batch_size}",
                f"LR: {self.config.learning_rate}",
                "",
                f"Group size: {self.config.group_size}",
                f"Steps/rollout: {self.config.steps_per_rollout}",
                f"Max queue: {self.config.max_queue_size}",
                f"Entropy coef: {self.config.entropy_coef}",
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

        for step in progress:
            # Wait for batch (smaller batch for memory)
            batch = []
            wait_start = time.time()
            while len(batch) < self.config.batch_size:
                # Try to pull from queue
                pulled = TRAJECTORY_QUEUE.pull(self.config.batch_size - len(batch))
                batch.extend(pulled)

                if len(batch) < self.config.batch_size:
                    if time.time() - wait_start > 60:
                        print(Colors.warning(f"\n⚠ Waiting for rollouts... (queue: {TRAJECTORY_QUEUE.size()})"))
                        wait_start = time.time()
                    time.sleep(0.5)

            # Multiple gradient steps per batch (reuse rollouts)
            for _ in range(self.config.steps_per_rollout):
                step_metrics = self.train_on_batch(batch)

                # Gradient step
                if (step + 1) % self.config.gradient_accumulation == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    ).item()
                    self.metrics.gradient_norms.append(grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Memory cleanup every 10 steps
            if (step + 1) % 10 == 0:
                torch.cuda.empty_cache()

            # Update progress
            summary = self.metrics.get_summary()
            progress.set_postfix_str(
                f"R={Colors.reward(summary['reward_mean'])} | "
                f"Tool={summary['tool_accuracy']*100:.0f}% | "
                f"Task={summary['task_accuracy']*100:.0f}% | "
                f"Q={TRAJECTORY_QUEUE.size()}"
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


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Atropos RLVR Training")

    # Model
    parser.add_argument("--model", default="ai-sage/GigaChat3-10B-A1.8B-bf16")
    parser.add_argument("--output-dir", default="./outputs/rlvr-full")
    parser.add_argument("--dataset", default="nvidia/Nemotron-Agentic-v1")

    # vLLM
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1")
    parser.add_argument("--vllm-port", type=int, default=8001)

    # Trajectory API
    parser.add_argument("--api-port", type=int, default=8000)

    # Training
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4, help="Reduced for memory")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--steps-per-rollout", type=int, default=2,
                        help="Gradient steps per batch (reuse rollouts)")

    # Generation (with throttling)
    parser.add_argument("--group-size", type=int, default=4, help="Rollouts per prompt for GRPO-style training")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-queue-size", type=int, default=64,
                        help="Pause generation when queue exceeds this")
    parser.add_argument("--max-new-tokens", type=int, default=384, help="Reduced for memory")

    # Logging
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)

    # Mode
    parser.add_argument("--trainer-only", action="store_true",
                        help="Run only the trainer (assumes API and Env are running)")

    args = parser.parse_args()

    # Config
    config = RLVRConfig(
        model_name=args.model,
        output_dir=args.output_dir,
        vllm_url=args.vllm_url,
        api_url=f"http://localhost:{args.api_port}",
        dataset_name=args.dataset,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        group_size=args.group_size,
        temperature=args.temperature,
        log_steps=args.log_steps,
        save_steps=args.save_steps,
        steps_per_rollout=args.steps_per_rollout,
        max_queue_size=args.max_queue_size,
        max_new_tokens=args.max_new_tokens,
    )

    print_box(
        "ATROPOS RLVR PIPELINE",
        [
            f"Model: {args.model}",
            f"Dataset: {args.dataset}",
            "",
            f"Components:",
            f"  • Trajectory API: port {args.api_port}",
            f"  • vLLM Server: {args.vllm_url}",
            f"  • Environment: rollout generator",
            f"  • Trainer: policy gradient",
        ],
        width=70,
        color=Colors.HEADER
    )

    if args.trainer_only:
        # Run only trainer
        print("\n  Running trainer only mode...")
        trainer = FullRLVRTrainer(config)
        trainer.train()
    else:
        # Start all components

        # 1. Start Trajectory API
        print("\n  Starting components...")
        api_thread = threading.Thread(
            target=run_trajectory_api,
            args=(args.api_port,),
            daemon=True
        )
        api_thread.start()
        time.sleep(1)

        # 2. Start Environment (with throttling)
        env = RolloutEnvironment(
            vllm_url=args.vllm_url,
            api_url=config.api_url,
            dataset_name=args.dataset,
            model_name=args.model,
            max_samples=args.max_samples,
            group_size=config.group_size,
            max_new_tokens=config.max_new_tokens,
            temperature=args.temperature,
            max_queue_size=config.max_queue_size,
            generation_pause_time=config.generation_pause_time,
        )

        env_thread = threading.Thread(target=env.run, daemon=True)
        env_thread.start()
        print(f"  {Colors.success('✓')} Environment started")

        # Wait for some initial rollouts
        print("  Generating initial rollouts...")
        while TRAJECTORY_QUEUE.size() < config.batch_size * 2:
            time.sleep(1)
            print(f"    Queue size: {TRAJECTORY_QUEUE.size()}", end="\r")
        print(f"  {Colors.success('✓')} Initial rollouts ready ({TRAJECTORY_QUEUE.size()} in queue)")

        # 3. Start Trainer
        trainer = FullRLVRTrainer(config)
        trainer.train()

        # Cleanup
        env.stop()


if __name__ == "__main__":
    main()
