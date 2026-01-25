#!/usr/bin/env python3
"""
GigaChat Agentic Tool Calling Environment for Atropos RLVR

Based on NousResearch/atropos framework.
Uses nvidia/Nemotron-Agentic-v1 dataset for tool-calling training.

Usage:
    # Start trajectory API (Terminal 1)
    run-api

    # Start environment server (Terminal 2)
    python gigachat_tool_calling_env.py serve

    # Or generate data offline
    python gigachat_tool_calling_env.py process --env.data_path_to_save_groups output.jsonl
"""

import json
import os
import re
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from atroposlib.envs.base import BaseEnv, BaseEnvConfig
from atroposlib.type_definitions import (
    GameHistory,
    Item,
    Message,
    Reward,
)


class GigaChatToolCallingConfig(BaseEnvConfig):
    """Configuration for GigaChat Tool Calling Environment"""

    # Dataset settings
    dataset_name: str = "nvidia/Nemotron-Agentic-v1"
    dataset_split: str = "train"
    max_samples: int = 10000

    # Reward settings
    correct_tool_reward: float = 1.0
    incorrect_tool_reward: float = -1.0
    partial_reward_scale: float = 0.5
    length_penalty_threshold: float = 0.5

    # Tool call parsing
    tool_call_start_tag: str = "<tool_call>"
    tool_call_end_tag: str = "</tool_call>"
    think_start_tag: str = "<think>"
    think_end_tag: str = "</think>"


class GigaChatToolCallingEnv(BaseEnv):
    """
    Atropos Environment for training GigaChat on agentic tool-calling tasks.

    Reward function:
    - +1.0 for correct tool calls (name + arguments match)
    - Partial reward for partially correct calls
    - -1.0 for incorrect/missing tool calls
    - Length penalty for overly verbose responses
    """

    def __init__(self, config: GigaChatToolCallingConfig, server_configs: List[Any], slurm: bool = False):
        super().__init__(config, server_configs, slurm)
        self.config: GigaChatToolCallingConfig = config

        # Metrics tracking
        self.percent_correct_buffer: List[float] = []
        self.eval_metrics: Dict[str, List[float]] = defaultdict(list)
        self.completion_lengths: List[int] = []
        self.rollouts_for_wandb: List[Dict] = []

        # Dataset
        self.train_data: List[Dict] = []
        self.eval_data: List[Dict] = []
        self._load_dataset()

    @classmethod
    def config_init(cls) -> Tuple[GigaChatToolCallingConfig, List[Dict]]:
        """Initialize default configuration"""
        env_config = GigaChatToolCallingConfig(
            tokenizer_name="ai-sage/GigaChat3-10B-A1.8B-bf16",
            group_size=8,  # Rollouts per item
            use_wandb=True,
            max_num_workers=16,
            rollout_server_url="http://localhost:8000",
            total_steps=2000,
            batch_size=512,
            steps_per_eval=50,
            max_token_length=1536,  # GigaChat3 context limit
            wandb_name="gigachat-tool-calling",
        )

        # Server config for vLLM
        server_configs = [
            {
                "model_name": "ai-sage/GigaChat3-10B-A1.8B-bf16",
                "base_url": "http://localhost:8001/v1",
                "api_key": "EMPTY",
                "num_requests_per_second": 10,
                "temperature": 0.8,
                "max_tokens": 1024,
            }
        ]

        return env_config, server_configs

    def _load_dataset(self):
        """Load Nemotron-Agentic dataset and split into train/eval"""
        print(f"Loading dataset: {self.config.dataset_name}")

        try:
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split,
                trust_remote_code=True
            )

            # Convert to list and shuffle
            all_samples = list(dataset)
            random.seed(42)
            random.shuffle(all_samples)

            # Limit samples
            if self.config.max_samples > 0:
                all_samples = all_samples[:self.config.max_samples]

            # Split 95/5 for train/eval
            split_idx = int(len(all_samples) * 0.95)
            self.train_data = all_samples[:split_idx]
            self.eval_data = all_samples[split_idx:]

            print(f"Loaded {len(self.train_data)} train samples, {len(self.eval_data)} eval samples")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.train_data = []
            self.eval_data = []

    def _parse_messages(self, sample: Dict) -> List[Dict]:
        """Parse messages from dataset sample"""
        if "messages" in sample:
            messages = sample["messages"]
            if isinstance(messages, str):
                messages = json.loads(messages)
            return messages
        elif "messages_json" in sample:
            return json.loads(sample["messages_json"])
        return []

    def _extract_tool_calls(self, text: str) -> List[Dict]:
        """Extract tool calls from model response"""
        tool_calls = []

        # Pattern for <tool_call>...</tool_call>
        pattern = rf'{re.escape(self.config.tool_call_start_tag)}(.*?){re.escape(self.config.tool_call_end_tag)}'
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            try:
                # Try to parse as JSON
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                # Try to extract function name and arguments
                try:
                    # Handle format: {"name": "func", "arguments": {...}}
                    if "name" in match and "arguments" in match:
                        tool_call = json.loads(match.strip())
                        tool_calls.append(tool_call)
                except:
                    pass

        # Also check for function_call format
        if not tool_calls:
            func_pattern = r'"function_call"\s*:\s*\{([^}]+)\}'
            func_matches = re.findall(func_pattern, text, re.DOTALL)
            for match in func_matches:
                try:
                    tool_call = json.loads("{" + match + "}")
                    tool_calls.append(tool_call)
                except:
                    pass

        return tool_calls

    def _get_expected_tool_calls(self, messages: List[Dict]) -> List[Dict]:
        """Extract expected tool calls from assistant messages in dataset"""
        expected_calls = []

        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")

                # Check for tool_calls field
                if "tool_calls" in msg:
                    for tc in msg["tool_calls"]:
                        if "function" in tc:
                            expected_calls.append({
                                "name": tc["function"].get("name", ""),
                                "arguments": tc["function"].get("arguments", {})
                            })

                # Also extract from content
                content_calls = self._extract_tool_calls(content)
                expected_calls.extend(content_calls)

        return expected_calls

    def _compare_tool_calls(self, expected: List[Dict], actual: List[Dict]) -> Tuple[float, Dict]:
        """
        Compare expected and actual tool calls.
        Returns (score, details)
        """
        if not expected:
            # No tool calls expected
            if not actual:
                return 1.0, {"match": "no_tools_expected"}
            else:
                return 0.5, {"match": "unexpected_tools"}

        if not actual:
            return 0.0, {"match": "missing_tools"}

        # Count matches
        matched = 0
        details = {"expected": len(expected), "actual": len(actual), "matched": 0}

        for exp_call in expected:
            exp_name = exp_call.get("name", "")
            exp_args = exp_call.get("arguments", {})

            for act_call in actual:
                act_name = act_call.get("name", "")
                act_args = act_call.get("arguments", {})

                # Check name match
                if exp_name == act_name:
                    # Check arguments match
                    if self._compare_arguments(exp_args, act_args):
                        matched += 1
                        break
                    else:
                        # Partial match for correct function name
                        matched += 0.5
                        break

        details["matched"] = matched
        score = matched / len(expected) if expected else 1.0

        return score, details

    def _compare_arguments(self, expected: Any, actual: Any) -> bool:
        """Recursively compare arguments"""
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key, value in expected.items():
                if key not in actual:
                    return False
                if not self._compare_arguments(value, actual[key]):
                    return False
            return True
        elif isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False
            return all(self._compare_arguments(e, a) for e, a in zip(expected, actual))
        else:
            return str(expected) == str(actual)

    async def setup(self):
        """Setup environment (called once at start)"""
        print("Setting up GigaChat Tool Calling Environment...")
        if not self.train_data:
            self._load_dataset()

    async def get_next_item(self) -> Item:
        """Get next training item"""
        if not self.train_data:
            raise RuntimeError("No training data loaded")

        # Random sample
        sample = random.choice(self.train_data)
        messages = self._parse_messages(sample)

        # Build prompt from messages (exclude last assistant response)
        prompt_messages = []
        for msg in messages:
            if msg.get("role") == "assistant":
                # Check if this is the response we want to generate
                if self._get_expected_tool_calls([msg]):
                    break
            prompt_messages.append(msg)

        # Format as chat
        prompt = self._format_prompt(prompt_messages)

        return Item(
            messages=[Message(role="user", content=prompt)],
            metadata={
                "expected_tool_calls": self._get_expected_tool_calls(messages),
                "original_messages": messages,
            }
        )

    def _format_prompt(self, messages: List[Dict]) -> str:
        """Format messages as prompt"""
        formatted = []

        # Add system prompt for tool calling
        system_prompt = """You are a helpful AI assistant with access to tools. When you need to use a tool, format your response as:

<think>
Your reasoning about which tool to use and why
</think>

<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

Always think step by step before making tool calls."""

        formatted.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted.append(f"System: {content}\n")
            elif role == "user":
                formatted.append(f"User: {content}\n")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}\n")
            elif role == "tool":
                tool_name = msg.get("name", "tool")
                formatted.append(f"Tool ({tool_name}): {content}\n")

        formatted.append("Assistant: ")

        return "".join(formatted)

    async def collect_trajectories(self, item: Item) -> GameHistory:
        """Collect model responses for item"""
        responses = await self.server.generate(
            messages=item.messages,
            n=self.config.group_size,
            temperature=0.8,
            max_tokens=1024,
        )

        return GameHistory(
            item=item,
            responses=responses,
        )

    async def score(self, game_history: GameHistory) -> List[Reward]:
        """Score model responses"""
        rewards = []
        expected_calls = game_history.item.metadata.get("expected_tool_calls", [])

        for response in game_history.responses:
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Extract tool calls from response
            actual_calls = self._extract_tool_calls(response_text)

            # Compare
            score, details = self._compare_tool_calls(expected_calls, actual_calls)

            # Apply reward scaling
            if score >= 1.0:
                reward = self.config.correct_tool_reward
            elif score > 0:
                reward = score * self.config.partial_reward_scale
            else:
                reward = self.config.incorrect_tool_reward

            # Length penalty for correct responses
            if reward > 0:
                response_len = len(response_text)
                max_len = self.config.max_token_length
                if response_len > max_len * self.config.length_penalty_threshold:
                    length_ratio = (response_len - max_len * self.config.length_penalty_threshold) / (max_len * self.config.length_penalty_threshold)
                    length_penalty = min(length_ratio * 0.5, 0.5)
                    reward -= length_penalty

            rewards.append(Reward(
                value=reward,
                metadata={
                    "score": score,
                    "details": details,
                    "response_length": len(response_text),
                }
            ))

            # Track metrics
            self.percent_correct_buffer.append(1.0 if score >= 1.0 else 0.0)
            self.completion_lengths.append(len(response_text))

        return rewards

    async def evaluate(self) -> Dict[str, float]:
        """Run evaluation on held-out data"""
        if not self.eval_data:
            return {}

        correct = 0
        total = 0

        for sample in self.eval_data[:100]:  # Limit eval samples
            messages = self._parse_messages(sample)
            expected_calls = self._get_expected_tool_calls(messages)

            # Build prompt
            prompt_messages = []
            for msg in messages:
                if msg.get("role") == "assistant" and self._get_expected_tool_calls([msg]):
                    break
                prompt_messages.append(msg)

            prompt = self._format_prompt(prompt_messages)

            # Generate response
            responses = await self.server.generate(
                messages=[Message(role="user", content=prompt)],
                n=1,
                temperature=0.0,  # Greedy for eval
                max_tokens=1024,
            )

            if responses:
                response_text = responses[0].content if hasattr(responses[0], 'content') else str(responses[0])
                actual_calls = self._extract_tool_calls(response_text)
                score, _ = self._compare_tool_calls(expected_calls, actual_calls)

                if score >= 1.0:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            "eval_accuracy": accuracy,
            "eval_correct": correct,
            "eval_total": total,
            "avg_completion_length": sum(self.completion_lengths) / len(self.completion_lengths) if self.completion_lengths else 0,
            "train_accuracy": sum(self.percent_correct_buffer) / len(self.percent_correct_buffer) if self.percent_correct_buffer else 0,
        }

        # Reset buffers
        self.percent_correct_buffer = []
        self.completion_lengths = []

        return metrics

    def get_wandb_log_data(self) -> Dict[str, Any]:
        """Get data for W&B logging"""
        return {
            "train_accuracy": sum(self.percent_correct_buffer) / len(self.percent_correct_buffer) if self.percent_correct_buffer else 0,
            "avg_completion_length": sum(self.completion_lengths) / len(self.completion_lengths) if self.completion_lengths else 0,
            "num_samples": len(self.percent_correct_buffer),
        }


if __name__ == "__main__":
    import asyncio
    from atroposlib.cli import run_env

    # Run environment with CLI
    run_env(GigaChatToolCallingEnv)
