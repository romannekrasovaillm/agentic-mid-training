#!/usr/bin/env python3
"""
GigaChat RLVR (Reinforcement Learning with Verifiable Rewards) Training
Based on Atropos framework with Interleaved Reasoning

Key features:
- Tool calls inside <think> blocks
- Verifiable rewards based on boxed answers
- Multi-turn reasoning with tool execution
"""

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from atroposlib import BaseEnv, APIServerConfig, ManagedServer
    from atroposlib.envs import register_env
    ATROPOS_AVAILABLE = True
except ImportError:
    ATROPOS_AVAILABLE = False
    print("Warning: atroposlib not installed. Install with: pip install atroposlib")

try:
    from trl import GRPOConfig, GRPOTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Constants for interleaved thinking
MAX_REPLY_TOKENS = 2048
MAX_GEN_PER_TURN = 512
MAX_ROLLOUT_TURNS = 3
TOOL_USAGE_BONUS = 0.2
CORRECT_ANSWER_REWARD = 1.0
INCORRECT_ANSWER_REWARD = -1.0
INVALID_STRUCTURE_REWARD = -1.0


@dataclass
class RLVRConfig:
    """RLVR Training Configuration"""
    # Model
    model_name_or_path: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True

    # Training
    output_dir: str = "./outputs/gigachat-rlvr"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-7
    max_grad_norm: float = 1.0

    # RLVR specific
    num_generations: int = 4  # Number of rollouts per prompt
    max_new_tokens: int = MAX_REPLY_TOKENS
    temperature: float = 0.7
    top_p: float = 0.9

    # Rewards
    correct_reward: float = CORRECT_ANSWER_REWARD
    incorrect_reward: float = INCORRECT_ANSWER_REWARD
    tool_bonus: float = TOOL_USAGE_BONUS

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch.dtype"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


# Tool implementations for interleaved thinking
def execute_calculator(expression: str) -> dict:
    """Safe calculator execution"""
    try:
        # Restricted eval for math expressions
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "int": int, "float": float,
        }
        # Remove dangerous builtins
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"success": True, "result": str(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_python(code: str) -> dict:
    """Execute Python code in restricted environment"""
    try:
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Create restricted globals
        restricted_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range,
                "int": int, "float": float, "str": str,
                "list": list, "dict": dict, "set": set,
                "sum": sum, "min": min, "max": max,
                "abs": abs, "round": round, "sorted": sorted,
                "enumerate": enumerate, "zip": zip,
                "True": True, "False": False, "None": None,
            }
        }

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, restricted_globals)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        if errors:
            return {"success": False, "error": errors}
        return {"success": True, "result": output.strip() if output else "Executed successfully"}
    except Exception as e:
        return {"success": False, "error": str(e)}


AVAILABLE_TOOLS = {
    "calculator": execute_calculator,
    "python_interpreter": execute_python,
}


def extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from text"""
    tool_calls = []

    # Pattern: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    pattern = r'<tool_call>\s*(\{[^}]+\})\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        try:
            tool_call = json.loads(match)
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            continue

    # Also try simpler JSON format
    json_pattern = r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]*\})\}'
    json_matches = re.findall(json_pattern, text)

    for name, args in json_matches:
        try:
            tool_calls.append({
                "name": name,
                "arguments": json.loads(args)
            })
        except json.JSONDecodeError:
            tool_calls.append({
                "name": name,
                "arguments": {"expression": args.strip('"')}
            })

    return tool_calls


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format"""
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def execute_tools(tool_calls: list[dict]) -> list[dict]:
    """Execute tool calls and return results"""
    results = []
    for tool_call in tool_calls:
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        if name in AVAILABLE_TOOLS:
            if isinstance(args, dict):
                # Get the first argument value
                arg_value = next(iter(args.values()), "")
            else:
                arg_value = str(args)

            result = AVAILABLE_TOOLS[name](arg_value)
            results.append({
                "tool": name,
                "input": arg_value,
                "output": result
            })
        else:
            results.append({
                "tool": name,
                "input": args,
                "output": {"success": False, "error": f"Unknown tool: {name}"}
            })

    return results


def compute_reward(
    generated_text: str,
    expected_answer: Optional[str] = None,
    config: Optional[RLVRConfig] = None
) -> tuple[float, dict]:
    """
    Compute reward for generated text based on:
    1. Correctness of boxed answer
    2. Valid structure (think block, tool usage)
    3. Tool usage bonus
    """
    if config is None:
        config = RLVRConfig()

    reward = 0.0
    info = {
        "has_think_block": False,
        "has_boxed_answer": False,
        "tool_calls_count": 0,
        "answer_correct": False,
        "extracted_answer": None,
    }

    # Check for think block
    has_think_open = "<think>" in generated_text
    has_think_close = "</think>" in generated_text
    info["has_think_block"] = has_think_open and has_think_close

    # Extract and check answer
    extracted = extract_boxed_answer(generated_text)
    info["extracted_answer"] = extracted
    info["has_boxed_answer"] = extracted is not None

    # Count tool calls
    tool_calls = extract_tool_calls(generated_text)
    info["tool_calls_count"] = len(tool_calls)

    # Structure validation
    if not info["has_boxed_answer"]:
        reward = config.incorrect_reward
    elif expected_answer is not None:
        # Compare answers (normalize both)
        try:
            extracted_normalized = str(extracted).strip().lower()
            expected_normalized = str(expected_answer).strip().lower()

            # Try numeric comparison
            try:
                extracted_num = float(extracted_normalized.replace(",", ""))
                expected_num = float(expected_normalized.replace(",", ""))
                info["answer_correct"] = abs(extracted_num - expected_num) < 1e-6
            except ValueError:
                info["answer_correct"] = extracted_normalized == expected_normalized

            if info["answer_correct"]:
                reward = config.correct_reward
                # Tool usage bonus
                if info["tool_calls_count"] > 0:
                    reward += config.tool_bonus
            else:
                reward = config.incorrect_reward
        except Exception:
            reward = config.incorrect_reward
    else:
        # No expected answer - reward based on structure
        reward = 0.5 if info["has_think_block"] else 0.0
        if info["tool_calls_count"] > 0:
            reward += config.tool_bonus

    return reward, info


def format_tool_use_prompt(question: str, tools: list[str] = None) -> str:
    """Format prompt for tool use with interleaved thinking"""
    if tools is None:
        tools = list(AVAILABLE_TOOLS.keys())

    tools_description = """Available tools:
- calculator: Evaluate mathematical expressions. Usage: {"name": "calculator", "arguments": {"expression": "2+2"}}
- python_interpreter: Execute Python code. Usage: {"name": "python_interpreter", "arguments": {"code": "print(2+2)"}}"""

    prompt = f"""You are a helpful AI assistant that can use tools to solve problems.

{tools_description}

When solving problems:
1. Start with <think> to begin your reasoning
2. Use <tool_call>...</tool_call> to call tools inside your thinking
3. Tool results will appear in <tool_response>...</tool_response>
4. Continue reasoning with tool results
5. End with </think> and provide your final answer in \\boxed{{answer}}

Question: {question}

"""
    return prompt


class InterleavedThinkingEnv:
    """
    Environment for interleaved thinking with tool use
    Compatible with Atropos-style training
    """

    def __init__(self, config: RLVRConfig, tokenizer, model=None):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.max_turns = MAX_ROLLOUT_TURNS

    def generate_with_tools(self, prompt: str, max_turns: int = None) -> tuple[str, list]:
        """Generate response with multi-turn tool execution"""
        if max_turns is None:
            max_turns = self.max_turns

        if self.model is None:
            raise ValueError("Model not set for generation")

        full_response = ""
        tool_history = []

        current_prompt = prompt

        for turn in range(max_turns):
            # Generate
            inputs = self.tokenizer(current_prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_GEN_PER_TURN,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            full_response += generated

            # Check for completion
            if "</think>" in generated and "\\boxed{" in generated:
                break

            # Extract and execute tools
            tool_calls = extract_tool_calls(generated)
            if not tool_calls:
                # No tool calls and not complete - continue or break
                if turn < max_turns - 1:
                    continue
                break

            # Execute tools
            results = execute_tools(tool_calls)
            tool_history.extend(results)

            # Append tool results to prompt
            for result in results:
                tool_response = f"\n<tool_response>{json.dumps(result['output'], ensure_ascii=False)}</tool_response>\n"
                full_response += tool_response
                current_prompt = prompt + full_response

        return full_response, tool_history

    def compute_rewards(
        self,
        prompts: list[str],
        responses: list[str],
        expected_answers: list[Optional[str]] = None
    ) -> list[float]:
        """Compute rewards for batch of responses"""
        if expected_answers is None:
            expected_answers = [None] * len(responses)

        rewards = []
        for response, expected in zip(responses, expected_answers):
            reward, _ = compute_reward(response, expected, self.config)
            rewards.append(reward)

        return rewards


def prepare_rlvr_dataset(data_config: dict, tokenizer) -> list[dict]:
    """Prepare dataset for RLVR training"""
    logger.info("Loading dataset for RLVR...")

    dataset_name = data_config.get("dataset_name", "nvidia/Nemotron-Agentic-v1")
    max_samples = data_config.get("max_train_samples", 5000)

    # Load dataset
    from huggingface_hub import hf_hub_download

    split_name = data_config.get("dataset_config", "tool_calling")
    file_path = hf_hub_download(
        repo_id=dataset_name,
        filename=f"data/{split_name}.jsonl",
        repo_type="dataset",
    )

    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                example = json.loads(line.strip())
                if "messages" in example:
                    messages = example["messages"]

                    # Extract question from user message
                    question = None
                    expected_answer = None

                    for msg in messages:
                        if msg.get("role") == "user":
                            question = msg.get("content", "")
                        elif msg.get("role") == "assistant":
                            # Try to extract expected answer
                            content = msg.get("content", "")
                            extracted = extract_boxed_answer(content)
                            if extracted:
                                expected_answer = extracted

                    if question:
                        prompt = format_tool_use_prompt(question)
                        samples.append({
                            "prompt": prompt,
                            "question": question,
                            "expected_answer": expected_answer,
                        })
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(samples)} samples for RLVR")
    return samples


def setup_model_and_tokenizer(config: RLVRConfig):
    """Initialize model and tokenizer with LoRA"""
    logger.info(f"Loading model: {config.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = get_torch_dtype(config.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code,
        attn_implementation="sdpa",
        device_map="auto",
    )

    if config.use_lora:
        logger.info("Applying LoRA configuration...")

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def train_with_grpo_rlvr(config: RLVRConfig, data_config: dict):
    """
    Train using GRPO with RLVR rewards
    Falls back to custom implementation if TRL not available
    """
    model, tokenizer = setup_model_and_tokenizer(config)

    # Prepare dataset
    dataset = prepare_rlvr_dataset(data_config, tokenizer)

    # Create environment
    env = InterleavedThinkingEnv(config, tokenizer, model)

    if TRL_AVAILABLE:
        logger.info("Using TRL GRPO Trainer with RLVR rewards")

        # Convert to HF dataset format
        from datasets import Dataset

        hf_dataset = Dataset.from_list([
            {"prompt": s["prompt"], "expected_answer": s.get("expected_answer")}
            for s in dataset
        ])

        # Custom reward function for GRPO
        def reward_fn(completions, prompts=None, **kwargs):
            rewards = []
            for completion in completions:
                reward, _ = compute_reward(completion, config=config)
                rewards.append(reward)
            return rewards

        training_args = GRPOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            num_generations=config.num_generations,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
        )

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_fn,
        )

        logger.info("Starting GRPO training with RLVR rewards...")
        trainer.train()
        trainer.save_model()

    else:
        logger.info("TRL not available, using custom RLVR training loop")
        custom_rlvr_training(model, tokenizer, env, dataset, config)


def custom_rlvr_training(model, tokenizer, env, dataset, config: RLVRConfig):
    """Custom RLVR training loop when TRL is not available"""
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    logger.info(f"Starting custom RLVR training with {len(dataset)} samples")

    for epoch in range(config.num_train_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_train_epochs}")

        total_reward = 0
        num_batches = 0

        for i in range(0, len(dataset), config.per_device_train_batch_size):
            batch = dataset[i:i + config.per_device_train_batch_size]

            batch_rewards = []
            batch_log_probs = []

            for sample in batch:
                prompt = sample["prompt"]
                expected = sample.get("expected_answer")

                # Generate with tools
                response, tool_history = env.generate_with_tools(prompt)

                # Compute reward
                reward, info = compute_reward(response, expected, config)
                batch_rewards.append(reward)

                # Get log probs for policy gradient
                inputs = tokenizer(prompt + response, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                    # Simplified: use mean log prob
                    mean_log_prob = log_probs.mean()
                    batch_log_probs.append(mean_log_prob)

            # Policy gradient update
            if batch_log_probs:
                rewards_tensor = torch.tensor(batch_rewards, device=model.device)
                rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

                # Simplified policy gradient loss
                loss = -torch.stack(batch_log_probs).mean() * rewards_tensor.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                total_reward += sum(batch_rewards)
                num_batches += 1

                if num_batches % 10 == 0:
                    avg_reward = total_reward / (num_batches * config.per_device_train_batch_size)
                    logger.info(f"Batch {num_batches}, Avg Reward: {avg_reward:.4f}")

        # Save checkpoint
        save_path = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    # Final save
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Training complete! Model saved to {config.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="GigaChat RLVR Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # Load config
    config_dict = load_config(args.config)

    # Set seed
    seed = config_dict.get("training", {}).get("seed", 42)
    set_seed(seed)

    # Create RLVR config
    training_config = config_dict.get("training", {})
    lora_config = config_dict.get("lora", {})

    rlvr_config = RLVRConfig(
        model_name_or_path=config_dict["model"]["name"],
        torch_dtype=config_dict["model"].get("dtype", "bfloat16"),
        trust_remote_code=config_dict["model"].get("trust_remote_code", True),
        output_dir=training_config.get("output_dir", "./outputs/gigachat-rlvr"),
        num_train_epochs=training_config.get("num_train_epochs", 1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 5e-7),
        num_generations=training_config.get("num_generations", 4),
        max_new_tokens=training_config.get("max_new_tokens", MAX_REPLY_TOKENS),
        temperature=training_config.get("temperature", 0.7),
        use_lora=lora_config.get("enabled", True),
        lora_r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        lora_target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]),
    )

    # Get reward config
    rewards_config = config_dict.get("rewards", {})
    rlvr_config.correct_reward = rewards_config.get("correct_answer", CORRECT_ANSWER_REWARD)
    rlvr_config.incorrect_reward = rewards_config.get("incorrect_answer", INCORRECT_ANSWER_REWARD)
    rlvr_config.tool_bonus = rewards_config.get("tool_usage_bonus", TOOL_USAGE_BONUS)

    # Train
    data_config = config_dict.get("data", {})
    train_with_grpo_rlvr(rlvr_config, data_config)


if __name__ == "__main__":
    main()
