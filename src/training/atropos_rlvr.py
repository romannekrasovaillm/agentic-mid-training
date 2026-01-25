#!/usr/bin/env python3
"""
GigaChat RLVR Training with Full Atropos Integration
=====================================================

Полная реализация RLVR на базе Atropos с:
- vLLM сервером для быстрого инференса
- Async rollouts для параллельной генерации
- Interleaved thinking с tool execution
- Verifiable rewards

Usage:
    # Запуск vLLM сервера (в отдельном терминале)
    python -m vllm.entrypoints.openai.api_server \
        --model ai-sage/GigaChat3-10B-A1.8B-bf16 \
        --port 8000 --dtype bfloat16

    # Запуск обучения
    python atropos_rlvr.py --config configs/post_training_rlvr.yaml
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

# Настройка максимального логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("atropos_rlvr")

# Отключаем лишние логи от библиотек
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_REPLY_TOKENS = 2048
MAX_GEN_PER_TURN = 512
MAX_ROLLOUT_TURNS = 3
TOOL_USAGE_BONUS = 0.2
CORRECT_ANSWER_REWARD = 1.0
INCORRECT_ANSWER_REWARD = -1.0
INVALID_STRUCTURE_REWARD = -1.0


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AtroposConfig:
    """Конфигурация Atropos RLVR"""
    # Model
    model_name: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"

    # vLLM Server
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"

    # Training
    output_dir: str = "./outputs/atropos-rlvr"
    num_episodes: int = 100
    batch_size: int = 8  # Увеличен для лучшей утилизации
    num_rollouts_per_prompt: int = 8  # Увеличено для параллельности
    learning_rate: float = 5e-7
    max_grad_norm: float = 1.0

    # Async Generation (для максимальной утилизации KV-кэша)
    max_concurrent_requests: int = 32  # Параллельные запросы к vLLM
    use_beam_search: bool = False
    best_of: int = 1  # Можно увеличить для better sampling

    # Generation
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = MAX_REPLY_TOKENS

    # Rewards
    correct_reward: float = CORRECT_ANSWER_REWARD
    incorrect_reward: float = INCORRECT_ANSWER_REWARD
    tool_bonus: float = TOOL_USAGE_BONUS

    # LoRA
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Data
    dataset_name: str = "nvidia/Nemotron-Agentic-v1"
    dataset_split: str = "tool_calling"
    max_samples: int = 1000

    # Logging
    log_every_n_steps: int = 1
    save_every_n_steps: int = 50
    verbose: bool = True


# ============================================================================
# TOOLS
# ============================================================================

def execute_calculator(expression: str) -> Dict[str, Any]:
    """Безопасное выполнение математических выражений"""
    logger.debug(f"Calculator: evaluating '{expression}'")
    try:
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "int": int, "float": float,
            "len": len,
        }
        import math
        allowed_names.update({
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "exp": math.exp,
            "pi": math.pi, "e": math.e,
        })
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        logger.debug(f"Calculator result: {result}")
        return {"success": True, "result": str(result)}
    except Exception as e:
        logger.warning(f"Calculator error: {e}")
        return {"success": False, "error": str(e)}


def execute_python(code: str) -> Dict[str, Any]:
    """Выполнение Python кода в ограниченном окружении"""
    logger.debug(f"Python interpreter: executing code ({len(code)} chars)")
    try:
        import io
        from contextlib import redirect_stdout, redirect_stderr

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        restricted_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range,
                "int": int, "float": float, "str": str, "bool": bool,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "sum": sum, "min": min, "max": max, "abs": abs,
                "round": round, "sorted": sorted, "reversed": reversed,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "True": True, "False": False, "None": None,
                "isinstance": isinstance, "type": type,
            }
        }

        local_vars = {}

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, restricted_globals, local_vars)

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        if errors:
            logger.warning(f"Python stderr: {errors}")
            return {"success": False, "error": errors}

        result = output.strip() if output else "Executed successfully"
        logger.debug(f"Python result: {result[:100]}...")
        return {"success": True, "result": result}
    except Exception as e:
        logger.warning(f"Python error: {e}")
        return {"success": False, "error": str(e)}


TOOLS = {
    "calculator": {
        "function": execute_calculator,
        "description": "Evaluate mathematical expressions",
        "usage": '{"name": "calculator", "arguments": {"expression": "2+2*3"}}'
    },
    "python_interpreter": {
        "function": execute_python,
        "description": "Execute Python code",
        "usage": '{"name": "python_interpreter", "arguments": {"code": "print(sum(range(10)))"}}'
    }
}


# ============================================================================
# PARSING & REWARDS
# ============================================================================

def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Извлечение tool calls из текста"""
    tool_calls = []

    # Pattern 1: <tool_call>JSON</tool_call>
    pattern1 = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern1, text, re.DOTALL)

    for match in matches:
        try:
            parsed = json.loads(match)
            tool_calls.append(parsed)
            logger.debug(f"Extracted tool call: {parsed.get('name', 'unknown')}")
        except json.JSONDecodeError:
            continue

    # Pattern 2: Direct JSON in text
    pattern2 = r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]*\})\}'
    json_matches = re.findall(pattern2, text)

    for name, args_str in json_matches:
        try:
            args = json.loads(args_str)
            tool_calls.append({"name": name, "arguments": args})
            logger.debug(f"Extracted inline tool call: {name}")
        except json.JSONDecodeError:
            tool_calls.append({"name": name, "arguments": {"input": args_str}})

    return tool_calls


def extract_boxed_answer(text: str) -> Optional[str]:
    """Извлечение ответа из \\boxed{}"""
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\\boxed\s*\{([^}]+)\}',
        r'boxed\{([^}]+)\}',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip()
            logger.debug(f"Extracted boxed answer: {answer}")
            return answer

    return None


def execute_tool_calls(tool_calls: List[Dict]) -> List[Dict]:
    """Выполнение списка tool calls"""
    results = []

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        logger.info(f"Executing tool: {name}")

        if name in TOOLS:
            if isinstance(args, dict):
                # Get first argument value
                arg_value = next(iter(args.values()), "")
            else:
                arg_value = str(args)

            result = TOOLS[name]["function"](arg_value)
            results.append({
                "tool": name,
                "input": arg_value,
                "output": result
            })
        else:
            logger.warning(f"Unknown tool: {name}")
            results.append({
                "tool": name,
                "input": args,
                "output": {"success": False, "error": f"Unknown tool: {name}"}
            })

    return results


def compute_reward(
    response: str,
    expected_answer: Optional[str] = None,
    config: Optional[AtroposConfig] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    Вычисление награды за ответ

    Returns:
        Tuple[float, Dict]: (reward, info_dict)
    """
    if config is None:
        config = AtroposConfig()

    info = {
        "has_think_block": False,
        "has_boxed_answer": False,
        "tool_calls_count": 0,
        "answer_correct": None,
        "extracted_answer": None,
        "reward_breakdown": {},
    }

    # Check structure
    info["has_think_block"] = "<think>" in response and "</think>" in response

    # Extract answer
    extracted = extract_boxed_answer(response)
    info["extracted_answer"] = extracted
    info["has_boxed_answer"] = extracted is not None

    # Count tools
    tool_calls = extract_tool_calls(response)
    info["tool_calls_count"] = len(tool_calls)

    # Calculate reward
    reward = 0.0
    breakdown = {}

    # No boxed answer = penalty
    if not info["has_boxed_answer"]:
        reward = config.incorrect_reward
        breakdown["no_answer"] = config.incorrect_reward
        logger.debug(f"No boxed answer found, reward={reward}")

    # Check correctness if expected answer provided
    elif expected_answer is not None:
        try:
            ext_norm = str(extracted).strip().lower().replace(",", "")
            exp_norm = str(expected_answer).strip().lower().replace(",", "")

            # Try numeric comparison
            try:
                ext_num = float(ext_norm)
                exp_num = float(exp_norm)
                info["answer_correct"] = abs(ext_num - exp_num) < 1e-6
            except ValueError:
                info["answer_correct"] = ext_norm == exp_norm

            if info["answer_correct"]:
                reward = config.correct_reward
                breakdown["correct"] = config.correct_reward
                logger.info(f"✓ Correct answer: {extracted}")
            else:
                reward = config.incorrect_reward
                breakdown["incorrect"] = config.incorrect_reward
                logger.info(f"✗ Wrong answer: {extracted} (expected: {expected_answer})")

        except Exception as e:
            logger.warning(f"Error comparing answers: {e}")
            reward = config.incorrect_reward
            breakdown["error"] = config.incorrect_reward
    else:
        # No expected answer - reward based on structure
        reward = 0.5 if info["has_think_block"] else 0.0
        breakdown["structure"] = reward

    # Tool usage bonus
    if info["tool_calls_count"] > 0 and reward > 0:
        tool_bonus = config.tool_bonus * min(info["tool_calls_count"], 3)
        reward += tool_bonus
        breakdown["tool_bonus"] = tool_bonus
        logger.debug(f"Tool bonus: +{tool_bonus} ({info['tool_calls_count']} tools)")

    info["reward_breakdown"] = breakdown
    logger.info(f"Final reward: {reward:.3f} | Breakdown: {breakdown}")

    return reward, info


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def format_system_prompt() -> str:
    """Системный промпт для агента"""
    tools_desc = "\n".join([
        f"- {name}: {info['description']}. Usage: {info['usage']}"
        for name, info in TOOLS.items()
    ])

    return f"""You are a helpful AI assistant that can use tools to solve problems.

Available tools:
{tools_desc}

When solving problems:
1. Start with <think> to begin your reasoning
2. Inside <think>, you can use tools by writing <tool_call>JSON</tool_call>
3. Tool results will appear in <tool_response>JSON</tool_response>
4. You can make multiple tool calls within the same <think> block
5. End your reasoning with </think>
6. Provide your final answer in \\boxed{{answer}}

Example:
<think>
I need to calculate 15 * 23.
<tool_call>{{"name": "calculator", "arguments": {{"expression": "15 * 23"}}}}</tool_call>
<tool_response>{{"success": true, "result": "345"}}</tool_response>
The result is 345.
</think>
\\boxed{{345}}"""


def format_prompt(question: str) -> str:
    """Форматирование промпта с вопросом"""
    return f"{format_system_prompt()}\n\nQuestion: {question}\n\n"


# ============================================================================
# VLLM CLIENT
# ============================================================================

class VLLMClient:
    """Клиент для vLLM OpenAI-compatible API"""

    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._client = None
        logger.info(f"VLLMClient initialized: {base_url}, model={model_name}")

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key="dummy"  # vLLM doesn't need real key
                )
                logger.info("OpenAI client created successfully")
            except ImportError:
                logger.error("openai package not installed!")
                raise
        return self._client

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Генерация ответов через vLLM"""
        logger.debug(f"Generating {n} responses, max_tokens={max_tokens}")

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop or ["</think>\\boxed"],
            )

            results = [choice.text for choice in response.choices]
            logger.debug(f"Generated {len(results)} responses")
            return results

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return [""] * n

    async def generate_with_tools(
        self,
        prompt: str,
        max_turns: int = MAX_ROLLOUT_TURNS,
        max_tokens_per_turn: int = MAX_GEN_PER_TURN,
        temperature: float = 0.7,
    ) -> Tuple[str, List[Dict]]:
        """
        Генерация с multi-turn tool execution

        Returns:
            Tuple[str, List[Dict]]: (full_response, tool_history)
        """
        logger.info(f"Starting multi-turn generation (max {max_turns} turns)")

        full_response = ""
        tool_history = []
        current_prompt = prompt

        for turn in range(max_turns):
            logger.debug(f"Turn {turn + 1}/{max_turns}")

            # Generate
            responses = await self.generate(
                current_prompt,
                max_tokens=max_tokens_per_turn,
                temperature=temperature,
                n=1,
            )

            if not responses or not responses[0]:
                logger.warning(f"Empty response at turn {turn + 1}")
                break

            generated = responses[0]
            full_response += generated

            # Check for completion
            if "</think>" in generated and "\\boxed{" in generated:
                logger.info(f"Completed at turn {turn + 1}")
                break

            # Extract and execute tools
            tool_calls = extract_tool_calls(generated)

            if not tool_calls:
                logger.debug("No tool calls found, continuing...")
                if turn < max_turns - 1:
                    continue
                break

            # Execute tools
            results = execute_tool_calls(tool_calls)
            tool_history.extend(results)

            # Append tool results
            for result in results:
                tool_response = f"\n<tool_response>{json.dumps(result['output'], ensure_ascii=False)}</tool_response>\n"
                full_response += tool_response

            current_prompt = prompt + full_response

        logger.info(f"Generation complete: {len(full_response)} chars, {len(tool_history)} tool calls")
        return full_response, tool_history


# ============================================================================
# ATROPOS ENVIRONMENT
# ============================================================================

class InterleavedThinkingEnv:
    """
    Atropos-style environment for interleaved thinking with tool use
    """

    def __init__(self, config: AtroposConfig):
        self.config = config
        self.vllm_client = VLLMClient(
            config.vllm_base_url,
            config.vllm_model_name
        )
        self.episode_count = 0
        self.total_reward = 0.0
        logger.info("InterleavedThinkingEnv initialized")

    async def collect_rollouts(
        self,
        prompts: List[str],
        expected_answers: Optional[List[str]] = None,
        num_rollouts: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Collect rollouts for a batch of prompts with high parallelism

        Returns:
            List of rollout dictionaries with response, reward, info
        """
        total_requests = len(prompts) * num_rollouts
        logger.info(f"Collecting rollouts: {len(prompts)} prompts × {num_rollouts} = {total_requests} total requests")
        logger.info(f"Max concurrent: {self.config.max_concurrent_requests}")

        if expected_answers is None:
            expected_answers = [None] * len(prompts)

        # Создаём все задачи сразу для максимальной параллельности
        all_tasks = []
        task_metadata = []  # (prompt_idx, rollout_idx, expected_answer)

        for i, (prompt, expected) in enumerate(zip(prompts, expected_answers)):
            for j in range(num_rollouts):
                task = self.vllm_client.generate_with_tools(
                    prompt,
                    max_turns=MAX_ROLLOUT_TURNS,
                    temperature=self.config.temperature,
                )
                all_tasks.append(task)
                task_metadata.append((i, j, prompt, expected))

        # Запускаем с ограничением параллельности через семафор
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def limited_task(task, meta):
            async with semaphore:
                result = await task
                return result, meta

        # Собираем все результаты параллельно
        logger.info(f"Launching {len(all_tasks)} async generation tasks...")
        wrapped_tasks = [limited_task(task, meta) for task, meta in zip(all_tasks, task_metadata)]
        results = await asyncio.gather(*wrapped_tasks)

        # Обрабатываем результаты
        all_rollouts = []
        for (response, tool_history), (prompt_idx, rollout_idx, prompt, expected) in results:
            reward, info = compute_reward(response, expected, self.config)

            rollout = {
                "prompt": prompt,
                "response": response,
                "expected_answer": expected,
                "tool_history": tool_history,
                "reward": reward,
                "info": info,
                "prompt_idx": prompt_idx,
                "rollout_idx": rollout_idx,
            }
            all_rollouts.append(rollout)

            logger.debug(
                f"Rollout P{prompt_idx}R{rollout_idx}: reward={reward:.3f}, "
                f"tools={info['tool_calls_count']}, "
                f"correct={info.get('answer_correct')}"
                )

        avg_reward = sum(r["reward"] for r in all_rollouts) / len(all_rollouts)
        logger.info(f"Rollouts complete: avg_reward={avg_reward:.3f}")

        return all_rollouts

    def compute_advantages(
        self,
        rollouts: List[Dict],
        baseline: str = "mean"
    ) -> List[float]:
        """Compute advantages for policy gradient"""
        rewards = [r["reward"] for r in rollouts]

        if baseline == "mean":
            mean_reward = sum(rewards) / len(rewards)
            std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
            std_reward = max(std_reward, 1e-8)
            advantages = [(r - mean_reward) / std_reward for r in rewards]
        else:
            advantages = rewards

        logger.debug(f"Advantages: min={min(advantages):.3f}, max={max(advantages):.3f}")
        return advantages


# ============================================================================
# TRAINER
# ============================================================================

class AtroposTrainer:
    """
    Full Atropos RLVR Trainer with LoRA
    """

    def __init__(self, config: AtroposConfig):
        self.config = config
        self.env = InterleavedThinkingEnv(config)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.step = 0

        # Statistics
        self.stats = {
            "rewards": [],
            "losses": [],
            "tool_usage": [],
            "correct_rate": [],
        }

        logger.info("AtroposTrainer initialized")

    def setup_model(self):
        """Load and setup model with LoRA"""
        logger.info("=" * 60)
        logger.info("LOADING MODEL")
        logger.info("=" * 60)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        # Tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            device_map={"": 0},
        )

        # LoRA
        if self.config.use_lora:
            logger.info("Applying LoRA...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Gradient checkpointing
        logger.info("Enabling gradient checkpointing...")
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        # Optimizer
        from torch.optim import AdamW
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        logger.info("Model setup complete")

    def policy_gradient_step(
        self,
        rollouts: List[Dict],
        advantages: List[float],
    ) -> float:
        """Perform a policy gradient update"""
        self.model.train()

        total_loss = 0.0
        num_samples = 0

        for rollout, advantage in zip(rollouts, advantages):
            if advantage == 0:
                continue

            # Tokenize
            full_text = rollout["prompt"] + rollout["response"]
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Forward pass
            outputs = self.model(**inputs, labels=inputs["input_ids"])

            # Policy gradient loss: -advantage * log_prob
            loss = -advantage * outputs.loss

            # Backward
            loss.backward()
            total_loss += loss.item()
            num_samples += 1

        if num_samples > 0:
            # Gradient clipping and step
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(num_samples, 1)
        logger.debug(f"Policy gradient step: loss={avg_loss:.4f}")
        return avg_loss

    def load_dataset(self) -> List[Dict]:
        """Load and prepare dataset"""
        logger.info(f"Loading dataset: {self.config.dataset_name}")

        from huggingface_hub import hf_hub_download

        file_path = hf_hub_download(
            repo_id=self.config.dataset_name,
            filename=f"data/{self.config.dataset_split}.jsonl",
            repo_type="dataset",
        )

        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.config.max_samples:
                    break
                try:
                    example = json.loads(line.strip())
                    if "messages" in example:
                        messages = example["messages"]

                        # Extract question
                        question = None
                        for msg in messages:
                            if msg.get("role") == "user":
                                question = msg.get("content", "")
                                break

                        if question:
                            samples.append({
                                "question": question,
                                "prompt": format_prompt(question),
                                "expected_answer": None,
                            })
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(samples)} samples")
        return samples

    async def train(self):
        """Main training loop"""
        logger.info("=" * 60)
        logger.info("STARTING ATROPOS RLVR TRAINING")
        logger.info("=" * 60)

        # Setup
        self.setup_model()
        samples = self.load_dataset()

        # Training loop
        num_batches = len(samples) // self.config.batch_size

        for episode in range(self.config.num_episodes):
            logger.info(f"\n{'='*40}")
            logger.info(f"EPISODE {episode + 1}/{self.config.num_episodes}")
            logger.info(f"{'='*40}")

            episode_rewards = []
            episode_losses = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = start_idx + self.config.batch_size
                batch = samples[start_idx:end_idx]

                self.step += 1

                # Collect rollouts
                prompts = [s["prompt"] for s in batch]
                expected = [s.get("expected_answer") for s in batch]

                rollouts = await self.env.collect_rollouts(
                    prompts,
                    expected,
                    num_rollouts=self.config.num_rollouts_per_prompt,
                )

                # Compute advantages
                advantages = self.env.compute_advantages(rollouts)

                # Policy gradient update
                loss = self.policy_gradient_step(rollouts, advantages)

                # Statistics
                batch_reward = sum(r["reward"] for r in rollouts) / len(rollouts)
                episode_rewards.append(batch_reward)
                episode_losses.append(loss)

                tool_usage = sum(r["info"]["tool_calls_count"] for r in rollouts) / len(rollouts)
                correct = sum(1 for r in rollouts if r["info"].get("answer_correct")) / len(rollouts)

                # Logging
                if self.step % self.config.log_every_n_steps == 0:
                    logger.info(
                        f"Step {self.step} | "
                        f"Batch {batch_idx + 1}/{num_batches} | "
                        f"Reward: {batch_reward:.3f} | "
                        f"Loss: {loss:.4f} | "
                        f"Tools: {tool_usage:.1f} | "
                        f"Correct: {correct:.1%}"
                    )

                # Save checkpoint
                if self.step % self.config.save_every_n_steps == 0:
                    self.save_checkpoint(f"step-{self.step}")

            # Episode summary
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            avg_loss = sum(episode_losses) / len(episode_losses)

            self.stats["rewards"].append(avg_reward)
            self.stats["losses"].append(avg_loss)

            logger.info(f"\nEpisode {episode + 1} Summary:")
            logger.info(f"  Avg Reward: {avg_reward:.3f}")
            logger.info(f"  Avg Loss: {avg_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(f"episode-{episode + 1}")

        # Final save
        self.save_checkpoint("final")
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save stats
        stats_path = os.path.join(save_path, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Checkpoint saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def load_config(config_path: str) -> AtroposConfig:
    """Load config from YAML file"""
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = AtroposConfig()

    if "model" in yaml_config:
        config.model_name = yaml_config["model"].get("name", config.model_name)

    if "training" in yaml_config:
        tc = yaml_config["training"]
        config.num_episodes = tc.get("num_train_epochs", config.num_episodes)
        config.learning_rate = tc.get("learning_rate", config.learning_rate)
        config.batch_size = tc.get("per_device_train_batch_size", config.batch_size)
        config.output_dir = tc.get("output_dir", config.output_dir)

    if "data" in yaml_config:
        dc = yaml_config["data"]
        config.dataset_name = dc.get("dataset_name", config.dataset_name)
        config.max_samples = dc.get("max_train_samples", config.max_samples)

    if "lora" in yaml_config:
        lc = yaml_config["lora"]
        config.use_lora = lc.get("enabled", config.use_lora)
        config.lora_r = lc.get("r", config.lora_r)
        config.lora_alpha = lc.get("lora_alpha", config.lora_alpha)

    if "rewards" in yaml_config:
        rc = yaml_config["rewards"]
        config.correct_reward = rc.get("correct_answer", config.correct_reward)
        config.incorrect_reward = rc.get("incorrect_answer", config.incorrect_reward)
        config.tool_bonus = rc.get("tool_usage_bonus", config.tool_bonus)

    return config


async def main():
    parser = argparse.ArgumentParser(description="Atropos RLVR Training")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--output-dir", type=str, default="./outputs/atropos-rlvr")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    args = parser.parse_args()

    # Load or create config
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = AtroposConfig()

    # Override with CLI args
    config.output_dir = args.output_dir
    config.max_samples = args.max_samples
    config.batch_size = args.batch_size
    config.num_episodes = args.num_episodes
    config.vllm_base_url = args.vllm_url

    # Create output dir
    os.makedirs(config.output_dir, exist_ok=True)

    # Add file handler for logging
    log_file = os.path.join(
        config.output_dir,
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info(f"Log file: {log_file}")
    logger.info(f"Config: {config}")

    # Train
    trainer = AtroposTrainer(config)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
