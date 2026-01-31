"""Dataset loader for nvidia/Nemotron-Agentic-v1.

Loads multi-turn tool-use conversations and prepares them for GRPO training:
- Extracts prompts (system + user turns)
- Extracts reference tool calls (for accuracy scoring)
- Extracts available tool schemas per conversation
- Supports both splits: interactive_agent, tool_calling
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class AgenticSample:
    """Single training sample extracted from Nemotron-Agentic-v1."""

    uuid: str = ""
    # Formatted prompt for the model (system + user context)
    prompt: str = ""
    # Reference assistant response (for accuracy evaluation)
    reference_response: str = ""
    # Tool schemas available in this conversation
    tools: list[dict[str, Any]] = field(default_factory=list)
    # Tool names extracted from schemas
    tool_names: list[str] = field(default_factory=list)
    # Reference tool calls made by the assistant (ground truth)
    reference_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Whether this sample contains reasoning traces
    has_reasoning: bool = False
    # Number of turns in the original conversation
    num_turns: int = 0


def _extract_tool_names(tools: list[dict]) -> list[str]:
    """Extract tool function names from tool schema list."""
    names = []
    for t in tools:
        if isinstance(t, dict):
            fn = t.get("function", {})
            name = fn.get("name", "")
            if name:
                names.append(name)
    return names


def _build_prompt(messages: list[dict], tools: list[dict]) -> str:
    """Build a prompt from system + user messages up to the first assistant turn.

    Format follows GigaChat3 chat template convention.
    """
    parts = []

    # System message
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                parts.append(f"[System]\n{content}")
            break

    # Tool definitions (compact)
    if tools:
        tool_descs = []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")
            tool_descs.append(f"- {name}: {desc}")
        if tool_descs:
            parts.append("[Available Tools]\n" + "\n".join(tool_descs))

    # User turns up to first assistant response
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and content:
            parts.append(f"[User]\n{content}")
            break  # Take first user turn as the prompt

    return "\n\n".join(parts)


def _build_multiturn_prompt(messages: list[dict], tools: list[dict], max_turns: int = 3) -> str:
    """Build prompt from multi-turn context (up to max_turns user-assistant pairs)."""
    parts = []

    # System
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                parts.append(f"[System]\n{content}")
            break

    # Tools
    if tools:
        tool_descs = []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")
            tool_descs.append(f"- {name}: {desc}")
        if tool_descs:
            parts.append("[Available Tools]\n" + "\n".join(tool_descs))

    # Collect conversation turns (skip system)
    turn_count = 0
    last_user_idx = -1
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue

        if role == "user":
            parts.append(f"[User]\n{content}")
            last_user_idx = i
            turn_count += 1
            if turn_count >= max_turns:
                break
        elif role == "assistant" and turn_count < max_turns:
            # Include prior assistant turns as context
            if content:
                parts.append(f"[Assistant]\n{content}")
            # Include tool calls if present
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                calls_str = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    calls_str.append(f"  <action>{fn.get('name', '')}({fn.get('arguments', '')})</action>")
                parts.append("[Tool Calls]\n" + "\n".join(calls_str))
        elif role == "tool":
            if content:
                tool_id = msg.get("tool_call_id", "")
                parts.append(f"[Tool Result: {tool_id}]\n{content[:500]}")

    return "\n\n".join(parts)


def _extract_reference(messages: list[dict]) -> tuple[str, list[dict]]:
    """Extract the first assistant response and its tool calls as reference."""
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""
            tool_calls = msg.get("tool_calls", []) or []
            reasoning = msg.get("reasoning_content", "") or ""
            if reasoning:
                content = f"<think>{reasoning}</think>\n{content}"
            return content, tool_calls
    return "", []


class NemotronAgenticLoader:
    """Load and prepare nvidia/Nemotron-Agentic-v1 for GRPO training."""

    def __init__(
        self,
        split: str = "tool_calling",
        max_samples: int = 0,
        multiturn: bool = False,
        max_context_turns: int = 3,
        seed: int = 42,
        cache_dir: str | None = None,
    ):
        self.split = split
        self.max_samples = max_samples
        self.multiturn = multiturn
        self.max_context_turns = max_context_turns
        self.seed = seed
        self.cache_dir = cache_dir

    def load(self) -> list[AgenticSample]:
        """Load dataset and convert to AgenticSample list."""
        logger.info(f"Loading nvidia/Nemotron-Agentic-v1 split={self.split}")

        # Use streaming=True to avoid Arrow schema cast errors —
        # Nemotron-Agentic-v1 has heterogeneous nested tool parameter
        # schemas across examples that cannot be reconciled into a
        # single Arrow table by the datasets library.
        ds = load_dataset(
            "nvidia/Nemotron-Agentic-v1",
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=True,
        )

        samples = []
        raw_count = 0
        for row in ds:
            raw_count += 1
            sample = self._process_row(row)
            if sample is not None:
                samples.append(sample)

        logger.info(f"Loaded {raw_count} raw examples from split={self.split}")

        if self.max_samples > 0 and len(samples) > self.max_samples:
            rng = random.Random(self.seed)
            samples = rng.sample(samples, self.max_samples)

        logger.info(
            f"Prepared {len(samples)} samples "
            f"({sum(1 for s in samples if s.reference_tool_calls)} with tool calls, "
            f"{sum(1 for s in samples if s.has_reasoning)} with reasoning)"
        )
        return samples

    def _process_row(self, row: dict) -> AgenticSample | None:
        """Convert a single dataset row to AgenticSample."""
        messages = row.get("messages", [])
        if not messages:
            return None

        tools = row.get("tools", []) or []
        uuid = row.get("uuid", "")

        # Build prompt
        if self.multiturn:
            prompt = _build_multiturn_prompt(messages, tools, self.max_context_turns)
        else:
            prompt = _build_prompt(messages, tools)

        if not prompt.strip():
            return None

        # Extract reference
        ref_response, ref_tool_calls = _extract_reference(messages)
        tool_names = _extract_tool_names(tools)

        # Check for reasoning
        has_reasoning = any(
            bool(m.get("reasoning_content"))
            for m in messages
            if m.get("role") == "assistant"
        )

        return AgenticSample(
            uuid=uuid,
            prompt=prompt,
            reference_response=ref_response,
            tools=tools,
            tool_names=tool_names,
            reference_tool_calls=ref_tool_calls,
            has_reasoning=has_reasoning,
            num_turns=sum(1 for m in messages if m.get("role") in ("user", "assistant")),
        )


def load_nemotron_splits(
    max_train: int = 0,
    max_eval: int = 200,
    eval_ratio: float = 0.1,
    multiturn: bool = False,
    seed: int = 42,
    cache_dir: str | None = None,
) -> tuple[list[AgenticSample], list[AgenticSample]]:
    """Load train and eval splits from Nemotron-Agentic-v1.

    Strategy:
    - tool_calling split -> training data (structured tool use)
    - interactive_agent split -> eval / OOD data (open-ended agent dialogues)
    - If interactive_agent is unavailable, split tool_calling by eval_ratio

    Returns:
        (train_samples, eval_samples)
    """
    # Primary: tool_calling for training
    train_loader = NemotronAgenticLoader(
        split="tool_calling",
        max_samples=max_train,
        multiturn=multiturn,
        seed=seed,
        cache_dir=cache_dir,
    )
    train_samples = train_loader.load()

    # Try interactive_agent for eval (OOD relative to tool_calling)
    eval_samples = []
    try:
        eval_loader = NemotronAgenticLoader(
            split="interactive_agent",
            max_samples=max_eval,
            multiturn=True,  # eval uses full context
            seed=seed,
            cache_dir=cache_dir,
        )
        eval_samples = eval_loader.load()
        logger.info(f"Loaded {len(eval_samples)} eval samples from interactive_agent split")
    except Exception as e:
        logger.warning(f"Could not load interactive_agent split: {e}")
        logger.info("Splitting tool_calling for eval instead")

    # Fallback: split train if no separate eval
    if not eval_samples:
        rng = random.Random(seed)
        rng.shuffle(train_samples)
        n_eval = min(max_eval, int(len(train_samples) * eval_ratio))
        eval_samples = train_samples[:n_eval]
        train_samples = train_samples[n_eval:]

    return train_samples, eval_samples
