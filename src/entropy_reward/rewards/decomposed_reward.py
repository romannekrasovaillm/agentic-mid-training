"""Decomposed reward: R_format + R_tool + R_acc.

Two modes:
- strict: binary R_format (1 if perfect format, 0 otherwise)
- partial: graduated credit for partial format compliance
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardComponents:
    r_format: float = 0.0
    r_tool: float = 0.0
    r_acc: float = 0.0
    r_total: float = 0.0


class FormatChecker:
    """Check format compliance of model outputs."""

    # Expected structural patterns
    THOUGHT_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)
    ACTION_TAG = re.compile(r"<action>.*?</action>", re.DOTALL)
    ANSWER_TAG = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)

    @classmethod
    def strict_check(cls, text: str) -> float:
        """Binary format reward: 1.0 if all required tags present, else 0.0."""
        has_think = bool(cls.THOUGHT_TAG.search(text))
        has_action_or_answer = bool(cls.ACTION_TAG.search(text)) or bool(
            cls.ANSWER_TAG.search(text)
        )
        return 1.0 if (has_think and has_action_or_answer) else 0.0

    @classmethod
    def partial_check(
        cls,
        text: str,
        tag_credit: float = 0.3,
        structure_credit: float = 0.5,
        full_credit: float = 1.0,
    ) -> float:
        """Graduated format reward with partial credit.

        Levels:
        - tag_credit: at least one recognized tag present
        - structure_credit: has thinking + some action/answer structure
        - full_credit: fully compliant format
        """
        has_think = bool(cls.THOUGHT_TAG.search(text))
        has_action = bool(cls.ACTION_TAG.search(text))
        has_answer = bool(cls.ANSWER_TAG.search(text))
        has_json = bool(cls.JSON_BLOCK.search(text))

        # Full compliance
        if has_think and (has_action or has_answer):
            return full_credit

        # Structural: has think + some structured output
        if has_think and has_json:
            return structure_credit

        # At least one recognized tag
        if has_think or has_action or has_answer:
            return tag_credit

        return 0.0


class ToolUseChecker:
    """Evaluate tool usage quality."""

    TOOL_CALL_PATTERN = re.compile(r"<action>\s*(\w+)\s*\(", re.DOTALL)

    @classmethod
    def check(cls, text: str, available_tools: list[str] | None = None) -> float:
        """Reward for valid tool usage.

        Returns:
            0.0: no tool calls
            0.5: tool call attempted but invalid tool name
            1.0: valid tool call with recognized tool
        """
        matches = cls.TOOL_CALL_PATTERN.findall(text)
        if not matches:
            return 0.0

        if available_tools is None:
            return 1.0 if matches else 0.0

        valid = sum(1 for m in matches if m in available_tools)
        return valid / len(matches) if matches else 0.0

    @classmethod
    def count_calls(cls, text: str) -> int:
        return len(cls.TOOL_CALL_PATTERN.findall(text))


class DecomposedReward:
    """Compute decomposed reward with configurable format scoring."""

    def __init__(
        self,
        format_mode: str = "strict",
        format_weight: float = 1.0,
        tool_weight: float = 1.0,
        accuracy_weight: float = 1.0,
        partial_tag_credit: float = 0.3,
        partial_structure_credit: float = 0.5,
        partial_full_credit: float = 1.0,
        available_tools: list[str] | None = None,
    ):
        self.format_mode = format_mode
        self.format_weight = format_weight
        self.tool_weight = tool_weight
        self.accuracy_weight = accuracy_weight
        self.partial_tag_credit = partial_tag_credit
        self.partial_structure_credit = partial_structure_credit
        self.partial_full_credit = partial_full_credit
        self.available_tools = available_tools

    def compute(
        self,
        text: str,
        accuracy_score: float = 0.0,
        available_tools: list[str] | None = None,
    ) -> RewardComponents:
        """Compute all reward components for a single trajectory.

        Args:
            text: model output text
            accuracy_score: external accuracy signal (0-1)
            available_tools: override available tools list
        """
        tools = available_tools or self.available_tools

        # R_format
        if self.format_mode == "strict":
            r_format = FormatChecker.strict_check(text)
        else:
            r_format = FormatChecker.partial_check(
                text,
                self.partial_tag_credit,
                self.partial_structure_credit,
                self.partial_full_credit,
            )

        # R_tool
        r_tool = ToolUseChecker.check(text, tools)

        # R_acc
        r_acc = accuracy_score

        # Weighted total
        r_total = (
            self.format_weight * r_format
            + self.tool_weight * r_tool
            + self.accuracy_weight * r_acc
        )

        return RewardComponents(
            r_format=r_format,
            r_tool=r_tool,
            r_acc=r_acc,
            r_total=r_total,
        )

    def compute_batch(
        self,
        texts: list[str],
        accuracy_scores: list[float],
        available_tools: list[str] | None = None,
    ) -> list[RewardComponents]:
        return [
            self.compute(t, a, available_tools) for t, a in zip(texts, accuracy_scores)
        ]
