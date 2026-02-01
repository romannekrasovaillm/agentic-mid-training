"""Decomposed reward: R_format + R_tool + R_acc.

Two modes:
- strict: binary R_format (1 if perfect format, 0 otherwise)
- partial: graduated credit for partial format compliance
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    r_format: float = 0.0
    r_tool: float = 0.0
    r_acc: float = 0.0
    r_total: float = 0.0


@dataclass
class RewardDiagnosis:
    """Human-readable explanation of why each reward component scored what it did."""

    # Format diagnosis
    has_think: bool = False
    has_action: bool = False
    has_answer: bool = False
    has_json: bool = False
    format_reason: str = ""

    # Tool diagnosis
    tool_calls_found: list[str] = field(default_factory=list)
    tool_pattern_raw: str = ""  # first 200 chars around <action> if present
    tool_reason: str = ""

    # Combined human-readable reason
    reason: str = ""


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
    def diagnose(cls, text: str) -> dict[str, Any]:
        """Return a dict explaining which format signals were found/missing."""
        has_think = bool(cls.THOUGHT_TAG.search(text))
        has_action = bool(cls.ACTION_TAG.search(text))
        has_answer = bool(cls.ANSWER_TAG.search(text))
        has_json = bool(cls.JSON_BLOCK.search(text))

        reasons = []
        if not has_think:
            reasons.append("no <think>...</think> tag")
        if not has_action and not has_answer:
            reasons.append("no <action>...</action> or <answer>...</answer> tag")
        if has_think and (has_action or has_answer):
            reasons.append("fully compliant format")

        # Show what the text starts with to help debug
        preview = text[:200].replace("\n", " ↵ ").strip()

        return {
            "has_think": has_think,
            "has_action": has_action,
            "has_answer": has_answer,
            "has_json": has_json,
            "reason": "; ".join(reasons) if reasons else "unknown",
            "text_preview": preview,
        }

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
    def diagnose(cls, text: str, available_tools: list[str] | None = None) -> dict[str, Any]:
        """Return a dict explaining tool-call detection results."""
        matches = cls.TOOL_CALL_PATTERN.findall(text)

        # Also check for <action> tag without the function-call pattern
        has_action_tag = bool(re.search(r"<action>", text))
        action_contents = re.findall(r"<action>(.*?)</action>", text, re.DOTALL)

        reasons = []
        if not has_action_tag:
            reasons.append("no <action> tag found at all")
        elif not matches:
            # Has <action> but pattern didn't match — show what's inside
            for ac in action_contents[:3]:
                snippet = ac.strip()[:100]
                reasons.append(f"<action> found but no tool_name( pattern inside: '{snippet}'")
        else:
            if available_tools is not None:
                valid = [m for m in matches if m in available_tools]
                invalid = [m for m in matches if m not in available_tools]
                if invalid:
                    reasons.append(
                        f"tool names not in available_tools: {invalid} "
                        f"(available: {available_tools})"
                    )
                if valid:
                    reasons.append(f"valid tool calls: {valid}")
            else:
                reasons.append(f"found tool calls: {matches} (no available_tools filter)")

        return {
            "tool_calls_found": matches,
            "has_action_tag": has_action_tag,
            "action_contents": [ac.strip()[:100] for ac in action_contents[:3]],
            "reason": "; ".join(reasons) if reasons else "no tool usage detected",
        }

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

    def compute_with_diagnosis(
        self,
        text: str,
        accuracy_score: float = 0.0,
        available_tools: list[str] | None = None,
    ) -> tuple[RewardComponents, dict[str, Any]]:
        """Compute reward + detailed diagnosis explaining the scores.

        Returns:
            (RewardComponents, diagnosis_dict) where diagnosis_dict has keys:
              has_think, has_action, has_answer, has_json, format_reason,
              tool_calls_found, tool_reason, reason
        """
        tools = available_tools or self.available_tools
        result = self.compute(text, accuracy_score, available_tools)

        fmt_diag = FormatChecker.diagnose(text)
        tool_diag = ToolUseChecker.diagnose(text, tools)

        # Build combined reason
        parts = []
        if result.r_format == 0.0:
            parts.append(f"R_fmt=0: {fmt_diag['reason']}")
        else:
            parts.append(f"R_fmt={result.r_format:.2f}: {fmt_diag['reason']}")
        if result.r_tool == 0.0:
            parts.append(f"R_tool=0: {tool_diag['reason']}")
        else:
            parts.append(f"R_tool={result.r_tool:.2f}: {tool_diag['reason']}")

        diag = {
            # Format
            "has_think": fmt_diag["has_think"],
            "has_action": fmt_diag["has_action"],
            "has_answer": fmt_diag["has_answer"],
            "has_json": fmt_diag.get("has_json", False),
            "format_reason": fmt_diag["reason"],
            "text_preview": fmt_diag.get("text_preview", ""),
            # Tool
            "tool_calls_found": tool_diag["tool_calls_found"],
            "has_action_tag": tool_diag.get("has_action_tag", False),
            "action_contents": tool_diag.get("action_contents", []),
            "tool_reason": tool_diag["reason"],
            # Combined
            "reason": " | ".join(parts),
        }
        return result, diag

    def compute_batch(
        self,
        texts: list[str],
        accuracy_scores: list[float],
        available_tools: list[str] | None = None,
    ) -> list[RewardComponents]:
        return [
            self.compute(t, a, available_tools) for t, a in zip(texts, accuracy_scores)
        ]

    def compute_batch_with_diagnosis(
        self,
        texts: list[str],
        accuracy_scores: list[float],
        available_tools: list[str] | None = None,
    ) -> tuple[list[RewardComponents], list[dict[str, Any]]]:
        """Compute rewards + diagnoses for a batch.

        Returns:
            (list[RewardComponents], list[diagnosis_dicts])
        """
        results = []
        diags = []
        for t, a in zip(texts, accuracy_scores):
            r, d = self.compute_with_diagnosis(t, a, available_tools)
            results.append(r)
            diags.append(d)
        return results, diags
