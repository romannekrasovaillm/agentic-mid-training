"""Metamorphic testing: apply semantics-preserving transformations and check consistency."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class MetamorphicResult:
    transform_name: str
    total: int
    consistent: int
    consistency_rate: float


class MetamorphicTester:
    """Apply metamorphic transformations and measure output consistency.

    If a transform preserves semantics, the model should produce
    equivalent outputs (same answer, same tool choice).
    """

    TRANSFORMS = {
        "reorder_tools": "_transform_reorder_tools",
        "synonym_replace": "_transform_synonym_replace",
        "format_shift": "_transform_format_shift",
        "whitespace_variation": "_transform_whitespace",
        "instruction_paraphrase": "_transform_paraphrase",
    }

    SYNONYMS = {
        "calculate": ["compute", "evaluate", "determine"],
        "search": ["find", "look up", "query"],
        "analyze": ["examine", "inspect", "review"],
        "create": ["generate", "produce", "make"],
        "explain": ["describe", "clarify", "elaborate on"],
    }

    def __init__(
        self,
        transforms: list[str] | None = None,
        generate_fn: Callable | None = None,
        answer_extract_fn: Callable | None = None,
    ):
        self.transforms = transforms or list(self.TRANSFORMS.keys())
        self.generate_fn = generate_fn
        self.answer_extract_fn = answer_extract_fn or self._default_answer_extract

    def set_generate_fn(self, fn: Callable):
        self.generate_fn = fn

    def test(self, prompts: list[str], tools: list[str] | None = None) -> list[MetamorphicResult]:
        """Run metamorphic tests across all configured transforms."""
        results = []
        for transform_name in self.transforms:
            if transform_name not in self.TRANSFORMS:
                continue
            method = getattr(self, self.TRANSFORMS[transform_name])
            result = self._run_transform(transform_name, method, prompts, tools)
            results.append(result)
        return results

    def _run_transform(
        self,
        name: str,
        transform_fn: Callable,
        prompts: list[str],
        tools: list[str] | None,
    ) -> MetamorphicResult:
        consistent = 0
        total = len(prompts)

        for prompt in prompts:
            transformed = transform_fn(prompt, tools)
            if self.generate_fn is None:
                continue

            original_output = self.generate_fn(prompt, tools)
            transformed_output = self.generate_fn(transformed, tools)

            original_answer = self.answer_extract_fn(original_output)
            transformed_answer = self.answer_extract_fn(transformed_output)

            if self._answers_equivalent(original_answer, transformed_answer):
                consistent += 1

        return MetamorphicResult(
            transform_name=name,
            total=total,
            consistent=consistent,
            consistency_rate=consistent / max(total, 1),
        )

    def _transform_reorder_tools(self, prompt: str, tools: list[str] | None) -> str:
        """Reorder tool descriptions in the prompt."""
        if tools and len(tools) > 1:
            shuffled = tools.copy()
            random.shuffle(shuffled)
            tool_str = "Available tools: " + ", ".join(f"{t}()" for t in shuffled)
            # Replace existing tool description if present
            prompt = re.sub(r"Available tools:.*?\n", "", prompt)
            return f"{tool_str}\n{prompt}"
        return prompt

    def _transform_synonym_replace(self, prompt: str, tools: list[str] | None) -> str:
        """Replace action verbs with synonyms."""
        result = prompt
        for word, synonyms in self.SYNONYMS.items():
            if word in result.lower():
                replacement = random.choice(synonyms)
                result = re.sub(rf"\b{word}\b", replacement, result, flags=re.IGNORECASE)
        return result

    def _transform_format_shift(self, prompt: str, tools: list[str] | None) -> str:
        """Shift format instruction phrasing while preserving semantics."""
        shifts = [
            ("step by step", "one step at a time"),
            ("think carefully", "reason through this"),
            ("use the tools", "leverage the available tools"),
            ("provide your answer", "give your response"),
        ]
        result = prompt
        for original, replacement in shifts:
            result = result.replace(original, replacement)
        return result

    def _transform_whitespace(self, prompt: str, tools: list[str] | None) -> str:
        """Vary whitespace: extra newlines, spaces."""
        return prompt.replace("\n", "\n\n").replace("  ", " ")

    def _transform_paraphrase(self, prompt: str, tools: list[str] | None) -> str:
        """Simple instruction paraphrase."""
        paraphrases = [
            ("Please ", "Could you "),
            ("What is", "Tell me"),
            ("How do", "What is the way to"),
            ("Explain", "Describe"),
        ]
        result = prompt
        for original, replacement in paraphrases:
            if original in result:
                result = result.replace(original, replacement, 1)
                break
        return result

    @staticmethod
    def _default_answer_extract(text: str) -> str:
        """Extract answer from <answer> tags or last line."""
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = text.strip().split("\n")
        return lines[-1].strip() if lines else ""

    @staticmethod
    def _answers_equivalent(a: str, b: str) -> bool:
        """Check if two answers are semantically equivalent (simplified)."""
        # Normalize whitespace and case
        a_norm = " ".join(a.lower().split())
        b_norm = " ".join(b.lower().split())
        return a_norm == b_norm
