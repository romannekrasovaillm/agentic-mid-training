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
        self.generate_batch_fn: Callable | None = None
        self.answer_extract_fn = answer_extract_fn or self._default_answer_extract

    def set_generate_fn(self, fn: Callable):
        self.generate_fn = fn

    def set_generate_batch_fn(self, fn: Callable):
        """Set the batched generation function: fn(prompts) -> list[str]."""
        self.generate_batch_fn = fn

    def test(self, prompts: list[str], tools: list[str] | None = None) -> list[MetamorphicResult]:
        """Run metamorphic tests across all configured transforms.

        When generate_batch_fn is available, collects ALL original+transformed
        prompts and generates in one batch for speed.
        """
        if self.generate_batch_fn is not None:
            return self._test_batched(prompts, tools)

        # Fallback: sequential
        results = []
        for transform_name in self.transforms:
            if transform_name not in self.TRANSFORMS:
                continue
            method = getattr(self, self.TRANSFORMS[transform_name])
            result = self._run_transform_seq(transform_name, method, prompts, tools)
            results.append(result)
        return results

    def _test_batched(self, prompts: list[str], tools: list[str] | None) -> list[MetamorphicResult]:
        """Batched metamorphic testing: one generation call for all prompts."""
        # Collect all prompts: for each transform × prompt we need
        # (original, transformed) pair → 2 generations per pair.
        batch_prompts: list[str] = []
        # Meta: (transform_name, prompt_idx, is_transformed)
        batch_meta: list[tuple[str, int, bool]] = []

        valid_transforms: list[str] = []
        for transform_name in self.transforms:
            if transform_name not in self.TRANSFORMS:
                continue
            valid_transforms.append(transform_name)
            method = getattr(self, self.TRANSFORMS[transform_name])
            for i, prompt in enumerate(prompts):
                transformed = method(prompt, tools)
                # Original
                batch_prompts.append(prompt)
                batch_meta.append((transform_name, i, False))
                # Transformed
                batch_prompts.append(transformed)
                batch_meta.append((transform_name, i, True))

        if not batch_prompts:
            return []

        # Single batched generation call
        all_outputs = self.generate_batch_fn(batch_prompts)

        # Pair up results: group by (transform_name, prompt_idx)
        # outputs come in pairs: [original_0, transformed_0, original_1, ...]
        from collections import defaultdict
        originals: dict[tuple[str, int], str] = {}
        transformeds: dict[tuple[str, int], str] = {}

        for output, (tname, pidx, is_trans) in zip(all_outputs, batch_meta):
            key = (tname, pidx)
            if is_trans:
                transformeds[key] = output
            else:
                originals[key] = output

        # Score consistency per transform
        results = []
        for transform_name in valid_transforms:
            consistent = 0
            total = len(prompts)
            for i in range(total):
                key = (transform_name, i)
                orig_out = originals.get(key, "")
                trans_out = transformeds.get(key, "")
                orig_answer = self.answer_extract_fn(orig_out)
                trans_answer = self.answer_extract_fn(trans_out)
                if self._answers_equivalent(orig_answer, trans_answer):
                    consistent += 1
            results.append(MetamorphicResult(
                transform_name=transform_name,
                total=total,
                consistent=consistent,
                consistency_rate=consistent / max(total, 1),
            ))

        return results

    def _run_transform_seq(
        self,
        name: str,
        transform_fn: Callable,
        prompts: list[str],
        tools: list[str] | None,
    ) -> MetamorphicResult:
        """Sequential per-prompt transform evaluation (fallback)."""
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
        """Reorder tool descriptions in the prompt.

        If explicit tools list is not provided, tries to extract tool names
        from an "Available tools:" line in the prompt itself.
        """
        if not tools:
            # Try to extract from prompt text
            match = re.search(r"Available tools:\s*(.+?)(?:\n|$)", prompt)
            if match:
                tools = re.findall(r"(\w+)\(\)", match.group(1))

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
    def _answers_equivalent(a: str, b: str, threshold: float = 0.5) -> bool:
        """Check if two answers are semantically equivalent via token F1.

        Exact string match is too strict for LLMs (temperature > 0 produces
        different wording for the same content).  Token F1 captures whether
        the same key information is present regardless of phrasing.
        """
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())

        if not a_tokens and not b_tokens:
            return True
        if not a_tokens or not b_tokens:
            return False

        common = a_tokens & b_tokens
        precision = len(common) / len(a_tokens)
        recall = len(common) / len(b_tokens)
        if precision + recall == 0:
            return False
        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= threshold
