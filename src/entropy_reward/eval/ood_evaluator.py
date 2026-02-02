"""Out-of-distribution evaluator for format and tool robustness.

Tests model on variations of format/tools not seen during training.
Measures robustness and recovery speed after stagnation.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class OODResult:
    dataset_name: str
    total: int
    format_pass: int
    tool_pass: int
    accuracy: float
    format_pass_rate: float
    tool_pass_rate: float


class OODEvaluator:
    """Evaluate model on OOD format and tool variations."""

    # Format variation templates
    FORMAT_VARIATIONS = {
        "xml_tags": {
            "instruction": "Respond using <reasoning>...</reasoning> and <result>...</result> tags.",
            "check_tags": ["reasoning", "result"],
        },
        "markdown_format": {
            "instruction": "Use ## Thinking for reasoning and ## Answer for your final answer.",
            "check_tags": ["## Thinking", "## Answer"],
        },
        "json_format": {
            "instruction": 'Respond with a JSON object having "thought" and "action" keys.',
            "check_tags": ['"thought"', '"action"'],
        },
        "numbered_steps": {
            "instruction": "Use numbered steps (1. 2. 3.) for reasoning, then ANSWER: for final answer.",
            "check_tags": ["1.", "ANSWER:"],
        },
    }

    # Tool variation templates — each value is a list of tool names to present
    # and check for.  "renamed_tools" uses names NOT seen during training.
    TOOL_VARIATIONS = {
        "renamed_tools": ["web_lookup", "math_eval", "execute_code"],
        "extra_tools": ["search", "calculate", "summarize", "translate"],
        "minimal_tools": ["do_task"],
    }

    # Regex for checking actual tool *calls* (not just substrings).
    # Matches <action>tool_name(  — same pattern as ToolUseChecker.
    _TOOL_CALL_RE = re.compile(r"<action>\s*(\w+)\s*\(")

    def __init__(
        self,
        datasets: list[str] | None = None,
        generate_fn: Callable | None = None,
    ):
        self.datasets = datasets or ["format_variation", "tool_variation"]
        self.generate_fn = generate_fn
        self.generate_batch_fn: Callable | None = None
        self._results_history: list[list[OODResult]] = []

    def set_generate_fn(self, fn: Callable):
        """Set the model generation function: fn(prompt, tools) -> str."""
        self.generate_fn = fn

    def set_generate_batch_fn(self, fn: Callable):
        """Set the batched generation function: fn(prompts) -> list[str]."""
        self.generate_batch_fn = fn

    @classmethod
    def _check_tool_usage(cls, output: str, tool_names: list[str]) -> bool:
        """Check if output contains an actual tool call for any of the given names.

        Uses <action>tool_name( pattern, not bare substring, to avoid
        false positives (e.g. the word 'do' in normal English text).
        """
        calls = cls._TOOL_CALL_RE.findall(output)
        return any(c in tool_names for c in calls)

    def evaluate(self, test_prompts: list[str]) -> list[OODResult]:
        """Run OOD evaluation across all configured datasets.

        When generate_batch_fn is available, collects ALL prompts across
        all variations and generates in one batch for speed.

        Args:
            test_prompts: base prompts to test with format/tool variations
        """
        if self.generate_batch_fn is not None:
            return self._evaluate_batched(test_prompts)

        # Fallback: sequential per-prompt generation
        results = []

        if "format_variation" in self.datasets:
            for var_name, var_config in self.FORMAT_VARIATIONS.items():
                result = self._eval_format_variation_seq(test_prompts, var_name, var_config)
                results.append(result)

        if "tool_variation" in self.datasets:
            for var_name, tool_names in self.TOOL_VARIATIONS.items():
                result = self._eval_tool_variation_seq(test_prompts, var_name, tool_names)
                results.append(result)

        self._results_history.append(results)
        return results

    def _evaluate_batched(self, test_prompts: list[str]) -> list[OODResult]:
        """Batched evaluation: collect all prompts, generate once, score."""
        # Collect all modified prompts and metadata
        batch_prompts: list[str] = []
        # Each entry: (variation_type, var_name, var_config/tool_names, prompt_idx)
        batch_meta: list[tuple[str, str, Any, int]] = []

        if "format_variation" in self.datasets:
            for var_name, var_config in self.FORMAT_VARIATIONS.items():
                for i, prompt in enumerate(test_prompts):
                    modified = f"{var_config['instruction']}\n\n{prompt}"
                    batch_prompts.append(modified)
                    batch_meta.append(("format", var_name, var_config, i))

        if "tool_variation" in self.datasets:
            for var_name, tool_names in self.TOOL_VARIATIONS.items():
                for i, prompt in enumerate(test_prompts):
                    tool_desc = "Available tools: " + ", ".join(
                        f"{t}()" for t in tool_names
                    )
                    modified = f"{tool_desc}\n\n{prompt}"
                    batch_prompts.append(modified)
                    batch_meta.append(("tool", var_name, tool_names, i))

        if not batch_prompts:
            self._results_history.append([])
            return []

        # Single batched generation call
        all_outputs = self.generate_batch_fn(batch_prompts)

        # Score results grouped by variation
        from collections import defaultdict
        format_pass_counts: dict[str, int] = defaultdict(int)
        tool_pass_counts: dict[str, int] = defaultdict(int)
        variation_totals: dict[str, int] = defaultdict(int)

        for output, (vtype, vname, vconfig, _pidx) in zip(all_outputs, batch_meta):
            key = f"{vtype}/{vname}"
            variation_totals[key] += 1

            if vtype == "format":
                if all(tag in output for tag in vconfig["check_tags"]):
                    format_pass_counts[key] += 1
            elif vtype == "tool":
                # vconfig is a list[str] of tool names
                if self._check_tool_usage(output, vconfig):
                    tool_pass_counts[key] += 1

        # Build OODResult list in same order as sequential version
        results = []
        if "format_variation" in self.datasets:
            for var_name in self.FORMAT_VARIATIONS:
                key = f"format/{var_name}"
                total = variation_totals.get(key, 0)
                fp = format_pass_counts.get(key, 0)
                results.append(OODResult(
                    dataset_name=key,
                    total=total,
                    format_pass=fp,
                    tool_pass=0,
                    accuracy=0.0,
                    format_pass_rate=fp / max(total, 1),
                    tool_pass_rate=0.0,
                ))

        if "tool_variation" in self.datasets:
            for var_name in self.TOOL_VARIATIONS:
                key = f"tool/{var_name}"
                total = variation_totals.get(key, 0)
                tp = tool_pass_counts.get(key, 0)
                results.append(OODResult(
                    dataset_name=key,
                    total=total,
                    format_pass=0,
                    tool_pass=tp,
                    accuracy=0.0,
                    format_pass_rate=0.0,
                    tool_pass_rate=tp / max(total, 1),
                ))

        self._results_history.append(results)
        return results

    def _eval_format_variation_seq(
        self, prompts: list[str], name: str, config: dict
    ) -> OODResult:
        """Test model with a different format instruction (sequential)."""
        format_pass = 0
        total = len(prompts)

        for prompt in prompts:
            modified_prompt = f"{config['instruction']}\n\n{prompt}"
            if self.generate_fn:
                output = self.generate_fn(modified_prompt, None)
                if all(tag in output for tag in config["check_tags"]):
                    format_pass += 1

        return OODResult(
            dataset_name=f"format/{name}",
            total=total,
            format_pass=format_pass,
            tool_pass=0,
            accuracy=0.0,
            format_pass_rate=format_pass / max(total, 1),
            tool_pass_rate=0.0,
        )

    def _eval_tool_variation_seq(
        self, prompts: list[str], name: str, tool_names: list[str]
    ) -> OODResult:
        """Test model with different tool names (sequential)."""
        tool_pass = 0
        total = len(prompts)

        for prompt in prompts:
            tool_desc = "Available tools: " + ", ".join(
                f"{t}()" for t in tool_names
            )
            modified_prompt = f"{tool_desc}\n\n{prompt}"
            if self.generate_fn:
                output = self.generate_fn(modified_prompt, tool_names)
                if self._check_tool_usage(output, tool_names):
                    tool_pass += 1

        return OODResult(
            dataset_name=f"tool/{name}",
            total=total,
            format_pass=0,
            tool_pass=tool_pass,
            accuracy=0.0,
            format_pass_rate=0.0,
            tool_pass_rate=tool_pass / max(total, 1),
        )

    def recovery_speed(self) -> float | None:
        """Measure how quickly pass rates recover after a dip.

        Returns number of eval rounds to recover to 90% of pre-dip level,
        or None if no recovery detected.
        """
        if len(self._results_history) < 3:
            return None

        # Aggregate format pass rates over time
        rates = []
        for results in self._results_history:
            avg_rate = sum(r.format_pass_rate for r in results) / max(len(results), 1)
            rates.append(avg_rate)

        # Find dip and recovery
        peak = max(rates[:len(rates) // 2]) if rates else 0
        threshold = peak * 0.9

        dip_idx = None
        for i in range(1, len(rates)):
            if rates[i] < threshold and dip_idx is None:
                dip_idx = i
            elif dip_idx is not None and rates[i] >= threshold:
                return float(i - dip_idx)

        return None

    @property
    def history(self) -> list[list[OODResult]]:
        return self._results_history
