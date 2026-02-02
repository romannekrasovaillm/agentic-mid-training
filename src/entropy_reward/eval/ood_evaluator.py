"""Out-of-distribution evaluator for format and tool robustness.

Tests model on variations of format/tools not seen during training.
Measures robustness and recovery speed after stagnation.
"""

from __future__ import annotations

import random
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

    # Tool variation templates
    TOOL_VARIATIONS = {
        "renamed_tools": {
            "search": "web_lookup",
            "calculate": "math_eval",
            "code_run": "execute_code",
        },
        "extra_tools": {
            "search": "search",
            "calculate": "calculate",
            "summarize": "summarize",
            "translate": "translate",
        },
        "minimal_tools": {
            "do": "do",
        },
    }

    def __init__(
        self,
        datasets: list[str] | None = None,
        generate_fn: Callable | None = None,
    ):
        self.datasets = datasets or ["format_variation", "tool_variation"]
        self.generate_fn = generate_fn
        self._results_history: list[list[OODResult]] = []

    def set_generate_fn(self, fn: Callable):
        """Set the model generation function: fn(prompt, tools) -> str."""
        self.generate_fn = fn

    def evaluate(self, test_prompts: list[str]) -> list[OODResult]:
        """Run OOD evaluation across all configured datasets.

        Args:
            test_prompts: base prompts to test with format/tool variations
        """
        results = []

        if "format_variation" in self.datasets:
            for var_name, var_config in self.FORMAT_VARIATIONS.items():
                result = self._eval_format_variation(test_prompts, var_name, var_config)
                results.append(result)

        if "tool_variation" in self.datasets:
            for var_name, var_config in self.TOOL_VARIATIONS.items():
                result = self._eval_tool_variation(test_prompts, var_name, var_config)
                results.append(result)

        self._results_history.append(results)
        return results

    def _eval_format_variation(
        self, prompts: list[str], name: str, config: dict
    ) -> OODResult:
        """Test model with a different format instruction."""
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

    def _eval_tool_variation(
        self, prompts: list[str], name: str, config: dict
    ) -> OODResult:
        """Test model with different tool names."""
        tool_pass = 0
        total = len(prompts)
        tools = list(config.keys())

        for prompt in prompts:
            tool_desc = "Available tools: " + ", ".join(
                f"{k}()" for k in config.keys()
            )
            modified_prompt = f"{tool_desc}\n\n{prompt}"
            if self.generate_fn:
                output = self.generate_fn(modified_prompt, tools)
                if any(t in output for t in config.keys()):
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
