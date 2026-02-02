"""Accuracy scoring for Nemotron-Agentic-v1 tool-use outputs.

Compares model generations against reference tool calls and responses.
Produces a 0-1 accuracy score used as R_acc in the decomposed reward.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class AccuracyResult:
    score: float = 0.0  # overall 0-1
    tool_name_match: float = 0.0  # fraction of correct tool names
    tool_args_match: float = 0.0  # fraction of correct arguments
    response_overlap: float = 0.0  # token overlap with reference


# Patterns to extract tool calls from generated text
ACTION_PATTERN = re.compile(r"<action>\s*(\w+)\s*\(([^)]*)\)\s*</action>", re.DOTALL)
FUNCTION_CALL_PATTERN = re.compile(
    r'"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*"?(\{[^}]*\})"?', re.DOTALL
)


def extract_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    """Extract tool calls from generated text.

    Supports formats:
    - <action>tool_name(args)</action>
    - JSON function call blocks
    """
    calls = []

    # Format 1: <action>tool(args)</action>
    for match in ACTION_PATTERN.finditer(text):
        name = match.group(1)
        args_str = match.group(2).strip()
        args = _parse_args(args_str)
        calls.append({"name": name, "arguments": args})

    if calls:
        return calls

    # Format 2: JSON function calls
    for match in FUNCTION_CALL_PATTERN.finditer(text):
        name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            args = {"raw": args_str}
        calls.append({"name": name, "arguments": args})

    return calls


def _parse_args(args_str: str) -> dict[str, Any]:
    """Parse tool arguments from string."""
    if not args_str:
        return {}
    # Try JSON first
    try:
        parsed = json.loads(args_str)
        if isinstance(parsed, dict):
            return parsed
        # json.loads can return int/float/list/str/bool — wrap them
        return {"arg_0": parsed}
    except (json.JSONDecodeError, ValueError):
        pass
    # Try key=value pairs
    args = {}
    for part in args_str.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip().strip("'\"")
        else:
            args[f"arg_{len(args)}"] = part
    return args


def _normalize_args(args: dict | str | Any) -> dict:
    """Normalize arguments to a comparable dict."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"arg_0": parsed}
        except (json.JSONDecodeError, ValueError):
            return {"raw": args}
    # int, float, list, bool — wrap so .keys() never fails
    return {"arg_0": args}


def score_tool_name_match(
    predicted_calls: list[dict],
    reference_calls: list[dict],
) -> float:
    """Score: fraction of reference tool names that appear in predictions."""
    if not reference_calls:
        # No tools expected — reward if model also didn't call tools
        return 1.0 if not predicted_calls else 0.5

    ref_names = [c.get("function", {}).get("name", c.get("name", "")) for c in reference_calls]
    pred_names = [c.get("name", "") for c in predicted_calls]

    if not ref_names:
        return 1.0

    matched = sum(1 for rn in ref_names if rn in pred_names)
    return matched / len(ref_names)


def score_tool_args_match(
    predicted_calls: list[dict],
    reference_calls: list[dict],
) -> float:
    """Score: how well predicted arguments match reference arguments."""
    if not reference_calls:
        return 1.0

    total_score = 0.0
    matched_refs = 0

    for ref_call in reference_calls:
        ref_fn = ref_call.get("function", ref_call)
        ref_name = ref_fn.get("name", "")
        ref_args = _normalize_args(ref_fn.get("arguments", {}))

        # Find matching predicted call
        best_score = 0.0
        for pred_call in predicted_calls:
            if pred_call.get("name", "") != ref_name:
                continue

            pred_args = _normalize_args(pred_call.get("arguments", {}))

            if not ref_args and not pred_args:
                best_score = 1.0
                break

            # Key overlap
            ref_keys = set(ref_args.keys())
            pred_keys = set(pred_args.keys())
            if not ref_keys:
                best_score = 1.0
                break

            key_overlap = len(ref_keys & pred_keys) / len(ref_keys)

            # Value match for overlapping keys
            val_matches = 0
            common = ref_keys & pred_keys
            for k in common:
                if str(ref_args[k]).lower().strip() == str(pred_args[k]).lower().strip():
                    val_matches += 1

            val_score = val_matches / max(len(common), 1)
            best_score = max(best_score, 0.5 * key_overlap + 0.5 * val_score)

        total_score += best_score
        matched_refs += 1

    return total_score / max(matched_refs, 1)


def score_response_overlap(predicted: str, reference: str) -> float:
    """Token-level overlap between predicted and reference response."""
    if not reference:
        return 0.5  # neutral if no reference text
    if not predicted:
        return 0.0

    # Simple word overlap (case-insensitive)
    pred_tokens = set(predicted.lower().split())
    ref_tokens = set(reference.lower().split())

    if not ref_tokens:
        return 0.5

    overlap = len(pred_tokens & ref_tokens)
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    # F1
    return 2 * precision * recall / (precision + recall)


def compute_accuracy(
    generated_text: str,
    reference_response: str,
    reference_tool_calls: list[dict],
    tool_weight: float = 0.6,
    response_weight: float = 0.4,
) -> AccuracyResult:
    """Compute accuracy score for a single generation.

    Args:
        generated_text: model output
        reference_response: ground truth assistant response
        reference_tool_calls: ground truth tool calls from dataset
        tool_weight: weight for tool matching in final score
        response_weight: weight for response overlap in final score

    Returns:
        AccuracyResult with overall score and component scores
    """
    # Extract predicted tool calls
    pred_calls = extract_tool_calls_from_text(generated_text)

    # Score components
    name_match = score_tool_name_match(pred_calls, reference_tool_calls)
    args_match = score_tool_args_match(pred_calls, reference_tool_calls)
    resp_overlap = score_response_overlap(generated_text, reference_response)

    # Tool score is average of name and args matching
    tool_score = 0.5 * name_match + 0.5 * args_match

    # Weighted combination
    if reference_tool_calls:
        # When tools are expected, weight tool matching higher
        overall = tool_weight * tool_score + response_weight * resp_overlap
    else:
        # No tools expected — rely more on response
        overall = 0.3 * tool_score + 0.7 * resp_overlap

    return AccuracyResult(
        score=overall,
        tool_name_match=name_match,
        tool_args_match=args_match,
        response_overlap=resp_overlap,
    )
