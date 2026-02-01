"""Tests for Nemotron-Agentic-v1 data loader and accuracy scorer."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.data import (
    AgenticSample,
    FORMAT_INSTRUCTION,
    _build_prompt,
    _build_multiturn_prompt,
    _extract_tool_names,
    _extract_reference,
)
from entropy_reward.data.accuracy import (
    extract_tool_calls_from_text,
    compute_accuracy,
    score_tool_name_match,
    score_tool_args_match,
    score_response_overlap,
)


# --- Fixtures ---

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Create a new food delivery order",
            "parameters": {"properties": {"user_id": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an existing order",
            "parameters": {"properties": {"order_id": {"type": "string"}}},
        },
    },
]

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a food delivery assistant."},
    {"role": "user", "content": "I want to order pizza."},
    {
        "role": "assistant",
        "content": "I'll place that order for you.",
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {
                    "name": "place_order",
                    "arguments": '{"user_id": "u123", "restaurants": []}',
                },
            }
        ],
        "reasoning_content": "The user wants to order pizza. I should use place_order.",
    },
    {"role": "tool", "content": '{"status": "success", "order_id": "ord_456"}', "tool_call_id": "call_1"},
    {"role": "assistant", "content": "Your order has been placed! Order ID: ord_456."},
]


# --- Data loader tests ---


class TestToolNameExtraction:
    def test_extract_names(self):
        names = _extract_tool_names(SAMPLE_TOOLS)
        assert names == ["place_order", "cancel_order"]

    def test_empty(self):
        assert _extract_tool_names([]) == []


class TestBuildPrompt:
    def test_basic_prompt(self):
        prompt = _build_prompt(SAMPLE_MESSAGES, SAMPLE_TOOLS)
        assert "[System]" in prompt
        assert "food delivery assistant" in prompt
        assert "[Available Tools]" in prompt
        assert "place_order" in prompt
        assert "[User]" in prompt
        assert "order pizza" in prompt

    def test_no_system(self):
        msgs = [{"role": "user", "content": "Hello"}]
        prompt = _build_prompt(msgs, [])
        assert "[User]" in prompt
        assert "Hello" in prompt

    def test_format_instruction_present(self):
        """Prompt must contain format instructions so the model knows about
        <think>/<action>/<answer> tags — otherwise R_fmt=0 forever."""
        prompt = _build_prompt(SAMPLE_MESSAGES, SAMPLE_TOOLS)
        assert "[Response Format]" in prompt
        assert "<think>" in prompt
        assert "<action>" in prompt
        assert "<answer>" in prompt

    def test_format_instruction_in_multiturn(self):
        prompt = _build_multiturn_prompt(SAMPLE_MESSAGES, SAMPLE_TOOLS, max_turns=2)
        assert "[Response Format]" in prompt
        assert "<think>" in prompt


class TestMultiturnPrompt:
    def test_includes_context(self):
        prompt = _build_multiturn_prompt(SAMPLE_MESSAGES, SAMPLE_TOOLS, max_turns=2)
        assert "[System]" in prompt
        assert "[User]" in prompt

    def test_limits_turns(self):
        prompt = _build_multiturn_prompt(SAMPLE_MESSAGES, SAMPLE_TOOLS, max_turns=1)
        assert "order pizza" in prompt


class TestExtractReference:
    def test_extract_first_assistant(self):
        response, tool_calls = _extract_reference(SAMPLE_MESSAGES)
        assert "place that order" in response
        assert "<think>" in response  # reasoning_content wrapped
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "place_order"


# --- Accuracy scorer tests ---


class TestExtractToolCalls:
    def test_action_format(self):
        text = "<action>place_order(user_id=u123)</action>"
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "place_order"

    def test_multiple_actions(self):
        text = (
            "<action>search(query=pizza)</action>\n"
            "<action>place_order(item=margherita)</action>"
        )
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 2

    def test_no_tools(self):
        text = "Just a plain response with no tool calls."
        calls = extract_tool_calls_from_text(text)
        assert len(calls) == 0


class TestScoreToolNameMatch:
    def test_exact_match(self):
        pred = [{"name": "place_order"}]
        ref = [{"function": {"name": "place_order"}}]
        assert score_tool_name_match(pred, ref) == 1.0

    def test_no_match(self):
        pred = [{"name": "cancel_order"}]
        ref = [{"function": {"name": "place_order"}}]
        assert score_tool_name_match(pred, ref) == 0.0

    def test_no_reference(self):
        # No tools expected, model didn't call any
        assert score_tool_name_match([], []) == 1.0
        # No tools expected but model called one
        assert score_tool_name_match([{"name": "foo"}], []) == 0.5


class TestScoreResponseOverlap:
    def test_identical(self):
        score = score_response_overlap("hello world", "hello world")
        assert score == 1.0

    def test_partial(self):
        score = score_response_overlap("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        score = score_response_overlap("abc", "xyz")
        assert score == 0.0

    def test_empty_predicted(self):
        assert score_response_overlap("", "hello") == 0.0


class TestComputeAccuracy:
    def test_with_tool_calls(self):
        generated = "<think>Order pizza</think>\n<action>place_order(user_id=u123)</action>"
        ref_response = "I'll place the order."
        ref_calls = [{"function": {"name": "place_order", "arguments": '{"user_id": "u123"}'}}]

        result = compute_accuracy(generated, ref_response, ref_calls)
        assert result.score > 0.0
        assert result.tool_name_match == 1.0

    def test_without_tool_calls(self):
        generated = "The answer is 42."
        result = compute_accuracy(generated, "The answer is 42.", [])
        assert result.score > 0.5

    def test_wrong_tool(self):
        generated = "<action>cancel_order(id=x)</action>"
        ref_calls = [{"function": {"name": "place_order", "arguments": "{}"}}]
        result = compute_accuracy(generated, "", ref_calls)
        assert result.tool_name_match == 0.0
