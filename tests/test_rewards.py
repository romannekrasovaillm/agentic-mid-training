"""Tests for decomposed reward and baselines."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.rewards.decomposed_reward import DecomposedReward, FormatChecker
from entropy_reward.rewards.baselines import GroupNormBaseline, LeaveOneOutBaseline, JackknifeBaseline


class TestFormatChecker:
    def test_strict_pass(self):
        text = "<think>I need to solve this.</think>\n<answer>42</answer>"
        assert FormatChecker.strict_check(text) == 1.0

    def test_strict_fail_no_think(self):
        text = "<answer>42</answer>"
        assert FormatChecker.strict_check(text) == 0.0

    def test_strict_fail_no_answer(self):
        text = "<think>thinking</think>\nno answer tag"
        assert FormatChecker.strict_check(text) == 0.0

    def test_partial_full_credit(self):
        text = "<think>reasoning</think>\n<answer>42</answer>"
        assert FormatChecker.partial_check(text) == 1.0

    def test_partial_structure_credit(self):
        text = '<think>reasoning</think>\n{"result": 42}'
        score = FormatChecker.partial_check(text)
        assert 0.0 < score < 1.0

    def test_partial_tag_credit(self):
        text = "<think>only thinking tag</think>\nno structure"
        score = FormatChecker.partial_check(text)
        assert score > 0.0


class TestDecomposedReward:
    def test_strict_mode(self):
        reward = DecomposedReward(format_mode="strict")
        text = "<think>reason</think>\n<answer>42</answer>"
        result = reward.compute(text, accuracy_score=0.5)
        assert result.r_format == 1.0
        assert result.r_acc == 0.5

    def test_partial_mode(self):
        reward = DecomposedReward(format_mode="partial")
        text = "<think>reason</think>"
        result = reward.compute(text, accuracy_score=0.0)
        assert result.r_format > 0.0
        assert result.r_format < 1.0

    def test_batch(self):
        reward = DecomposedReward()
        texts = ["<think>a</think><answer>b</answer>", "no format"]
        results = reward.compute_batch(texts, [1.0, 0.0])
        assert len(results) == 2
        assert results[0].r_format > results[1].r_format


class TestGoldenReward:
    """Golden examples: verify reward functions give non-zero scores on correctly formatted outputs.

    These tests catch spec-mismatches where R_fmt=0 / R_tool=0 even when the
    output is perfectly formatted.  See the Nemotron-Agentic-v1 reference
    format: <think>reasoning</think>  <action>tool_name(args)</action>
    """

    # ── Perfect tool-call example ──────────────────────────────

    GOLDEN_TOOL_CALL = (
        "<think>The user wants the current weather in London. "
        "I should call the get_weather function.</think>\n"
        "<action>get_weather(city=\"London\", units=\"metric\")</action>"
    )

    GOLDEN_ANSWER = (
        "<think>The user asked for the capital of France. "
        "I know the answer without needing any tool.</think>\n"
        "<answer>The capital of France is Paris.</answer>"
    )

    GOLDEN_MULTI_TOOL = (
        "<think>I need to search for flights and then book one.</think>\n"
        "<action>search_flights(origin=\"SVO\", dest=\"LED\", date=\"2025-07-01\")</action>\n"
        "Connecting to flight service...\n"
        "<action>book_flight(flight_id=\"SU42\", passenger=\"Ivan\")</action>"
    )

    # ── Realistic BAD model outputs (no tags) ──────────────────

    BAD_NO_TAGS = (
        "Sure! Let me check the weather for you.\n"
        "The weather in London is 15°C with light rain."
    )

    BAD_WRONG_TAGS = (
        "```json\n{\"tool\": \"get_weather\", \"args\": {\"city\": \"London\"}}\n```"
    )

    BAD_PARTIAL_THINK_ONLY = (
        "<think>I should call get_weather for London.</think>\n"
        "get_weather(city='London')"
    )

    # ── R_fmt tests ────────────────────────────────────────────

    def test_golden_tool_call_format(self):
        """Perfect tool-call must score R_fmt=1.0 (strict)."""
        assert FormatChecker.strict_check(self.GOLDEN_TOOL_CALL) == 1.0

    def test_golden_answer_format(self):
        """Perfect answer must score R_fmt=1.0 (strict)."""
        assert FormatChecker.strict_check(self.GOLDEN_ANSWER) == 1.0

    def test_golden_multi_tool_format(self):
        """Multiple tool calls still give R_fmt=1.0."""
        assert FormatChecker.strict_check(self.GOLDEN_MULTI_TOOL) == 1.0

    def test_bad_no_tags_format(self):
        """Bare text with no tags → R_fmt=0."""
        assert FormatChecker.strict_check(self.BAD_NO_TAGS) == 0.0

    def test_bad_wrong_tags_format(self):
        """JSON code block (no <think>/<action>) → R_fmt=0."""
        assert FormatChecker.strict_check(self.BAD_WRONG_TAGS) == 0.0

    def test_bad_partial_think_only_format(self):
        """<think> present but no <action>/<answer> → R_fmt=0."""
        assert FormatChecker.strict_check(self.BAD_PARTIAL_THINK_ONLY) == 0.0

    # ── R_tool tests ───────────────────────────────────────────

    def test_golden_tool_call_tool_score(self):
        """Valid <action>tool(</action> pattern → R_tool=1.0."""
        from entropy_reward.rewards.decomposed_reward import ToolUseChecker
        assert ToolUseChecker.check(self.GOLDEN_TOOL_CALL) == 1.0

    def test_golden_tool_call_with_available_tools(self):
        """Tool name matches available list → R_tool=1.0."""
        from entropy_reward.rewards.decomposed_reward import ToolUseChecker
        assert ToolUseChecker.check(
            self.GOLDEN_TOOL_CALL, available_tools=["get_weather", "book_hotel"]
        ) == 1.0

    def test_golden_tool_call_wrong_available_tools(self):
        """Tool name NOT in available list → R_tool < 1.0."""
        from entropy_reward.rewards.decomposed_reward import ToolUseChecker
        score = ToolUseChecker.check(
            self.GOLDEN_TOOL_CALL, available_tools=["book_hotel", "search_flights"]
        )
        assert score == 0.0

    def test_golden_multi_tool_count(self):
        """Two <action> tags → count_calls=2."""
        from entropy_reward.rewards.decomposed_reward import ToolUseChecker
        assert ToolUseChecker.count_calls(self.GOLDEN_MULTI_TOOL) == 2

    def test_bad_no_tags_tool_score(self):
        """No <action> → R_tool=0."""
        from entropy_reward.rewards.decomposed_reward import ToolUseChecker
        assert ToolUseChecker.check(self.BAD_NO_TAGS) == 0.0

    def test_bad_partial_think_tool_score(self):
        """Tool call without <action> tags → R_tool=0."""
        from entropy_reward.rewards.decomposed_reward import ToolUseChecker
        assert ToolUseChecker.check(self.BAD_PARTIAL_THINK_ONLY) == 0.0

    # ── End-to-end DecomposedReward on golden examples ─────────

    def test_golden_tool_call_full_reward(self):
        """Full pipeline: golden tool-call → R_fmt=1, R_tool=1."""
        reward = DecomposedReward(format_mode="strict")
        result = reward.compute(self.GOLDEN_TOOL_CALL, accuracy_score=0.0)
        assert result.r_format == 1.0, f"R_fmt should be 1.0, got {result.r_format}"
        assert result.r_tool == 1.0, f"R_tool should be 1.0, got {result.r_tool}"
        assert result.r_total >= 2.0, f"R_total should be >=2.0, got {result.r_total}"

    def test_golden_answer_full_reward(self):
        """Full pipeline: golden answer → R_fmt=1, R_tool=0 (no tool expected)."""
        reward = DecomposedReward(format_mode="strict")
        result = reward.compute(self.GOLDEN_ANSWER, accuracy_score=0.0)
        assert result.r_format == 1.0
        assert result.r_tool == 0.0  # no tool call in answer — that's correct

    def test_bad_output_full_reward(self):
        """Full pipeline: bare text → R_fmt=0, R_tool=0."""
        reward = DecomposedReward(format_mode="strict")
        result = reward.compute(self.BAD_NO_TAGS, accuracy_score=0.0)
        assert result.r_format == 0.0
        assert result.r_tool == 0.0
        assert result.r_total == 0.0

    def test_partial_mode_gives_credit(self):
        """Partial mode: <think> only → partial credit > 0."""
        reward = DecomposedReward(format_mode="partial")
        result = reward.compute(self.BAD_PARTIAL_THINK_ONLY, accuracy_score=0.0)
        assert result.r_format > 0.0, "Partial mode should give credit for <think> tag"
        assert result.r_format < 1.0, "But not full credit without <action>/<answer>"

    # ── Diagnostic logging ─────────────────────────────────────

    def test_diagnose_golden(self):
        """diagnose() returns non-empty info for golden example."""
        reward = DecomposedReward(format_mode="strict")
        result, diag = reward.compute_with_diagnosis(self.GOLDEN_TOOL_CALL)
        assert result.r_format == 1.0
        assert diag["has_think"] is True
        assert diag["has_action"] is True

    def test_diagnose_bad(self):
        """diagnose() explains exactly what's missing."""
        reward = DecomposedReward(format_mode="strict")
        result, diag = reward.compute_with_diagnosis(self.BAD_NO_TAGS)
        assert result.r_format == 0.0
        assert diag["has_think"] is False
        assert diag["has_action"] is False
        assert diag["has_answer"] is False
        assert "reason" in diag  # human-readable reason string


class TestMultiplicativeReward:
    """Tests for multiplicative format gating with floor."""

    GOOD = "<think>reasoning</think>\n<action>get_weather(city=\"London\")</action>"
    BAD = "Sure! The weather in London is 15°C."

    def test_multiplicative_perfect_format(self):
        """r_fmt=1 → multiplier=1.0 → full base reward."""
        reward = DecomposedReward(
            format_mode="strict", multiplicative_format=True, format_floor=0.1,
        )
        result = reward.compute(self.GOOD, accuracy_score=1.0)
        assert result.r_format == 1.0
        # multiplier = 0.1 + 0.9 * 1.0 = 1.0
        # base = 1.0 * 1.0 (tool) + 1.0 * 1.0 (acc) = 2.0
        assert result.r_total == pytest.approx(2.0)

    def test_multiplicative_zero_format(self):
        """r_fmt=0 → multiplier=floor → 10% of base reward."""
        reward = DecomposedReward(
            format_mode="strict", multiplicative_format=True, format_floor=0.1,
        )
        result = reward.compute(self.BAD, accuracy_score=1.0)
        assert result.r_format == 0.0
        # multiplier = 0.1 + 0.9 * 0.0 = 0.1
        # base = 0.0 (tool) + 1.0 (acc) = 1.0
        assert result.r_total == pytest.approx(0.1)

    def test_multiplicative_preserves_gradient_signal(self):
        """Even with r_fmt=0, r_total > 0 when base > 0 (gradient signal preserved)."""
        reward = DecomposedReward(
            format_mode="strict", multiplicative_format=True, format_floor=0.1,
        )
        result = reward.compute(self.BAD, accuracy_score=0.5)
        assert result.r_total > 0.0, "floor should preserve nonzero reward"

    def test_multiplicative_vs_additive(self):
        """Multiplicative penalises bad format harder than additive."""
        mult = DecomposedReward(
            format_mode="strict", multiplicative_format=True, format_floor=0.1,
        )
        add = DecomposedReward(
            format_mode="strict", multiplicative_format=False,
        )
        r_mult = mult.compute(self.BAD, accuracy_score=1.0)
        r_add = add.compute(self.BAD, accuracy_score=1.0)
        # Both have r_fmt=0, r_tool=0, r_acc=1.0
        # Additive: 0 + 0 + 1.0 = 1.0
        # Multiplicative: 0.1 * (0 + 1.0) = 0.1
        assert r_mult.r_total < r_add.r_total

    def test_additive_mode_unchanged(self):
        """multiplicative_format=False keeps the old additive behaviour."""
        reward = DecomposedReward(
            format_mode="strict", multiplicative_format=False,
        )
        result = reward.compute(self.GOOD, accuracy_score=0.5)
        # additive: 1.0*1.0 + 1.0*1.0 + 1.0*0.5 = 2.5
        assert result.r_total == pytest.approx(2.5)


class TestBaselines:
    def test_group_norm(self):
        bl = GroupNormBaseline()
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = bl.compute_advantages(rewards, group_size=4)
        assert adv.shape == (4,)
        assert abs(adv.mean().item()) < 0.01  # should be zero-centered

    def test_loo(self):
        bl = LeaveOneOutBaseline()
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = bl.compute_advantages(rewards, group_size=4)
        assert adv.shape == (4,)

    def test_jackknife(self):
        bl = JackknifeBaseline()
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = bl.compute_advantages(rewards, group_size=4)
        assert adv.shape == (4,)

    def test_separate_baselines(self):
        bl = GroupNormBaseline(separate=True)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        components = {
            "format": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "tool": torch.tensor([0.5, 0.5, 1.0, 1.0]),
            "acc": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        }
        adv = bl.compute_advantages(rewards, group_size=4, reward_components=components)
        assert adv.shape == (4,)
