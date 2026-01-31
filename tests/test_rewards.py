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
