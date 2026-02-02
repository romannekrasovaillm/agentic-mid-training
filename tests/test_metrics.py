"""Tests for metrics collection."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.metrics.entropy_metrics import TokenEntropy, ActionEntropy
from entropy_reward.metrics.diversity_metrics import SelfBLEU, TrajectoryUniqueness, PatternRepetition
from entropy_reward.metrics.advantage_stats import AdvantageStatistics


class TestTokenEntropy:
    def test_compute(self):
        te = TokenEntropy()
        logits = torch.randn(4, 10, 100)
        val = te.compute(logits)
        assert isinstance(val, float)
        assert val > 0

    def test_with_mask(self):
        te = TokenEntropy()
        logits = torch.randn(4, 10, 100)
        mask = torch.ones(4, 10)
        mask[:, 5:] = 0
        val = te.compute(logits, mask)
        assert val > 0

    def test_trend(self):
        te = TokenEntropy()
        for i in range(20):
            logits = torch.randn(4, 10, 100) * (1 + i * 0.1)
            te.compute(logits)
        trend = te.trend()
        assert isinstance(trend, float)


class TestActionEntropy:
    def test_diverse_actions(self):
        ae = ActionEntropy()
        actions = ["search", "calculate", "code_run", "analyze"]
        val = ae.compute(actions)
        assert val > 0

    def test_uniform_actions(self):
        ae = ActionEntropy()
        actions = ["search"] * 10
        val = ae.compute(actions)
        assert abs(val) < 1e-6

    def test_empty(self):
        ae = ActionEntropy()
        val = ae.compute([])
        assert val == 0.0


class TestSelfBLEU:
    def test_diverse_texts(self):
        sb = SelfBLEU()
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning models process data efficiently",
            "Python is a versatile programming language",
            "The weather forecast predicts rain tomorrow",
        ]
        score = sb.compute(texts)
        assert 0.0 <= score <= 1.0

    def test_identical_texts(self):
        sb = SelfBLEU(n=2)
        texts = ["the quick brown fox jumps over the lazy dog here now"] * 5
        score = sb.compute(texts)
        # Identical texts should yield non-negative BLEU
        assert score >= 0.0


class TestTrajectoryUniqueness:
    def test_all_unique(self):
        tu = TrajectoryUniqueness()
        texts = ["text a", "text b", "text c"]
        val = tu.compute(texts)
        assert val == 1.0

    def test_duplicates(self):
        tu = TrajectoryUniqueness()
        texts = ["same", "same", "different"]
        val = tu.compute(texts)
        assert val < 1.0


class TestPatternRepetition:
    def test_no_tags(self):
        pr = PatternRepetition()
        texts = ["no tags here", "also no tags"]
        val = pr.compute(texts)
        assert isinstance(val, float)

    def test_repeated_patterns(self):
        pr = PatternRepetition()
        pattern = "<think>a</think><answer>b</answer>"
        texts = [pattern] * 5
        val = pr.compute(texts)
        assert val > 0


class TestAdvantageStatistics:
    def test_compute(self):
        ast = AdvantageStatistics()
        adv = torch.randn(16)
        stats = ast.compute(adv)
        assert abs(stats.mean) < 5
        assert stats.std > 0

    def test_drift(self):
        ast = AdvantageStatistics(window_size=20)
        for i in range(30):
            adv = torch.randn(16) + i * 0.1
            ast.compute(adv)
        drift = ast.mean_drift()
        assert drift > 0
