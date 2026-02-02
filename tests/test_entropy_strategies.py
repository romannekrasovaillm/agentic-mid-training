"""Tests for entropy bonus strategies."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.strategies.entropy_bonus import ConstantEntropyBonus, AdaptiveEntropyBonus


class TestConstantEntropyBonus:
    def test_compute_returns_scalar(self):
        bonus = ConstantEntropyBonus(coeff=0.01)
        logits = torch.randn(4, 10, 100)
        result = bonus.compute(logits)
        assert result.dim() == 0

    def test_higher_coeff_higher_bonus(self):
        logits = torch.randn(4, 10, 100)
        low = ConstantEntropyBonus(coeff=0.01).compute(logits)
        high = ConstantEntropyBonus(coeff=0.1).compute(logits)
        assert high > low

    def test_uniform_logits_max_entropy(self):
        # Uniform distribution has maximum entropy
        logits = torch.zeros(4, 10, 50)
        bonus = ConstantEntropyBonus(coeff=1.0)
        result = bonus.compute(logits)
        # Max entropy for 50 classes ≈ ln(50) ≈ 3.91
        assert result.item() > 3.0

    def test_state_dict(self):
        bonus = ConstantEntropyBonus(coeff=0.05)
        state = bonus.state_dict()
        assert state["coeff"] == 0.05


class TestAdaptiveEntropyBonus:
    def test_initial_compute(self):
        bonus = AdaptiveEntropyBonus(min_coeff=0.001, max_coeff=0.1)
        logits = torch.randn(4, 10, 100)
        result = bonus.compute(logits)
        assert result.dim() == 0

    def test_coeff_changes_on_entropy_drop(self):
        bonus = AdaptiveEntropyBonus(
            min_coeff=0.001,
            max_coeff=0.5,
            entropy_drop_threshold=0.1,
            alpha=0.5,
        )
        # First call: high entropy (uniform)
        logits_high = torch.zeros(4, 10, 50)
        bonus.compute(logits_high)
        coeff_after_init = bonus.get_coeff()

        # Simulate entropy drop with peaked logits
        logits_low = torch.zeros(4, 10, 50)
        logits_low[:, :, 0] = 100.0
        for _ in range(20):
            bonus.compute(logits_low)

        coeff_after_drop = bonus.get_coeff()
        # Coefficient should increase to fight collapse
        assert coeff_after_drop >= coeff_after_init or bonus._entropy_ema < bonus._initial_entropy

    def test_diversity_signal(self):
        bonus = AdaptiveEntropyBonus()
        logits = torch.randn(4, 10, 100)
        bonus.compute(logits)
        bonus.update_diversity_signal(0.8)
        assert bonus._diversity_ema is not None

    def test_state_roundtrip(self):
        bonus = AdaptiveEntropyBonus()
        logits = torch.randn(4, 10, 100)
        bonus.compute(logits)
        state = bonus.state_dict()
        bonus2 = AdaptiveEntropyBonus()
        bonus2.load_state_dict(state)
        assert bonus2._coeff == bonus._coeff
