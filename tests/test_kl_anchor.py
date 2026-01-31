"""Tests for KL anchoring."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.strategies.kl_anchor import KLAnchor


class TestKLAnchor:
    def test_warmup_ramp(self):
        kl = KLAnchor(initial_coeff=0.2, warmup_steps=100)
        assert kl.get_coeff(0) == 0.0
        assert kl.get_coeff(50) == pytest.approx(0.1, abs=0.01)
        assert kl.get_coeff(100) == pytest.approx(0.2, abs=0.01)

    def test_linear_decay(self):
        kl = KLAnchor(initial_coeff=0.2, final_coeff=0.02, schedule="linear",
                       warmup_steps=0, total_steps=1000)
        c_start = kl.get_coeff(0)
        c_mid = kl.get_coeff(500)
        c_end = kl.get_coeff(1000)
        assert c_start == pytest.approx(0.2, abs=0.01)
        assert c_mid < c_start
        assert c_end == pytest.approx(0.02, abs=0.01)

    def test_cosine_decay(self):
        kl = KLAnchor(initial_coeff=0.2, final_coeff=0.02, schedule="cosine",
                       warmup_steps=0, total_steps=1000)
        c_start = kl.get_coeff(0)
        c_end = kl.get_coeff(1000)
        assert c_start == pytest.approx(0.2, abs=0.01)
        assert c_end == pytest.approx(0.02, abs=0.01)

    def test_step_decay(self):
        kl = KLAnchor(initial_coeff=0.2, final_coeff=0.01, schedule="step",
                       warmup_steps=0, total_steps=1000,
                       step_milestones=[0.5], step_gamma=0.5)
        c_before = kl.get_coeff(400)
        c_after = kl.get_coeff(600)
        assert c_before > c_after

    def test_compute_returns_penalty(self):
        kl = KLAnchor(initial_coeff=0.1, warmup_steps=0)
        current = torch.randn(2, 5, 100)
        ref = torch.randn(2, 5, 100)
        penalty, kl_raw, coeff = kl.compute(current, ref)
        assert penalty.dim() == 0
        assert penalty.item() >= 0
        assert coeff > 0

    def test_identical_logits_zero_kl(self):
        kl = KLAnchor(initial_coeff=0.1, warmup_steps=0)
        logits = torch.randn(2, 5, 100)
        penalty, kl_raw, _ = kl.compute(logits, logits.clone())
        assert kl_raw.item() < 0.01
