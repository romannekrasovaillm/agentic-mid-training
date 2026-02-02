"""Tests for stop conditions."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.monitoring.stop_conditions import (
    CollapseDetector,
    HackingDetector,
    AdvantageDriftDetector,
    StopConditionAggregator,
    StopReason,
)


class TestCollapseDetector:
    def test_no_collapse(self):
        cd = CollapseDetector(entropy_threshold=0.5, window=5)
        for i in range(10):
            signal = cd.update(entropy=3.0, diversity=0.8)
        assert not signal.should_stop

    def test_entropy_collapse(self):
        cd = CollapseDetector(entropy_threshold=0.3, window=5)
        # Initialize with high entropy
        cd.update(entropy=3.0, diversity=0.8)
        # Then drop
        for _ in range(10):
            signal = cd.update(entropy=1.0, diversity=0.8)
        assert signal.should_stop
        assert signal.reason == StopReason.ENTROPY_COLLAPSE

    def test_diversity_collapse(self):
        cd = CollapseDetector(diversity_threshold=0.5, window=5)
        cd.update(entropy=3.0, diversity=0.8)
        for _ in range(10):
            signal = cd.update(entropy=3.0, diversity=0.2)
        assert signal.should_stop
        assert signal.reason == StopReason.DIVERSITY_COLLAPSE


class TestHackingDetector:
    def test_no_hacking(self):
        hd = HackingDetector(passrate_threshold=0.8)
        signal = hd.update(0.3)
        assert not signal.should_stop

    def test_hacking_detected(self):
        hd = HackingDetector(passrate_threshold=0.5)
        signal = hd.update(0.9)
        assert signal.should_stop
        assert signal.reason == StopReason.REWARD_HACKING


class TestAdvantageDriftDetector:
    def test_no_drift(self):
        dd = AdvantageDriftDetector(drift_threshold=3.0, window=10)
        for _ in range(20):
            signal = dd.update(0.0, 1.0)
        assert not signal.should_stop

    def test_mean_drift(self):
        dd = AdvantageDriftDetector(drift_threshold=1.0, window=20)
        # First half: low mean
        for i in range(10):
            dd.update(0.0, 1.0)
        # Second half: large jump to create clear drift
        for i in range(15):
            signal = dd.update(50.0, 1.0)
        assert signal.should_stop


class TestAggregator:
    def test_no_stop(self):
        agg = StopConditionAggregator()
        signal = agg.check(entropy=3.0, diversity=0.8, adv_mean=0.0, adv_std=1.0)
        assert not signal.should_stop

    def test_passes_through_collapse(self):
        agg = StopConditionAggregator(
            collapse=CollapseDetector(entropy_threshold=0.3, window=3),
        )
        agg.check(entropy=3.0, diversity=0.8)
        for _ in range(5):
            signal = agg.check(entropy=1.0, diversity=0.8)
        assert signal.should_stop
