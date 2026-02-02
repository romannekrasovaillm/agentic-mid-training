"""Stop conditions: detect collapse, reward hacking, and advantage drift.

Three independent detectors that can halt training early:
1. CollapseDetector: entropy↓ + diversity↓
2. HackingDetector: exploit pass-rate↑
3. AdvantageDriftDetector: advantage statistics drifting
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class StopReason(Enum):
    NONE = "none"
    ENTROPY_COLLAPSE = "entropy_collapse"
    DIVERSITY_COLLAPSE = "diversity_collapse"
    REWARD_HACKING = "reward_hacking"
    ADVANTAGE_DRIFT = "advantage_drift"


@dataclass
class StopSignal:
    should_stop: bool = False
    reason: StopReason = StopReason.NONE
    details: str = ""
    severity: float = 0.0  # 0-1, 1=critical


class CollapseDetector:
    """Detect entropy and diversity collapse.

    Triggers when:
    - Token entropy drops below threshold relative to initial (sustained over window)
    - Diversity metric (1 - self_BLEU) drops below threshold
    """

    def __init__(
        self,
        entropy_threshold: float = 0.3,
        diversity_threshold: float = 0.4,
        window: int = 50,
    ):
        self.entropy_threshold = entropy_threshold
        self.diversity_threshold = diversity_threshold
        self.window = window
        self._entropy_history: list[float] = []
        self._diversity_history: list[float] = []
        self._initial_entropy: float | None = None

    def update(self, entropy: float, diversity: float) -> StopSignal:
        """Check for collapse signals.

        Args:
            entropy: current token entropy
            diversity: 1 - self_BLEU (higher = more diverse)
        """
        if self._initial_entropy is None:
            self._initial_entropy = entropy

        self._entropy_history.append(entropy)
        self._diversity_history.append(diversity)

        # Check entropy collapse
        if len(self._entropy_history) >= self.window:
            recent = self._entropy_history[-self.window :]
            avg_recent = np.mean(recent)
            relative_drop = 1.0 - avg_recent / (self._initial_entropy + 1e-8)

            if relative_drop > self.entropy_threshold:
                return StopSignal(
                    should_stop=True,
                    reason=StopReason.ENTROPY_COLLAPSE,
                    details=(
                        f"Entropy dropped {relative_drop:.1%} from initial "
                        f"({self._initial_entropy:.4f} -> {avg_recent:.4f})"
                    ),
                    severity=min(relative_drop / self.entropy_threshold, 1.0),
                )

        # Check diversity collapse
        if len(self._diversity_history) >= self.window:
            recent_div = self._diversity_history[-self.window :]
            avg_div = np.mean(recent_div)

            if avg_div < self.diversity_threshold:
                return StopSignal(
                    should_stop=True,
                    reason=StopReason.DIVERSITY_COLLAPSE,
                    details=f"Diversity dropped to {avg_div:.4f} (threshold: {self.diversity_threshold})",
                    severity=1.0 - avg_div / self.diversity_threshold,
                )

        return StopSignal()


class HackingDetector:
    """Detect reward hacking via red-team exploit pass rate.

    Triggers when exploit success rate exceeds threshold.
    """

    def __init__(
        self,
        passrate_threshold: float = 0.8,
        eval_interval: int = 200,
    ):
        self.passrate_threshold = passrate_threshold
        self.eval_interval = eval_interval
        self._exploit_rates: list[float] = []

    def update(self, exploit_pass_rate: float) -> StopSignal:
        """Check for reward hacking.

        Args:
            exploit_pass_rate: fraction of exploits that got high reward
        """
        self._exploit_rates.append(exploit_pass_rate)

        if exploit_pass_rate > self.passrate_threshold:
            return StopSignal(
                should_stop=True,
                reason=StopReason.REWARD_HACKING,
                details=(
                    f"Exploit pass rate {exploit_pass_rate:.1%} exceeds "
                    f"threshold {self.passrate_threshold:.1%}"
                ),
                severity=min(exploit_pass_rate / self.passrate_threshold, 1.0),
            )

        # Check trend: increasing exploit rate
        if len(self._exploit_rates) >= 3:
            recent = self._exploit_rates[-3:]
            if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                trend_rate = recent[-1]
                if trend_rate > self.passrate_threshold * 0.7:
                    return StopSignal(
                        should_stop=False,  # warning, not stop
                        reason=StopReason.REWARD_HACKING,
                        details=f"Exploit rate trending up: {[f'{r:.1%}' for r in recent]}",
                        severity=trend_rate / self.passrate_threshold,
                    )

        return StopSignal()


class AdvantageDriftDetector:
    """Detect advantage statistics drift.

    Triggers when advantage mean or std drifts beyond thresholds,
    indicating unstable training dynamics.
    """

    def __init__(
        self,
        drift_threshold: float = 2.0,
        window: int = 100,
    ):
        self.drift_threshold = drift_threshold
        self.window = window
        self._means: list[float] = []
        self._stds: list[float] = []

    def update(self, adv_mean: float, adv_std: float) -> StopSignal:
        self._means.append(adv_mean)
        self._stds.append(adv_std)

        if len(self._means) < self.window:
            return StopSignal()

        recent_means = self._means[-self.window :]
        recent_stds = self._stds[-self.window :]

        # Check mean drift
        first_half_mean = np.mean(recent_means[: self.window // 2])
        second_half_mean = np.mean(recent_means[self.window // 2 :])
        overall_std = np.std(recent_means)

        if overall_std > 1e-8:
            drift = abs(second_half_mean - first_half_mean) / overall_std
            if drift > self.drift_threshold:
                return StopSignal(
                    should_stop=True,
                    reason=StopReason.ADVANTAGE_DRIFT,
                    details=(
                        f"Advantage mean drifted {drift:.2f} std devs "
                        f"({first_half_mean:.4f} -> {second_half_mean:.4f})"
                    ),
                    severity=min(drift / self.drift_threshold, 1.0),
                )

        # Check variance explosion
        first_half_std = np.mean(recent_stds[: self.window // 2])
        second_half_std = np.mean(recent_stds[self.window // 2 :])

        if first_half_std > 1e-8:
            std_ratio = second_half_std / first_half_std
            if std_ratio > self.drift_threshold:
                return StopSignal(
                    should_stop=True,
                    reason=StopReason.ADVANTAGE_DRIFT,
                    details=(
                        f"Advantage std exploded: ratio={std_ratio:.2f} "
                        f"({first_half_std:.4f} -> {second_half_std:.4f})"
                    ),
                    severity=min(std_ratio / self.drift_threshold, 1.0),
                )

        return StopSignal()


class StopConditionAggregator:
    """Aggregate multiple stop condition detectors."""

    def __init__(
        self,
        collapse: CollapseDetector | None = None,
        hacking: HackingDetector | None = None,
        drift: AdvantageDriftDetector | None = None,
    ):
        self.collapse = collapse or CollapseDetector()
        self.hacking = hacking or HackingDetector()
        self.drift = drift or AdvantageDriftDetector()
        self._signals: list[StopSignal] = []

    def check(
        self,
        entropy: float | None = None,
        diversity: float | None = None,
        exploit_rate: float | None = None,
        adv_mean: float | None = None,
        adv_std: float | None = None,
    ) -> StopSignal:
        """Check all stop conditions. Returns the most critical signal."""
        signals = []

        if entropy is not None and diversity is not None:
            signals.append(self.collapse.update(entropy, diversity))

        if exploit_rate is not None:
            signals.append(self.hacking.update(exploit_rate))

        if adv_mean is not None and adv_std is not None:
            signals.append(self.drift.update(adv_mean, adv_std))

        # Return most severe signal
        stop_signals = [s for s in signals if s.should_stop]
        if stop_signals:
            worst = max(stop_signals, key=lambda s: s.severity)
            logger.warning(f"STOP CONDITION: {worst.reason.value} — {worst.details}")
            self._signals.append(worst)
            return worst

        # Log warnings
        warning_signals = [s for s in signals if s.severity > 0.5]
        for s in warning_signals:
            logger.warning(f"WARNING: {s.reason.value} — {s.details}")

        return StopSignal()

    @property
    def history(self) -> list[StopSignal]:
        return self._signals
