"""Advantage statistics tracker for monitoring training health."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class AdvStats:
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    skew: float = 0.0
    kurtosis: float = 0.0
    fraction_positive: float = 0.0


class AdvantageStatistics:
    """Track advantage distribution statistics for drift detection.

    Monitors: mean, std, min, max, skew, kurtosis, fraction positive.
    These statistics are used to detect advantage drift (stop condition).
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._history: list[AdvStats] = []
        self._means: deque[float] = deque(maxlen=window_size)
        self._stds: deque[float] = deque(maxlen=window_size)

    def compute(self, advantages: torch.Tensor) -> AdvStats:
        """Compute advantage statistics from a batch."""
        adv = advantages.detach().float().cpu().numpy()

        stats = AdvStats(
            mean=float(np.mean(adv)),
            std=float(np.std(adv)),
            min=float(np.min(adv)),
            max=float(np.max(adv)),
            skew=float(self._skewness(adv)),
            kurtosis=float(self._kurtosis(adv)),
            fraction_positive=float((adv > 0).mean()),
        )

        self._history.append(stats)
        self._means.append(stats.mean)
        self._stds.append(stats.std)
        return stats

    def mean_drift(self) -> float:
        """Measure absolute drift in advantage mean over recent window."""
        if len(self._means) < 10:
            return 0.0
        recent = list(self._means)
        first_half = np.mean(recent[: len(recent) // 2])
        second_half = np.mean(recent[len(recent) // 2 :])
        return abs(second_half - first_half)

    def std_trend(self) -> float:
        """Trend in advantage std (positive = increasing variance)."""
        if len(self._stds) < 10:
            return 0.0
        y = np.array(list(self._stds))
        x = np.arange(len(y))
        return float(np.polyfit(x, y, 1)[0])

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        n = len(x)
        if n < 3:
            return 0.0
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-8:
            return 0.0
        return float(np.mean(((x - m) / s) ** 3))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        n = len(x)
        if n < 4:
            return 0.0
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-8:
            return 0.0
        return float(np.mean(((x - m) / s) ** 4) - 3)

    @property
    def history(self) -> list[AdvStats]:
        return self._history
