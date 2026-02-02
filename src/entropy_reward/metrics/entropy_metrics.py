"""Token-level and action-level entropy metrics over training time."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
import numpy as np


class TokenEntropy:
    """Track token-level entropy H(pi(token|context)) over training.

    Maintains a rolling window for trend analysis.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._history: deque[float] = deque(maxlen=window_size)
        self._all_values: list[float] = []

    def compute(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> float:
        """Compute mean token entropy from logits.

        Args:
            logits: (batch, seq_len, vocab)
            mask: (batch, seq_len) — 1 for valid tokens
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

        if mask is not None:
            token_entropy = (token_entropy * mask).sum() / mask.sum().clamp(min=1)
        else:
            token_entropy = token_entropy.mean()

        val = token_entropy.item()
        self._history.append(val)
        self._all_values.append(val)
        return val

    def trend(self) -> float:
        """Return slope of entropy over recent window (negative = declining)."""
        if len(self._history) < 10:
            return 0.0
        y = np.array(list(self._history))
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def relative_drop(self) -> float:
        """Return relative drop from initial to current entropy."""
        if len(self._all_values) < 2:
            return 0.0
        initial = np.mean(self._all_values[:10])
        current = np.mean(list(self._history)[-10:])
        return 1.0 - current / (initial + 1e-8)

    @property
    def current(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def history(self) -> list[float]:
        return self._all_values


class ActionEntropy:
    """Track action-level entropy: diversity of chosen actions/tool calls.

    Measures entropy of the distribution over action types within a batch.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._history: deque[float] = deque(maxlen=window_size)
        self._all_values: list[float] = []

    def compute(self, actions: list[str]) -> float:
        """Compute entropy of action type distribution.

        Args:
            actions: list of action names/types from current batch
        """
        if not actions:
            return 0.0

        from collections import Counter

        counts = Counter(actions)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)

        self._history.append(entropy)
        self._all_values.append(entropy)
        return entropy

    def trend(self) -> float:
        if len(self._history) < 10:
            return 0.0
        y = np.array(list(self._history))
        x = np.arange(len(y))
        return float(np.polyfit(x, y, 1)[0])

    @property
    def current(self) -> float:
        return self._history[-1] if self._history else 0.0

    @property
    def history(self) -> list[float]:
        return self._all_values
