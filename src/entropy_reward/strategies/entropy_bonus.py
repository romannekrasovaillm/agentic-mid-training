"""Entropy bonus strategies: constant vs adaptive.

Constant: fixed coefficient added to reward as entropy regularization.
Adaptive: dynamically adjusts coefficient based on real-time entropy/diversity signals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


class ConstantEntropyBonus:
    """Fixed entropy bonus: R_ent = coeff * H(pi(·|s))."""

    def __init__(self, coeff: float = 0.01):
        self.coeff = coeff

    def compute(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy bonus from logits.

        Args:
            logits: shape (batch, seq_len, vocab) or (batch, vocab)

        Returns:
            Scalar entropy bonus (mean over batch).
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch,) or (batch, seq_len)
        return self.coeff * entropy.mean()

    def get_coeff(self) -> float:
        return self.coeff

    def state_dict(self) -> dict:
        return {"coeff": self.coeff}


class AdaptiveEntropyBonus:
    """Adaptive entropy bonus that responds to diversity/entropy collapse signals.

    When entropy or diversity drops below a threshold (relative to initial measurement),
    the coefficient is increased. When entropy is healthy, the coefficient decays back.

    Uses dual signals:
    - Token-level entropy (direct from logits)
    - Diversity EMA (external signal from diversity metrics)
    """

    def __init__(
        self,
        target_entropy: float = 0.0,
        alpha: float = 0.1,
        min_coeff: float = 0.001,
        max_coeff: float = 0.1,
        diversity_ema_decay: float = 0.99,
        entropy_drop_threshold: float = 0.3,
    ):
        self.target_entropy = target_entropy
        self.alpha = alpha
        self.min_coeff = min_coeff
        self.max_coeff = max_coeff
        self.diversity_ema_decay = diversity_ema_decay
        self.entropy_drop_threshold = entropy_drop_threshold

        # State
        self._coeff = (min_coeff + max_coeff) / 2
        self._initial_entropy: float | None = None
        self._entropy_ema: float | None = None
        self._diversity_ema: float | None = None
        self._step = 0

    def compute(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute adaptive entropy bonus and update internal state."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        mean_entropy = entropy.mean().item()

        self._update_entropy_state(mean_entropy)
        self._step += 1

        return self._coeff * entropy.mean()

    def update_diversity_signal(self, diversity: float):
        """Update with external diversity measurement (e.g., 1 - self_BLEU)."""
        if self._diversity_ema is None:
            self._diversity_ema = diversity
        else:
            self._diversity_ema = (
                self.diversity_ema_decay * self._diversity_ema
                + (1 - self.diversity_ema_decay) * diversity
            )

    def _update_entropy_state(self, current_entropy: float):
        """Adjust coefficient based on entropy trend."""
        if self._initial_entropy is None:
            self._initial_entropy = current_entropy
            self._entropy_ema = current_entropy
            if self.target_entropy <= 0:
                self.target_entropy = current_entropy * 0.7  # maintain at least 70%
            return

        # Update EMA
        self._entropy_ema = (
            self.diversity_ema_decay * self._entropy_ema
            + (1 - self.diversity_ema_decay) * current_entropy
        )

        # Compute relative entropy drop
        relative_drop = 1.0 - (self._entropy_ema / (self._initial_entropy + 1e-8))

        # Dual signal: entropy drop and diversity drop
        entropy_alarm = relative_drop > self.entropy_drop_threshold
        diversity_alarm = (
            self._diversity_ema is not None and self._diversity_ema < (1 - self.entropy_drop_threshold)
        )

        if entropy_alarm or diversity_alarm:
            # Increase coefficient to fight collapse
            error = self.target_entropy - self._entropy_ema
            self._coeff += self.alpha * max(error, 0)
        else:
            # Gentle decay toward minimum when entropy is healthy
            self._coeff *= 0.999

        self._coeff = max(self.min_coeff, min(self.max_coeff, self._coeff))

    def get_coeff(self) -> float:
        return self._coeff

    def state_dict(self) -> dict:
        return {
            "coeff": self._coeff,
            "initial_entropy": self._initial_entropy,
            "entropy_ema": self._entropy_ema,
            "diversity_ema": self._diversity_ema,
            "step": self._step,
        }

    def load_state_dict(self, state: dict):
        self._coeff = state["coeff"]
        self._initial_entropy = state["initial_entropy"]
        self._entropy_ema = state["entropy_ema"]
        self._diversity_ema = state["diversity_ema"]
        self._step = state["step"]
