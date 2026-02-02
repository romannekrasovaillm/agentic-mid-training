"""KL-divergence anchoring to a reference policy.

Schedule: strong anchoring early (high KL penalty) → weak anchoring late.
Supports linear, cosine, and step decay schedules.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class KLAnchor:
    """KL penalty between current policy and frozen reference policy.

    KL_penalty = coeff(t) * KL(pi_current || pi_ref)

    The coefficient decays from initial_coeff to final_coeff over training.
    """

    def __init__(
        self,
        initial_coeff: float = 0.2,
        final_coeff: float = 0.02,
        schedule: str = "linear",
        warmup_steps: int = 100,
        total_steps: int = 5000,
        step_milestones: list[float] | None = None,
        step_gamma: float = 0.5,
    ):
        self.initial_coeff = initial_coeff
        self.final_coeff = final_coeff
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_milestones = step_milestones or [0.3, 0.6, 0.9]
        self.step_gamma = step_gamma
        self._current_step = 0

    def get_coeff(self, step: int | None = None) -> float:
        """Get current KL coefficient based on schedule."""
        if step is not None:
            self._current_step = step
        t = self._current_step

        # During warmup, ramp up from 0 to initial_coeff
        if t < self.warmup_steps:
            return self.initial_coeff * (t / max(self.warmup_steps, 1))

        # Effective progress after warmup
        progress = min(
            (t - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1), 1.0
        )

        if self.schedule == "linear":
            coeff = self.initial_coeff + (self.final_coeff - self.initial_coeff) * progress
        elif self.schedule == "cosine":
            coeff = self.final_coeff + 0.5 * (self.initial_coeff - self.final_coeff) * (
                1 + math.cos(math.pi * progress)
            )
        elif self.schedule == "step":
            coeff = self.initial_coeff
            for milestone in self.step_milestones:
                if progress >= milestone:
                    coeff *= self.step_gamma
            coeff = max(coeff, self.final_coeff)
        else:
            raise ValueError(f"Unknown KL schedule: {self.schedule}")

        return coeff

    def compute(
        self,
        current_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute KL divergence penalty.

        Args:
            current_logits: (batch, seq_len, vocab) from current policy
            ref_logits: (batch, seq_len, vocab) from reference policy
            mask: (batch, seq_len) attention mask, 1 = valid token

        Returns:
            Scalar KL penalty weighted by current schedule coefficient.
        """
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)

        # KL(ref || current) = sum ref * (log ref - log current)
        kl = F.kl_div(current_log_probs, ref_probs, reduction="none").sum(dim=-1)

        if mask is not None:
            kl = (kl * mask).sum() / mask.sum().clamp(min=1)
        else:
            kl = kl.mean()

        coeff = self.get_coeff()
        self._current_step += 1

        return coeff * kl, kl.detach(), coeff

    def state_dict(self) -> dict:
        return {"current_step": self._current_step}

    def load_state_dict(self, state: dict):
        self._current_step = state["current_step"]
