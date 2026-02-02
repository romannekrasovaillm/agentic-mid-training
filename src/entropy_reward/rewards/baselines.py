"""Reward baselines for advantage estimation.

Implements:
- Group normalization baseline (standard GRPO)
- Leave-one-out (LOO) baseline (hypothesis C2)
- Jackknife baseline (hypothesis C2)

Each supports optional separate baselines for different reward components.
"""

from __future__ import annotations

import torch
import numpy as np


class GroupNormBaseline:
    """Standard GRPO group normalization baseline.

    Advantage = (R - mean(R_group)) / (std(R_group) + eps)

    Hypothesis C1: this + strict R_format creates systematic bias —
    policy learns risk avoidance over problem solving.
    """

    def __init__(self, eps: float = 1e-8, separate: bool = False):
        self.eps = eps
        self.separate = separate

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        reward_components: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute normalized advantages.

        Args:
            rewards: (batch,) total rewards
            group_size: number of samples per prompt group
            reward_components: if separate=True, dict with 'format', 'tool', 'acc' tensors

        Returns:
            (batch,) advantage values
        """
        if self.separate and reward_components is not None:
            return self._compute_separate(reward_components, group_size)

        batch = rewards.shape[0]
        n_groups = batch // group_size
        advantages = torch.zeros_like(rewards)

        for i in range(n_groups):
            start = i * group_size
            end = start + group_size
            group = rewards[start:end]
            mean = group.mean()
            std = group.std()
            advantages[start:end] = (group - mean) / (std + self.eps)

        return advantages

    def _compute_separate(
        self, components: dict[str, torch.Tensor], group_size: int
    ) -> torch.Tensor:
        """Compute advantages with separate baselines per component."""
        total_adv = torch.zeros_like(next(iter(components.values())))
        batch = total_adv.shape[0]
        n_groups = batch // group_size

        for name, rewards in components.items():
            for i in range(n_groups):
                start = i * group_size
                end = start + group_size
                group = rewards[start:end]
                mean = group.mean()
                std = group.std()
                total_adv[start:end] += (group - mean) / (std + self.eps)

        return total_adv / len(components)


class LeaveOneOutBaseline:
    """Leave-one-out baseline: baseline for sample i = mean of group excluding i.

    Reduces bias from group normalization without variance explosion.
    """

    def __init__(self, eps: float = 1e-8, separate: bool = False):
        self.eps = eps
        self.separate = separate

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        reward_components: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.separate and reward_components is not None:
            return self._compute_separate(reward_components, group_size)

        batch = rewards.shape[0]
        n_groups = batch // group_size
        advantages = torch.zeros_like(rewards)

        for i in range(n_groups):
            start = i * group_size
            end = start + group_size
            group = rewards[start:end]

            for j in range(group_size):
                # LOO mean: exclude current sample
                mask = torch.ones(group_size, dtype=torch.bool, device=rewards.device)
                mask[j] = False
                loo_mean = group[mask].mean()
                loo_std = group[mask].std()
                advantages[start + j] = (group[j] - loo_mean) / (loo_std + self.eps)

        return advantages

    def _compute_separate(
        self, components: dict[str, torch.Tensor], group_size: int
    ) -> torch.Tensor:
        total_adv = torch.zeros_like(next(iter(components.values())))
        batch = total_adv.shape[0]
        n_groups = batch // group_size

        for name, rewards in components.items():
            for i in range(n_groups):
                start = i * group_size
                end = start + group_size
                group = rewards[start:end]

                for j in range(group_size):
                    mask = torch.ones(group_size, dtype=torch.bool, device=rewards.device)
                    mask[j] = False
                    loo_mean = group[mask].mean()
                    loo_std = group[mask].std()
                    total_adv[start + j] += (group[j] - loo_mean) / (loo_std + self.eps)

        return total_adv / len(components)


class JackknifeBaseline:
    """Jackknife baseline: bias-corrected leave-one-out estimator.

    Jackknife advantage = n * mean_all - (n-1) * mean_loo
    This provides bias correction for the baseline estimate.
    """

    def __init__(self, eps: float = 1e-8, separate: bool = False):
        self.eps = eps
        self.separate = separate

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
        reward_components: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.separate and reward_components is not None:
            return self._compute_separate(reward_components, group_size)

        batch = rewards.shape[0]
        n_groups = batch // group_size
        advantages = torch.zeros_like(rewards)
        n = group_size

        for i in range(n_groups):
            start = i * group_size
            end = start + group_size
            group = rewards[start:end]
            mean_all = group.mean()
            std_all = group.std()

            for j in range(group_size):
                mask = torch.ones(group_size, dtype=torch.bool, device=rewards.device)
                mask[j] = False
                mean_loo = group[mask].mean()
                # Jackknife bias-corrected baseline
                jk_baseline = n * mean_all - (n - 1) * mean_loo
                advantages[start + j] = (group[j] - jk_baseline) / (std_all + self.eps)

        return advantages

    def _compute_separate(
        self, components: dict[str, torch.Tensor], group_size: int
    ) -> torch.Tensor:
        total_adv = torch.zeros_like(next(iter(components.values())))
        batch = total_adv.shape[0]
        n_groups = batch // group_size
        n = group_size

        for name, rewards in components.items():
            for i in range(n_groups):
                start = i * group_size
                end = start + group_size
                group = rewards[start:end]
                mean_all = group.mean()
                std_all = group.std()

                for j in range(group_size):
                    mask = torch.ones(group_size, dtype=torch.bool, device=rewards.device)
                    mask[j] = False
                    mean_loo = group[mask].mean()
                    jk_baseline = n * mean_all - (n - 1) * mean_loo
                    total_adv[start + j] += (group[j] - jk_baseline) / (std_all + self.eps)

        return total_adv / len(components)
