"""Configuration loader for experiment configs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EntropyConfig:
    strategy: str = "constant"  # "constant" | "adaptive"
    constant_bonus: float = 0.01
    # Adaptive params
    adaptive_target_entropy: float = 0.0  # 0 = auto-detect from initial
    adaptive_alpha: float = 0.1  # learning rate for entropy coefficient
    adaptive_min_coeff: float = 0.001
    adaptive_max_coeff: float = 0.1
    diversity_ema_decay: float = 0.99
    entropy_drop_threshold: float = 0.3  # trigger adaptive if entropy drops >30%


@dataclass
class KLConfig:
    enabled: bool = True
    initial_coeff: float = 0.2
    final_coeff: float = 0.02
    schedule: str = "linear"  # "linear" | "cosine" | "step"
    warmup_steps: int = 100
    total_steps: int = 5000
    step_milestones: list[float] = field(default_factory=lambda: [0.3, 0.6, 0.9])
    step_gamma: float = 0.5


@dataclass
class RewardConfig:
    format_mode: str = "strict"  # "strict" | "partial"
    format_weight: float = 1.0
    tool_weight: float = 1.0
    accuracy_weight: float = 1.0
    # Partial format credit params
    partial_tag_credit: float = 0.3
    partial_structure_credit: float = 0.5
    partial_full_credit: float = 1.0
    # Baseline
    baseline: str = "group_norm"  # "group_norm" | "leave_one_out" | "jackknife"
    separate_baselines: bool = False  # separate baselines for tool/acc


@dataclass
class MetricsConfig:
    log_interval: int = 10
    eval_interval: int = 100
    self_bleu_n: int = 4
    self_bleu_sample_size: int = 100
    pattern_window: int = 50
    track_advantage_stats: bool = True
    track_cps: bool = True  # calls-per-step


@dataclass
class EvalConfig:
    ood_enabled: bool = True
    ood_datasets: list[str] = field(default_factory=lambda: ["format_variation", "tool_variation"])
    metamorphic_enabled: bool = True
    metamorphic_transforms: list[str] = field(
        default_factory=lambda: ["reorder_tools", "synonym_replace", "format_shift"]
    )
    redteam_enabled: bool = True
    redteam_exploit_budget: int = 50


@dataclass
class StopConfig:
    entropy_collapse_threshold: float = 0.3
    entropy_collapse_window: int = 50
    diversity_collapse_threshold: float = 0.4
    hacking_passrate_threshold: float = 0.8
    hacking_eval_interval: int = 200
    advantage_drift_threshold: float = 2.0
    advantage_drift_window: int = 100


@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    ref_model_name: str = ""  # if empty, use model_name
    learning_rate: float = 1e-5
    batch_size: int = 8
    group_size: int = 4  # GRPO group size
    max_steps: int = 5000
    max_seq_len: int = 2048
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    output_dir: str = "outputs"
    wandb_project: str = "entropy-reward-experiments"
    wandb_run_name: str = ""


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    stop: StopConfig = field(default_factory=StopConfig)


def _merge_dict(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _dict_to_dataclass(cls, data: dict[str, Any]):
    """Recursively convert dict to nested dataclasses."""
    if not isinstance(data, dict):
        return data
    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for k, v in data.items():
        if k in fieldtypes:
            ft = fieldtypes[k]
            # Resolve string type annotations
            if isinstance(ft, str):
                ft = eval(ft)  # noqa: S307 — controlled context
            if hasattr(ft, "__dataclass_fields__") and isinstance(v, dict):
                kwargs[k] = _dict_to_dataclass(ft, v)
            else:
                kwargs[k] = v
    return cls(**kwargs)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load experiment config from YAML with optional overrides."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _merge_dict(raw, overrides)

    return _dict_to_dataclass(ExperimentConfig, raw)
