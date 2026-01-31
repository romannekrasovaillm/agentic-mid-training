"""Configuration loader for experiment configs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class ModelConfig:
    name: str = "ai-sage/GigaChat3-10B-A1.8B"
    ref_model_name: str = ""  # if empty, use name
    torch_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"
    attn_implementation: str = "flash_attention_2"  # "eager" | "sdpa" | "flash_attention_2"
    trust_remote_code: bool = True
    device_map: str = "auto"
    max_memory: dict[str, str] = field(default_factory=dict)
    use_chat_template: bool = True
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class DatasetConfig:
    name: str = "nvidia/Nemotron-Agentic-v1"
    train_split: str = "tool_calling"
    eval_split: str = "interactive_agent"
    max_train_samples: int = 0  # 0 = all
    max_eval_samples: int = 200
    multiturn: bool = False
    max_context_turns: int = 3
    accuracy_tool_weight: float = 0.6
    accuracy_response_weight: float = 0.4
    cache_dir: str = ""


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
    learning_rate: float = 5e-6
    batch_size: int = 4
    group_size: int = 4  # GRPO group size
    max_steps: int = 5000
    max_seq_len: int = 4096
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    output_dir: str = "outputs"
    wandb_project: str = "entropy-reward-experiments"
    wandb_run_name: str = ""
    checkpoint_interval: int = 500


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
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
