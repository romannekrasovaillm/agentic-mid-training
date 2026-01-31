"""Logging utilities for reward decomposition and metrics tracking."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Create a logger writing to file and stderr."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)

    fmt = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@dataclass
class StepLog:
    step: int = 0
    # Reward decomposition
    r_format: float = 0.0
    r_tool: float = 0.0
    r_acc: float = 0.0
    r_total: float = 0.0
    # Entropy / density
    token_entropy: float = 0.0
    action_entropy: float = 0.0
    entropy_coeff: float = 0.0
    kl_div: float = 0.0
    kl_coeff: float = 0.0
    # CPS / tool frequencies
    calls_per_step: float = 0.0
    tool_frequencies: dict[str, float] = field(default_factory=dict)
    # Advantage statistics
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    # Diversity
    self_bleu: float = 0.0
    trajectory_uniqueness: float = 0.0
    pattern_repetition: float = 0.0
    # Timing
    wall_time: float = 0.0


class RewardLogger:
    """Structured logger that writes JSONL for easy downstream analysis."""

    def __init__(self, output_dir: str | Path, run_name: str = "run"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / f"{run_name}_log.jsonl"
        self._start_time = time.time()
        self._file = open(self.log_path, "a")  # noqa: SIM115
        self._wandb = None

    def init_wandb(self, project: str, run_name: str, config: dict[str, Any] | None = None):
        try:
            import wandb

            self._wandb = wandb.init(project=project, name=run_name, config=config)
        except ImportError:
            pass

    def log_step(self, log: StepLog):
        log.wall_time = time.time() - self._start_time
        record = asdict(log)
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        if self._wandb is not None:
            import wandb

            flat = {k: v for k, v in record.items() if not isinstance(v, dict)}
            # Flatten tool_frequencies
            for tool, freq in record.get("tool_frequencies", {}).items():
                flat[f"tool_freq/{tool}"] = freq
            wandb.log(flat, step=log.step)

    def close(self):
        self._file.close()
        if self._wandb is not None:
            import wandb

            wandb.finish()
