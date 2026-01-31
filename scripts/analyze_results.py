#!/usr/bin/env python3
"""Analyze and compare experiment results across all runs.

Produces:
- Entropy/diversity curves per experiment
- Comparison tables
- Stop condition summary
- Entropy-stability recipe recommendation

Usage:
    python scripts/analyze_results.py --output-dir outputs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_run_logs(output_dir: Path) -> dict[str, list[dict]]:
    """Load JSONL logs from all experiment runs."""
    runs = {}
    for logfile in output_dir.rglob("*_log.jsonl"):
        run_name = logfile.stem.replace("_log", "")
        records = []
        with open(logfile) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            runs[run_name] = records
    return runs


def compute_summary(name: str, records: list[dict]) -> dict:
    """Compute summary statistics for a single run."""
    if not records:
        return {"name": name, "steps": 0}

    steps = [r["step"] for r in records]
    token_entropy = [r.get("token_entropy", 0) for r in records]
    self_bleu = [r.get("self_bleu", 0) for r in records]
    r_format = [r.get("r_format", 0) for r in records]
    r_acc = [r.get("r_acc", 0) for r in records]
    uniqueness = [r.get("trajectory_uniqueness", 0) for r in records]

    # Entropy trend (slope of last 50%)
    half = len(token_entropy) // 2
    if half > 5:
        x = np.arange(len(token_entropy[half:]))
        slope = np.polyfit(x, token_entropy[half:], 1)[0]
    else:
        slope = 0

    return {
        "name": name,
        "steps": len(records),
        "final_step": steps[-1] if steps else 0,
        "entropy_initial": np.mean(token_entropy[:10]) if len(token_entropy) >= 10 else 0,
        "entropy_final": np.mean(token_entropy[-10:]) if len(token_entropy) >= 10 else 0,
        "entropy_slope": slope,
        "self_bleu_mean": np.mean(self_bleu),
        "self_bleu_final": np.mean(self_bleu[-10:]) if len(self_bleu) >= 10 else 0,
        "uniqueness_mean": np.mean(uniqueness),
        "r_format_mean": np.mean(r_format),
        "r_acc_mean": np.mean(r_acc),
        "diversity_score": np.mean(uniqueness) * (1 - np.mean(self_bleu)),
        "stopped_early": records[-1].get("step", 0) < 5000 if records else False,
    }


def print_comparison_table(summaries: list[dict]):
    """Print comparison table of all experiments."""
    header = (
        f"{'Experiment':<25} {'Steps':>6} {'H_init':>7} {'H_final':>7} "
        f"{'H_slope':>8} {'BLEU':>6} {'Uniq':>6} {'R_fmt':>6} {'Diversity':>9} {'Early?':>6}"
    )
    print("\n" + "=" * len(header))
    print("EXPERIMENT COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for s in summaries:
        print(
            f"{s['name']:<25} {s['steps']:>6} {s['entropy_initial']:>7.4f} "
            f"{s['entropy_final']:>7.4f} {s['entropy_slope']:>8.5f} "
            f"{s['self_bleu_mean']:>6.3f} {s['uniqueness_mean']:>6.3f} "
            f"{s['r_format_mean']:>6.3f} {s['diversity_score']:>9.4f} "
            f"{'YES' if s['stopped_early'] else 'no':>6}"
        )

    print("=" * len(header))


def recommend_recipe(summaries: list[dict]) -> str:
    """Recommend the best entropy-stability recipe."""
    if not summaries:
        return "No data available."

    # Score: high diversity + high R_format + stable entropy (small negative slope)
    for s in summaries:
        entropy_stability = max(0, 1.0 - abs(s["entropy_slope"]) * 100)
        s["composite_score"] = (
            0.4 * s["diversity_score"]
            + 0.3 * s["r_format_mean"]
            + 0.3 * entropy_stability
        )

    best = max(summaries, key=lambda s: s["composite_score"])
    return (
        f"\nRECOMMENDED RECIPE: {best['name']}\n"
        f"  Composite score: {best['composite_score']:.4f}\n"
        f"  Diversity: {best['diversity_score']:.4f}\n"
        f"  Format success: {best['r_format_mean']:.3f}\n"
        f"  Entropy stability: slope={best['entropy_slope']:.5f}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist.")
        return

    runs = load_run_logs(output_dir)
    if not runs:
        print("No experiment logs found.")
        return

    print(f"Found {len(runs)} experiment runs: {list(runs.keys())}")

    summaries = [compute_summary(name, records) for name, records in runs.items()]
    print_comparison_table(summaries)
    print(recommend_recipe(summaries))


if __name__ == "__main__":
    main()
