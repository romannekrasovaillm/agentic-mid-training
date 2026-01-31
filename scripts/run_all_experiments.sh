#!/usr/bin/env bash
# Run all entropy-reward experiments sequentially.
# Usage: bash scripts/run_all_experiments.sh [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIGS_DIR="$PROJECT_DIR/configs"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN MODE ==="
fi

EXPERIMENTS=(
    "base.yaml"
    "adaptive_entropy.yaml"
    "partial_format.yaml"
    "loo_baseline.yaml"
    "jackknife_baseline.yaml"
    "full_recipe.yaml"
)

echo "========================================"
echo " Entropy-Reward Experiments Suite"
echo " $(date)"
echo "========================================"
echo ""

for config in "${EXPERIMENTS[@]}"; do
    config_path="$CONFIGS_DIR/$config"
    exp_name="${config%.yaml}"

    echo "----------------------------------------"
    echo " Experiment: $exp_name"
    echo " Config: $config_path"
    echo "----------------------------------------"

    python "$SCRIPT_DIR/run_experiment.py" \
        --config "$config_path" \
        --base-config "$CONFIGS_DIR/base.yaml" \
        $DRY_RUN \
        2>&1 | tee "$PROJECT_DIR/outputs/${exp_name}_log.txt"

    echo ""
    echo " $exp_name — done"
    echo ""
done

echo "========================================"
echo " All experiments complete"
echo " $(date)"
echo "========================================"
