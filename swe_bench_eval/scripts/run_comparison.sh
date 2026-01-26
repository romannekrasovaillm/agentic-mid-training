#!/bin/bash
#
# A/B Comparison: Baseline vs RL-trained LoRA on SWE-Verified
#
# Runs both models on the same subset of SWE-bench_Verified
# and compares results.
#
# Usage:
#   ./run_comparison.sh \
#       --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
#       --lora-baseline ./checkpoints/baseline-lora \
#       --lora-trained ./checkpoints/rl-trained-lora \
#       --output-dir ./outputs/comparison
#

set -e

# =============================================================================
# Default Configuration
# =============================================================================
BASE_MODEL=""
LORA_BASELINE=""
LORA_TRAINED=""
OUTPUT_DIR="./outputs/comparison-$(date +%Y%m%d-%H%M%S)"
SERVER_PORT=8080
MAX_INSTANCES=50  # Default to subset for comparison
SANDBOX="docker"
MAX_TURNS=30

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --lora-baseline)
            LORA_BASELINE="$2"
            shift 2
            ;;
        --lora-trained)
            LORA_TRAINED="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --sandbox)
            SANDBOX="$2"
            shift 2
            ;;
        --max-turns)
            MAX_TURNS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --base-model <model> --lora-baseline <path> --lora-trained <path> [options]"
            echo ""
            echo "Required:"
            echo "  --base-model       Base model path or HuggingFace repo ID"
            echo "  --lora-baseline    Path to baseline LoRA adapter"
            echo "  --lora-trained     Path to RL-trained LoRA adapter"
            echo ""
            echo "Optional:"
            echo "  --output-dir       Output directory"
            echo "  --port             Server port (default: 8080)"
            echo "  --max-instances    Instances to evaluate (default: 50)"
            echo "  --sandbox          Sandbox type (default: docker)"
            echo "  --max-turns        Max turns per instance (default: 30)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate Arguments
# =============================================================================
if [[ -z "$BASE_MODEL" ]] || [[ -z "$LORA_BASELINE" ]] || [[ -z "$LORA_TRAINED" ]]; then
    echo "Error: --base-model, --lora-baseline, and --lora-trained are required"
    exit 1
fi

# =============================================================================
# Setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "$OUTPUT_DIR"/{baseline,trained,comparison,logs}

echo "=============================================="
echo "A/B Comparison: Baseline vs RL-Trained LoRA"
echo "=============================================="
echo "Base Model:       $BASE_MODEL"
echo "Baseline LoRA:    $LORA_BASELINE"
echo "Trained LoRA:     $LORA_TRAINED"
echo "Output Dir:       $OUTPUT_DIR"
echo "Max Instances:    $MAX_INSTANCES"
echo "=============================================="

# =============================================================================
# Start Multi-LoRA Server
# =============================================================================
echo ""
echo "[1/4] Starting multi-LoRA server..."

SERVER_LOG="$OUTPUT_DIR/logs/server.log"
SERVER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

python "$PROJECT_DIR/models/vllm_lora_server.py" \
    --base-model "$BASE_MODEL" \
    --lora-modules "baseline:$LORA_BASELINE,trained:$LORA_TRAINED" \
    --port "$SERVER_PORT" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Wait for server
echo "Waiting for server..."
MAX_WAIT=300
WAIT_COUNT=0
while ! curl -s "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; do
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [[ $WAIT_COUNT -ge $MAX_WAIT ]]; then
        echo "Error: Server failed to start"
        exit 1
    fi
done
echo "Server ready!"

# =============================================================================
# Run Baseline
# =============================================================================
echo ""
echo "[2/4] Running baseline model..."

export OPENAI_API_KEY="not-needed"
export OPENAI_API_BASE="http://localhost:$SERVER_PORT/v1"

mini swebench \
    --model openai/baseline \
    --sandbox "$SANDBOX" \
    --output "$OUTPUT_DIR/baseline/trajectories" \
    --max-turns "$MAX_TURNS" \
    --max-instances "$MAX_INSTANCES" \
    2>&1 | tee "$OUTPUT_DIR/logs/baseline.log"

# =============================================================================
# Run Trained
# =============================================================================
echo ""
echo "[3/4] Running trained model..."

mini swebench \
    --model openai/trained \
    --sandbox "$SANDBOX" \
    --output "$OUTPUT_DIR/trained/trajectories" \
    --max-turns "$MAX_TURNS" \
    --max-instances "$MAX_INSTANCES" \
    2>&1 | tee "$OUTPUT_DIR/logs/trained.log"

# =============================================================================
# Compare Results
# =============================================================================
echo ""
echo "[4/4] Comparing results..."

# Create comparison report
python3 << 'PYTHON_SCRIPT'
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

output_dir = os.environ.get('OUTPUT_DIR', './outputs/comparison')
baseline_dir = Path(output_dir) / "baseline" / "trajectories"
trained_dir = Path(output_dir) / "trained" / "trajectories"
comparison_dir = Path(output_dir) / "comparison"

def load_results(traj_dir):
    """Load results from trajectory files."""
    results = {}
    if not traj_dir.exists():
        return results

    for f in traj_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                instance_id = data.get("instance_id", f.stem)
                results[instance_id] = {
                    "resolved": data.get("resolved", False),
                    "num_turns": len(data.get("trajectory", [])),
                    "has_patch": bool(data.get("patch")),
                }
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}", file=sys.stderr)

    return results

# Load results
baseline_results = load_results(baseline_dir)
trained_results = load_results(trained_dir)

# Calculate metrics
all_instances = set(baseline_results.keys()) | set(trained_results.keys())

comparison = {
    "total_instances": len(all_instances),
    "baseline": {
        "evaluated": len(baseline_results),
        "resolved": sum(1 for r in baseline_results.values() if r["resolved"]),
        "has_patch": sum(1 for r in baseline_results.values() if r["has_patch"]),
        "avg_turns": sum(r["num_turns"] for r in baseline_results.values()) / max(len(baseline_results), 1),
    },
    "trained": {
        "evaluated": len(trained_results),
        "resolved": sum(1 for r in trained_results.values() if r["resolved"]),
        "has_patch": sum(1 for r in trained_results.values() if r["has_patch"]),
        "avg_turns": sum(r["num_turns"] for r in trained_results.values()) / max(len(trained_results), 1),
    },
    "instance_comparison": [],
}

# Per-instance comparison
for instance_id in sorted(all_instances):
    baseline = baseline_results.get(instance_id, {})
    trained = trained_results.get(instance_id, {})

    comparison["instance_comparison"].append({
        "instance_id": instance_id,
        "baseline_resolved": baseline.get("resolved", None),
        "trained_resolved": trained.get("resolved", None),
        "improvement": (trained.get("resolved", False) and not baseline.get("resolved", False)),
        "regression": (baseline.get("resolved", False) and not trained.get("resolved", False)),
    })

# Calculate improvements
improvements = sum(1 for c in comparison["instance_comparison"] if c["improvement"])
regressions = sum(1 for c in comparison["instance_comparison"] if c["regression"])
comparison["improvements"] = improvements
comparison["regressions"] = regressions
comparison["net_improvement"] = improvements - regressions

# Save comparison
comparison_dir.mkdir(parents=True, exist_ok=True)
with open(comparison_dir / "comparison.json", "w") as fp:
    json.dump(comparison, fp, indent=2)

# Print summary
print()
print("=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)
print()
print(f"Total instances evaluated: {comparison['total_instances']}")
print()
print("BASELINE:")
print(f"  Resolved: {comparison['baseline']['resolved']} / {comparison['baseline']['evaluated']}")
if comparison['baseline']['evaluated'] > 0:
    print(f"  Rate: {100 * comparison['baseline']['resolved'] / comparison['baseline']['evaluated']:.1f}%")
print(f"  Avg turns: {comparison['baseline']['avg_turns']:.1f}")
print()
print("TRAINED (RL):")
print(f"  Resolved: {comparison['trained']['resolved']} / {comparison['trained']['evaluated']}")
if comparison['trained']['evaluated'] > 0:
    print(f"  Rate: {100 * comparison['trained']['resolved'] / comparison['trained']['evaluated']:.1f}%")
print(f"  Avg turns: {comparison['trained']['avg_turns']:.1f}")
print()
print("DELTA:")
print(f"  Improvements: +{improvements} (trained solved, baseline failed)")
print(f"  Regressions:  -{regressions} (baseline solved, trained failed)")
print(f"  Net change:   {'+' if comparison['net_improvement'] >= 0 else ''}{comparison['net_improvement']}")
print()
print(f"Detailed comparison saved to: {comparison_dir / 'comparison.json'}")
print("=" * 60)
PYTHON_SCRIPT

echo ""
echo "Comparison complete!"
echo "Results: $OUTPUT_DIR/comparison/"
