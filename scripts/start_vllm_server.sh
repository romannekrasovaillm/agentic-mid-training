#!/usr/bin/env bash
# Launch vLLM OpenAI-compatible server for GigaChat3-10B-A1.8B (MoE).
#
# This server handles fast batched generation (rollouts) while the HF model
# in the trainer handles forward-pass logits for GRPO loss and KL computation.
#
# NOTE: GigaChat3 uses FP8-quantized MoE experts. The default flashinfer_cutlass
# MoE kernel requires CUDA 12.7+ for FP8 block scaling. On CUDA ≤12.6 we force
# the Triton MoE backend via VLLM_FUSED_MOE_BACKEND=triton.
#
# Usage:
#   bash scripts/start_vllm_server.sh                       # defaults
#   bash scripts/start_vllm_server.sh --tp 1 --gpu-util 0.45
#   MODEL=my-org/MyModel bash scripts/start_vllm_server.sh  # custom model
#
# Environment variables (all optional):
#   MODEL               — HF model name (default: ai-sage/GigaChat3-10B-A1.8B)
#   HOST                — bind address   (default: 0.0.0.0)
#   PORT                — port           (default: 8000)
#   TP                  — tensor parallel (default: 1)
#   GPU_MEMORY_UTIL     — fraction        (default: 0.45)
#   MAX_MODEL_LEN       — max context     (default: 4096)
#   DTYPE               — dtype           (default: bfloat16)
#   ENFORCE_EAGER       — 0 or 1          (default: 1 for MoE stability)
#   MOE_BACKEND         — triton|cutlass  (default: triton for CUDA ≤12.6)

set -euo pipefail

MODEL="${MODEL:-ai-sage/GigaChat3-10B-A1.8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.45}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
MOE_BACKEND="${MOE_BACKEND:-triton}"

# Parse CLI overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";           shift 2 ;;
        --host)        HOST="$2";            shift 2 ;;
        --port)        PORT="$2";            shift 2 ;;
        --tp)          TP="$2";              shift 2 ;;
        --gpu-util)    GPU_MEMORY_UTIL="$2"; shift 2 ;;
        --max-len)     MAX_MODEL_LEN="$2";   shift 2 ;;
        --dtype)       DTYPE="$2";           shift 2 ;;
        --eager)       ENFORCE_EAGER=1;      shift ;;
        --no-eager)    ENFORCE_EAGER=0;      shift ;;
        --moe-backend) MOE_BACKEND="$2";     shift 2 ;;
        *)             echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Force Triton MoE backend to avoid flashinfer CUTLASS FP8 block scaling
# error on CUDA ≤12.6: "FP8 block scaling not implemented for CUDA 12.6 or lower"
export VLLM_FUSED_MOE_BACKEND="${MOE_BACKEND}"

echo "═══════════════════════════════════════════════════════════════"
echo "  vLLM Server"
echo "  Model:            ${MODEL}"
echo "  Host:             ${HOST}:${PORT}"
echo "  Tensor parallel:  ${TP}"
echo "  GPU mem util:     ${GPU_MEMORY_UTIL}"
echo "  Max model len:    ${MAX_MODEL_LEN}"
echo "  Dtype:            ${DTYPE}"
echo "  Enforce eager:    ${ENFORCE_EAGER}"
echo "  MoE backend:      ${MOE_BACKEND} (VLLM_FUSED_MOE_BACKEND)"
echo "═══════════════════════════════════════════════════════════════"

CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --gpu-memory-utilization "$GPU_MEMORY_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --dtype "$DTYPE"
    --trust-remote-code
    --disable-log-requests
)

if [[ "$ENFORCE_EAGER" == "1" ]]; then
    CMD+=(--enforce-eager)
fi

echo ""
echo "Running: ${CMD[*]}"
echo ""

exec "${CMD[@]}"
