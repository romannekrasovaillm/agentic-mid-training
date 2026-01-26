#!/usr/bin/env python3
"""
vLLM Server with LoRA Support for SWE-bench Evaluation

Uses vLLM for high-performance inference with LoRA adapters.
Provides OpenAI-compatible API for mini-swe-agent.

Usage:
    # Single LoRA adapter:
    python vllm_lora_server.py \
        --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
        --lora-path ./checkpoints/swe-agent-lora \
        --port 8080

    # Multiple LoRA adapters (A/B testing):
    python vllm_lora_server.py \
        --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
        --lora-modules "baseline:./checkpoints/baseline-lora,rl-trained:./checkpoints/rl-lora" \
        --port 8080
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_lora_modules(modules_str: str) -> List[dict]:
    """
    Parse LoRA modules from string format.
    Format: "name1:path1,name2:path2"
    """
    modules = []
    for module in modules_str.split(","):
        parts = module.strip().split(":")
        if len(parts) == 2:
            modules.append({
                "name": parts[0].strip(),
                "path": parts[1].strip(),
            })
    return modules


def start_vllm_server(
    base_model: str,
    lora_path: Optional[str] = None,
    lora_modules: Optional[List[dict]] = None,
    port: int = 8080,
    host: str = "0.0.0.0",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    enable_lora: bool = True,
    max_loras: int = 4,
    max_lora_rank: int = 64,
):
    """
    Start vLLM server with LoRA support.

    This function constructs and executes the vLLM serve command.
    """
    # Prepare LoRA modules for vLLM
    if lora_path and not lora_modules:
        lora_modules = [{"name": "default", "path": lora_path}]

    # Build vLLM command
    cmd_parts = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", base_model,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
    ]

    if enable_lora and lora_modules:
        cmd_parts.extend([
            "--enable-lora",
            "--max-loras", str(max_loras),
            "--max-lora-rank", str(max_lora_rank),
        ])

        # Add each LoRA module
        lora_args = []
        for module in lora_modules:
            lora_args.append(f"{module['name']}={module['path']}")

        if lora_args:
            cmd_parts.extend(["--lora-modules", *lora_args])

    logger.info("Starting vLLM server with command:")
    logger.info(" ".join(cmd_parts))

    # Execute vLLM server
    os.execvp(sys.executable, cmd_parts)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Server with LoRA Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single LoRA adapter:
    python vllm_lora_server.py \\
        --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \\
        --lora-path ./checkpoints/swe-agent-lora

    # Multiple LoRA adapters:
    python vllm_lora_server.py \\
        --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \\
        --lora-modules "baseline:./lora-v1,trained:./lora-v2"

    # Call with specific adapter:
    curl http://localhost:8080/v1/chat/completions \\
        -H "Content-Type: application/json" \\
        -d '{"model": "trained", "messages": [...]}'
        """,
    )

    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path or HuggingFace repo ID for base model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        help="Path to single LoRA adapter (will be named 'default')",
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        help="Multiple LoRA modules in format: 'name1:path1,name2:path2'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run server on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)",
    )
    parser.add_argument(
        "--max-loras",
        type=int,
        default=4,
        help="Maximum number of LoRA adapters (default: 4)",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=64,
        help="Maximum LoRA rank (default: 64)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.lora_path and not args.lora_modules:
        parser.error("Either --lora-path or --lora-modules must be specified")

    # Parse LoRA modules
    lora_modules = None
    if args.lora_modules:
        lora_modules = parse_lora_modules(args.lora_modules)

    # Start server
    start_vllm_server(
        base_model=args.base_model,
        lora_path=args.lora_path,
        lora_modules=lora_modules,
        port=args.port,
        host=args.host,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_loras=args.max_loras,
        max_lora_rank=args.max_lora_rank,
    )


if __name__ == "__main__":
    main()
