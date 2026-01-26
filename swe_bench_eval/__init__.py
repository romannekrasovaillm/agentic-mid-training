"""
SWE-bench Evaluation for LoRA-adapted Models

This package provides tools for evaluating LoRA-adapted language models
on SWE-bench_Verified using the mini-swe-agent framework.

Components:
- models/lora_server.py: Transformers-based OpenAI-compatible server
- models/vllm_lora_server.py: vLLM-based server with LoRA support
- scripts/run_swe_verified.sh: Run full SWE-bench_Verified evaluation
- scripts/run_comparison.sh: A/B compare baseline vs trained LoRA
"""

__version__ = "0.1.0"
