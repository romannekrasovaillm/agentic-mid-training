#!/usr/bin/env python3
"""Run a single entropy-reward experiment from a config file.

Usage:
    python scripts/run_experiment.py --config configs/base.yaml
    python scripts/run_experiment.py --config configs/adaptive_entropy.yaml --base-config configs/base.yaml
    python scripts/run_experiment.py --config configs/base.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from entropy_reward.utils.config_loader import (
    ExperimentConfig,
    load_config,
    _merge_dict,
    _dict_to_dataclass,
)
from entropy_reward.utils.logging_utils import setup_logger
from entropy_reward.trainer import GRPOTrainer


TORCH_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run entropy-reward experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--override", type=str, default=None, help="JSON dict of config overrides")
    parser.add_argument("--base-config", type=str, default=None, help="Base config to merge with")
    parser.add_argument("--dry-run", action="store_true", help="Only print config without training")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint dir to resume from")
    return parser.parse_args()


def load_nemotron_data(config: ExperimentConfig):
    """Load nvidia/Nemotron-Agentic-v1 dataset.

    Returns:
        (train_samples, eval_samples) — lists of AgenticSample
    """
    from entropy_reward.data import load_nemotron_splits

    dcfg = config.dataset
    train_samples, eval_samples = load_nemotron_splits(
        max_train=dcfg.max_train_samples,
        max_eval=dcfg.max_eval_samples,
        multiturn=dcfg.multiturn,
        seed=config.training.seed,
        cache_dir=dcfg.cache_dir or None,
    )
    return train_samples, eval_samples


def load_model_and_tokenizer(config: ExperimentConfig):
    """Load model and tokenizer using config.model settings.

    Handles GigaChat3-10B-A1.8B (MoE with MLA + MTP) and standard HF models.
    """
    mcfg = config.model
    model_name = mcfg.name
    dtype = TORCH_DTYPE_MAP.get(mcfg.torch_dtype, torch.bfloat16)

    log = logging.getLogger("experiment")
    log.info(f"Loading model: {model_name}")
    log.info(
        f"  dtype={mcfg.torch_dtype}, attn={mcfg.attn_implementation}, "
        f"device_map={mcfg.device_map}, trust_remote_code={mcfg.trust_remote_code}"
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=mcfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model kwargs — use `dtype` (torch_dtype is deprecated in newer transformers)
    model_kwargs = dict(
        dtype=dtype,
        device_map=mcfg.device_map,
        trust_remote_code=mcfg.trust_remote_code,
    )

    # Attention implementation — auto-detect flash-attn availability
    attn_impl = mcfg.attn_implementation
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            log.warning(
                "flash_attn not installed — falling back to sdpa "
                "(scaled dot-product attention)"
            )
            attn_impl = "sdpa"
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    # Memory constraints
    if mcfg.max_memory:
        model_kwargs["max_memory"] = mcfg.max_memory

    # Policy model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Enable gradient checkpointing for memory efficiency
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled")

    # Apply HF GenerationConfig from model hub (GigaChat3 ships one)
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_name)
    except Exception:
        pass

    # Reference model (frozen copy for KL anchor)
    ref_model_name = mcfg.ref_model_name or model_name
    log.info(f"Loading reference model: {ref_model_name}")
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name, **model_kwargs)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Log GPU memory after loading both models
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            log.info(
                f"GPU {i} ({torch.cuda.get_device_name(i)}): "
                f"{alloc:.1f}GB allocated / {reserved:.1f}GB reserved / {total:.1f}GB total"
            )

    return model, ref_model, tokenizer


def _maybe_launch_vllm_server(config: ExperimentConfig) -> "subprocess.Popen | None":
    """Optionally launch vLLM server as a subprocess.

    Returns the Popen handle (or None if not launched).
    """
    vcfg = config.vllm
    if not vcfg.enabled or not vcfg.launch_server:
        return None

    import subprocess
    import shutil

    log = logging.getLogger("experiment")

    script = Path(__file__).resolve().parent / "start_vllm_server.sh"
    if not script.exists():
        log.error(f"vLLM launch script not found: {script}")
        return None

    # Parse port from base_url
    from urllib.parse import urlparse
    parsed = urlparse(vcfg.base_url)
    port = str(parsed.port or 8000)

    env = {
        **dict(__import__("os").environ),
        "MODEL": config.model.name,
        "PORT": port,
        "TP": str(vcfg.tensor_parallel_size),
        "GPU_MEMORY_UTIL": str(vcfg.gpu_memory_utilization),
        "MAX_MODEL_LEN": str(vcfg.max_model_len),
        "DTYPE": config.model.torch_dtype,
        "ENFORCE_EAGER": "1" if vcfg.enforce_eager else "0",
        # Disable flashinfer/deepgemm FP8 MoE kernels — they require
        # CUDA 12.7+ for FP8 block scaling. Force Triton fallback.
        "VLLM_USE_FLASHINFER_MOE_FP8": "0",
        "VLLM_USE_FLASHINFER_MOE_FP16": "0",
        "VLLM_USE_FLASHINFER_MOE_FP4": "0",
        "VLLM_MOE_USE_DEEP_GEMM": "0",
        "VLLM_FUSED_MOE_BACKEND": "triton",
    }

    log.info(f"Launching vLLM server (port={port}, tp={vcfg.tensor_parallel_size}, "
             f"gpu_util={vcfg.gpu_memory_utilization})...")
    proc = subprocess.Popen(
        ["bash", str(script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.info(f"vLLM server PID: {proc.pid}")
    return proc


def _setup_vllm_client(config: ExperimentConfig):
    """Create and connect a VLLMClient if vLLM is enabled."""
    vcfg = config.vllm
    if not vcfg.enabled:
        return None

    log = logging.getLogger("experiment")

    from entropy_reward.inference import VLLMClient

    client = VLLMClient(
        base_url=vcfg.base_url,
        model_name=config.model.name,
        max_new_tokens=config.model.generation.max_new_tokens,
        temperature=config.model.generation.temperature,
        top_p=config.model.generation.top_p,
        timeout=vcfg.request_timeout,
        max_retries=vcfg.max_retries,
        use_chat_api=config.model.use_chat_template,
    )

    log.info(f"Waiting for vLLM server at {vcfg.base_url}...")
    if client.wait_until_ready(timeout=vcfg.server_timeout):
        log.info("vLLM client connected")
        return client
    else:
        log.warning("vLLM server not reachable — falling back to HF generation")
        return None


def _gpu_mem_str() -> str:
    """Return compact GPU memory usage string."""
    if not torch.cuda.is_available():
        return "no GPU"
    parts = []
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        peak = torch.cuda.max_memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        parts.append(f"GPU{i}: {alloc:.1f}/{peak:.1f}/{total:.1f}GB")
    return " | ".join(parts)


def main():
    args = parse_args()

    # Reduce CUDA memory fragmentation (helps when sharing GPU with vLLM)
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Load config
    overrides = json.loads(args.override) if args.override else None

    if args.base_config:
        # Load base, then overlay experiment config on top
        base_config = load_config(args.base_config)
        base_dict = asdict(base_config)
        with open(args.config) as f:
            exp_raw = yaml.safe_load(f) or {}
        merged = _merge_dict(base_dict, exp_raw)
        if overrides:
            merged = _merge_dict(merged, overrides)
        config = _dict_to_dataclass(ExperimentConfig, merged)
    else:
        config = load_config(args.config, overrides)

    # Setup logging
    log = setup_logger("experiment", log_dir=str(Path(config.training.output_dir) / "logs"))
    log.info(f"Experiment: {config.name}")
    log.info(f"Description: {config.description}")
    log.info(f"Model: {config.model.name}")
    log.info(f"Config: {config}")

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"Name:     {config.name}")
        print(f"Model:    {config.model.name}")
        print(f"  dtype:  {config.model.torch_dtype}")
        print(f"  attn:   {config.model.attn_implementation}")
        print(f"  chat:   {config.model.use_chat_template}")
        print(f"Dataset:  {config.dataset.name}")
        print(f"  train:  {config.dataset.train_split}")
        print(f"  eval:   {config.dataset.eval_split}")
        print(f"Entropy:  {config.entropy.strategy}")
        print(f"KL:       {config.kl.schedule} ({config.kl.initial_coeff} -> {config.kl.final_coeff})")
        print(f"Reward:   format={config.reward.format_mode}, baseline={config.reward.baseline}")
        print(f"Steps:    {config.training.max_steps}")
        print(f"Seq len:  {config.training.max_seq_len}")
        print(f"Batch:    {config.training.batch_size} x {config.training.gradient_accumulation_steps} accum")
        return

    # Load dataset (nvidia/Nemotron-Agentic-v1)
    log.info(f"Loading dataset: {config.dataset.name}")
    train_samples, eval_samples = load_nemotron_data(config)
    log.info(f"Train: {len(train_samples)} samples, Eval: {len(eval_samples)} samples")

    # Collect unique tool names across all training samples
    all_tool_names = set()
    for s in train_samples:
        all_tool_names.update(s.tool_names)
    log.info(f"Unique tools in training data: {len(all_tool_names)} — {sorted(all_tool_names)[:20]}...")

    # Load model & tokenizer
    model, ref_model, tokenizer = load_model_and_tokenizer(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create trainer — pass available tools for reward scoring
    trainer = GRPOTrainer(config)
    trainer.reward_fn.available_tools = sorted(all_tool_names)
    trainer.setup_model(model, ref_model, tokenizer, optimizer)

    # --- vLLM server (optional) ---
    vllm_proc = _maybe_launch_vllm_server(config)
    vllm_client = _setup_vllm_client(config)
    if vllm_client is not None:
        trainer.set_vllm_client(vllm_client)

    # Init wandb
    if config.training.wandb_project:
        trainer.reward_logger.init_wandb(
            project=config.training.wandb_project,
            run_name=config.training.wandb_run_name or config.name,
            config=asdict(config),
        )

    # Accuracy scorer
    from entropy_reward.data.accuracy import compute_accuracy

    # Training loop
    batch_size = config.training.batch_size
    log_interval = config.metrics.log_interval
    eval_interval = config.metrics.eval_interval
    total_steps = config.training.max_steps
    eff_batch = batch_size * config.training.gradient_accumulation_steps

    log.info("=" * 72)
    log.info("TRAINING START")
    log.info(f"  Total steps:            {total_steps}")
    log.info(f"  Batch size:             {batch_size}")
    log.info(f"  Group size:             {config.training.group_size}")
    log.info(f"  Gradient accum steps:   {config.training.gradient_accumulation_steps}")
    log.info(f"  Effective batch:        {eff_batch}")
    log.info(f"  Gradient checkpointing: {config.training.gradient_checkpointing}")
    log.info(f"  Max seq len:            {config.training.max_seq_len}")
    log.info(f"  LR:                     {config.training.learning_rate}")
    log.info(f"  Log every:              {log_interval} steps")
    log.info(f"  Eval every:             {eval_interval} steps")
    log.info(f"  Checkpoint every:       {config.training.checkpoint_interval} steps")
    log.info(f"  Entropy strategy:       {config.entropy.strategy}")
    log.info(f"  Reward format mode:     {config.reward.format_mode}")
    log.info(f"  Baseline:               {config.reward.baseline}")
    log.info(f"  KL schedule:            {config.kl.schedule} ({config.kl.initial_coeff} -> {config.kl.final_coeff})")
    if vllm_client is not None:
        log.info(f"  vLLM:                   ENABLED ({config.vllm.base_url})")
    else:
        log.info(f"  vLLM:                   disabled (sequential HF generation)")
    if torch.cuda.is_available():
        log.info(f"  GPU memory (pre-train): {_gpu_mem_str()}")
    log.info("=" * 72)

    import time as _time
    _train_start = _time.time()
    _interval_start = _time.time()
    _AGG_KEYS = [
        "loss", "pg_loss", "entropy_bonus", "kl_penalty", "kl_raw",
        "token_entropy", "action_entropy", "self_bleu", "uniqueness",
        "pattern_repetition", "r_format", "r_tool", "r_acc", "r_total",
        "calls_per_step",
    ]
    _interval_metrics: dict[str, list[float]] = {k: [] for k in _AGG_KEYS}
    _step_start = _time.time()

    for step in range(1, total_steps + 1):
        _step_start = _time.time()

        # Sample batch from Nemotron data
        indices = torch.randint(0, len(train_samples), (batch_size,))
        batch_samples = [train_samples[i] for i in indices]
        batch_prompts = [s.prompt for s in batch_samples]

        if step == 1:
            log.info(
                f"  Step 1: generating {batch_size}×{config.training.group_size}="
                f"{batch_size * config.training.group_size} rollouts "
                f"(up to {config.model.generation.max_new_tokens} tok each)…"
            )

        # Accuracy will be computed after generation inside trainer,
        # for now pass 0 (trainer.train_step handles R_acc externally)
        batch_acc = [0.0] * batch_size

        # Train step
        metrics = trainer.train_step(batch_prompts, batch_acc, step)

        step_time = _time.time() - _step_start

        # Accumulate metrics for averaging
        for k in _AGG_KEYS:
            if k in metrics:
                _interval_metrics[k].append(metrics[k])

        # Short per-step progress (every step when not at dashboard interval)
        if step % log_interval != 0:
            if step <= 3 or step % 10 == 0:
                log.info(
                    f"  step {step}/{total_steps}  "
                    f"{step_time:.1f}s/step  "
                    f"loss={metrics['loss']:.4f}  "
                    f"R={metrics['r_total']:.3f}"
                )

        # Verbose log every log_interval steps
        if step % log_interval == 0:
            elapsed = _time.time() - _train_start
            interval_time = _time.time() - _interval_start
            steps_per_sec = log_interval / interval_time if interval_time > 0 else 0
            eta_secs = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0

            # Compute averages over interval
            avg = {}
            for k, vals in _interval_metrics.items():
                avg[k] = sum(vals) / len(vals) if vals else 0.0

            log.info("")
            log.info("=" * 80)
            log.info(
                f"  STEP {step}/{total_steps}  |  "
                f"elapsed {elapsed:.0f}s  |  "
                f"{steps_per_sec:.2f} step/s  |  "
                f"ETA {eta_secs/60:.0f}min"
            )
            log.info("=" * 80)

            # ── Loss breakdown ──
            log.info(
                f"  Loss:       total={avg['loss']:.4f}  "
                f"pg={avg['pg_loss']:.4f}  "
                f"H_bonus={avg['entropy_bonus']:.5f}  "
                f"KL_pen={avg['kl_penalty']:.4f}"
            )

            # ── Decomposed rewards (interval average) ──
            log.info(
                f"  Reward avg: R_fmt={avg['r_format']:.3f}  "
                f"R_tool={avg['r_tool']:.3f}  "
                f"R_acc={avg['r_acc']:.3f}  "
                f"R_total={avg['r_total']:.3f}"
            )

            # ── Entropy ──
            log.info(
                f"  Entropy:    H_token={avg['token_entropy']:.4f}  "
                f"H_action={avg['action_entropy']:.4f}  "
                f"coeff={metrics.get('entropy_coeff', 0):.5f}"
            )

            # ── KL ──
            log.info(
                f"  KL:         raw={avg['kl_raw']:.4f}  "
                f"coeff={metrics.get('kl_coeff', 0):.4f}  "
                f"penalty={avg['kl_penalty']:.4f}"
            )

            # ── Diversity ──
            log.info(
                f"  Diversity:  self-BLEU={avg['self_bleu']:.4f}  "
                f"uniqueness={avg['uniqueness']:.4f}  "
                f"pattern_rep={avg['pattern_repetition']:.4f}"
            )

            # ── Advantage distribution (last step in interval) ──
            log.info(
                f"  Advantage:  mean={metrics.get('adv_mean', 0):.4f}  "
                f"std={metrics.get('adv_std', 0):.4f}  "
                f"min={metrics.get('adv_min', 0):.4f}  "
                f"max={metrics.get('adv_max', 0):.4f}  "
                f"skew={metrics.get('adv_skew', 0):.3f}  "
                f"kurt={metrics.get('adv_kurtosis', 0):.3f}  "
                f"frac+={metrics.get('adv_frac_positive', 0):.2f}"
            )

            # ── Tool call stats ──
            log.info(
                f"  Tools:      calls/step={avg['calls_per_step']:.2f}  "
                f"generated={metrics.get('n_generated', 0)}"
            )
            # Top-5 tool frequencies from last step
            tf = metrics.get("tool_frequencies", {})
            if tf:
                top5 = sorted(tf.items(), key=lambda x: -x[1])[:5]
                tf_str = "  ".join(f"{t}={f:.2f}" for t, f in top5)
                log.info(f"  Top tools:  {tf_str}")

            # ── GPU memory ──
            if torch.cuda.is_available():
                log.info(f"  GPU mem:    {_gpu_mem_str()}")

            # ── Prompt & Rollout samples (last step) ──
            rollout_details = metrics.get("rollout_details", [])
            if rollout_details:
                log.info("-" * 80)
                log.info("  LAST-STEP PROMPTS & ROLLOUTS:")
                current_prompt = -1
                for rd in rollout_details:
                    pi = rd["prompt_idx"]
                    gi = rd["rollout_idx"]
                    if pi != current_prompt:
                        current_prompt = pi
                        log.info(f"  ┌─ Prompt {pi}: {rd['prompt_snippet']}")
                    log.info(
                        f"  │ [{gi}] R_fmt={rd['r_format']:.1f} R_tool={rd['r_tool']:.1f} "
                        f"R_acc={rd['r_acc']:.2f} R_total={rd['r_total']:.2f} "
                        f"adv={rd['advantage']:+.3f} len={rd['output_len']}"
                    )
                    log.info(f"  │     ▸ {rd['output_snippet']}")
                log.info(f"  └─")
            log.info("=" * 80)

            # Reset interval tracking
            _interval_start = _time.time()
            for k in _AGG_KEYS:
                _interval_metrics[k] = []

        # Eval
        if step % eval_interval == 0:
            log.info("")
            log.info(f"{'~' * 72}")
            log.info(f"  EVAL at step {step}")
            log.info(f"{'~' * 72}")
            eval_prompts = [s.prompt for s in eval_samples[:50]]
            eval_results = trainer.run_eval(eval_prompts, step)

            if "ood" in eval_results:
                for r in eval_results["ood"]:
                    log.info(
                        f"  OOD [{r['name']}]: format_pass={r['format_rate']:.3f}  "
                        f"tool_pass={r['tool_rate']:.3f}"
                    )
            if "metamorphic" in eval_results:
                for r in eval_results["metamorphic"]:
                    log.info(f"  Metamorphic [{r['transform']}]: consistency={r['consistency']:.3f}")
            if "redteam" in eval_results:
                for r in eval_results["redteam"]:
                    log.info(
                        f"  RedTeam [{r['exploit']}]: success={r['rate']:.3f}  "
                        f"fp={r['fp']:.3f}"
                    )
            if "recovery_speed" in eval_results:
                log.info(f"  Recovery speed: {eval_results['recovery_speed']:.2f}")
            log.info(f"{'~' * 72}")
            log.info("")

        # Stop condition
        if metrics["should_stop"]:
            log.warning("")
            log.warning("!" * 72)
            log.warning(f"  EARLY STOP at step {step}: {metrics['stop_reason']}")
            log.warning("!" * 72)
            trainer.save_checkpoint(step)
            break

        # Periodic checkpoint
        if step % config.training.checkpoint_interval == 0:
            log.info(f"  [Checkpoint] Saving at step {step}...")
            trainer.save_checkpoint(step)
            log.info(f"  [Checkpoint] Saved to {config.training.output_dir}/checkpoint-{step}")

    # Final save
    total_time = _time.time() - _train_start
    trainer.save_checkpoint(step)
    trainer.close()

    # Cleanup vLLM
    if vllm_client is not None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(vllm_client.close())
        except Exception:
            pass
    if vllm_proc is not None:
        log.info("Terminating vLLM server...")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=10)
        except Exception:
            vllm_proc.kill()

    log.info("")
    log.info("=" * 72)
    log.info(f"  TRAINING COMPLETE — {step} steps in {total_time:.0f}s ({total_time/60:.1f}min)")
    if torch.cuda.is_available():
        log.info(f"  Peak GPU memory: {_gpu_mem_str()}")
    log.info("=" * 72)


if __name__ == "__main__":
    main()
