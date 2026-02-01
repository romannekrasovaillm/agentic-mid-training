"""Main GRPO trainer with entropy reward experiments.

Integrates:
- Entropy strategies (constant / adaptive)
- KL anchoring with schedule
- Decomposed rewards with baselines
- Full metrics + logging pipeline
- Stop conditions
- Periodic eval harness
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from entropy_reward.strategies import ConstantEntropyBonus, AdaptiveEntropyBonus, KLAnchor
from entropy_reward.rewards import (
    DecomposedReward,
    GroupNormBaseline,
    LeaveOneOutBaseline,
    JackknifeBaseline,
)
from entropy_reward.metrics import (
    TokenEntropy,
    ActionEntropy,
    SelfBLEU,
    TrajectoryUniqueness,
    PatternRepetition,
    AdvantageStatistics,
)
from entropy_reward.eval import OODEvaluator, MetamorphicTester, RedTeamGenerator
from entropy_reward.monitoring import (
    CollapseDetector,
    HackingDetector,
    AdvantageDriftDetector,
    StopConditionAggregator,
)
from entropy_reward.utils.logging_utils import RewardLogger, StepLog
from entropy_reward.utils.config_loader import ExperimentConfig, ModelConfig
from entropy_reward.data import AgenticSample
from entropy_reward.data.accuracy import compute_accuracy

logger = logging.getLogger(__name__)

# Max chars to show for prompt/rollout snippets in logs
_SNIPPET_HEAD = 120
_SNIPPET_TAIL = 80


def _snippet(text: str, head: int = _SNIPPET_HEAD, tail: int = _SNIPPET_TAIL) -> str:
    """Return head...tail preview of text for logging."""
    text = text.replace("\n", " ↵ ").strip()
    if len(text) <= head + tail + 5:
        return text
    return text[:head] + " … " + text[-tail:]


class GRPOTrainer:
    """GRPO trainer with entropy-stability experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Entropy strategy ---
        if config.entropy.strategy == "constant":
            self.entropy_bonus = ConstantEntropyBonus(config.entropy.constant_bonus)
        else:
            self.entropy_bonus = AdaptiveEntropyBonus(
                target_entropy=config.entropy.adaptive_target_entropy,
                alpha=config.entropy.adaptive_alpha,
                min_coeff=config.entropy.adaptive_min_coeff,
                max_coeff=config.entropy.adaptive_max_coeff,
                diversity_ema_decay=config.entropy.diversity_ema_decay,
                entropy_drop_threshold=config.entropy.entropy_drop_threshold,
            )

        # --- KL anchor ---
        self.kl_anchor = KLAnchor(
            initial_coeff=config.kl.initial_coeff,
            final_coeff=config.kl.final_coeff,
            schedule=config.kl.schedule,
            warmup_steps=config.kl.warmup_steps,
            total_steps=config.training.max_steps,
            step_milestones=config.kl.step_milestones,
            step_gamma=config.kl.step_gamma,
        ) if config.kl.enabled else None

        # --- Reward ---
        self.reward_fn = DecomposedReward(
            format_mode=config.reward.format_mode,
            format_weight=config.reward.format_weight,
            tool_weight=config.reward.tool_weight,
            accuracy_weight=config.reward.accuracy_weight,
            partial_tag_credit=config.reward.partial_tag_credit,
            partial_structure_credit=config.reward.partial_structure_credit,
            partial_full_credit=config.reward.partial_full_credit,
            multiplicative_format=config.reward.multiplicative_format,
            format_floor=config.reward.format_floor,
        )

        # --- Baseline ---
        separate = config.reward.separate_baselines
        if config.reward.baseline == "leave_one_out":
            self.baseline = LeaveOneOutBaseline(separate=separate)
        elif config.reward.baseline == "jackknife":
            self.baseline = JackknifeBaseline(separate=separate)
        else:
            self.baseline = GroupNormBaseline(separate=separate)

        # --- Metrics ---
        self.token_entropy = TokenEntropy()
        self.action_entropy = ActionEntropy()
        self.self_bleu = SelfBLEU(n=config.metrics.self_bleu_n)
        self.traj_unique = TrajectoryUniqueness()
        self.pattern_rep = PatternRepetition(window_size=config.metrics.pattern_window)
        self.adv_stats = AdvantageStatistics()

        # --- Eval harness ---
        self.ood_eval = OODEvaluator(datasets=config.eval.ood_datasets) if config.eval.ood_enabled else None
        self.metamorphic = MetamorphicTester(transforms=config.eval.metamorphic_transforms) if config.eval.metamorphic_enabled else None
        self.redteam = RedTeamGenerator(exploit_budget=config.eval.redteam_exploit_budget) if config.eval.redteam_enabled else None

        # --- Stop conditions ---
        self.stop_agg = StopConditionAggregator(
            collapse=CollapseDetector(
                entropy_threshold=config.stop.entropy_collapse_threshold,
                diversity_threshold=config.stop.diversity_collapse_threshold,
                window=config.stop.entropy_collapse_window,
            ),
            hacking=HackingDetector(
                passrate_threshold=config.stop.hacking_passrate_threshold,
                eval_interval=config.stop.hacking_eval_interval,
            ),
            drift=AdvantageDriftDetector(
                drift_threshold=config.stop.advantage_drift_threshold,
                window=config.stop.advantage_drift_window,
            ),
        )

        # --- Logger ---
        self.reward_logger = RewardLogger(
            output_dir=config.training.output_dir,
            run_name=config.name,
        )

        # Model placeholders (set via setup_model)
        self.model = None
        self.ref_model = None
        self.optimizer = None
        self.tokenizer = None
        self.vllm_client = None  # Optional VLLMClient for fast batched generation

    def setup_model(self, model, ref_model, tokenizer, optimizer):
        """Set model, reference model, tokenizer, optimizer."""
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer

    def set_vllm_client(self, client):
        """Attach a VLLMClient for fast batched rollout generation.

        When set, train_step() will use the vLLM server for generation
        instead of sequential HF model.generate() calls.
        """
        self.vllm_client = client
        logger.info("vLLM client attached — generation will use vLLM server")

        # Wire up eval harness generation — use vLLM (fast) instead of HF
        def generate_fn(prompt: str, tools: list[str] | None = None) -> str:
            return client.generate(prompt)

        if self.ood_eval:
            self.ood_eval.set_generate_fn(generate_fn)
        if self.metamorphic:
            self.metamorphic.set_generate_fn(generate_fn)
        if self.redteam:
            self.redteam.set_reward_fn(lambda text: self.reward_fn.compute(text).r_total)

    def _generate(self, prompt: str) -> str:
        """Generate text from current policy using chat template if available."""
        if self.model is None or self.tokenizer is None:
            return ""

        gen_cfg = self.config.model.generation

        # Use chat template for chat models (GigaChat3 etc.)
        if self.config.model.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            # apply_chat_template may return a tensor or a BatchEncoding
            if hasattr(encoded, "input_ids"):
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
            else:
                input_ids = encoded.to(self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                attention_mask = None
            prompt_len = input_ids.shape[1]
        else:
            encoded = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.config.training.max_seq_len,
            ).to(self.device)
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
            prompt_len = input_ids.shape[1]

        # Build explicit attention_mask if not provided (fixes warning
        # when pad_token == eos_token)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
            )
        return self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=False)

    def train_step(
        self,
        samples: list[AgenticSample],
        step: int,
    ) -> dict[str, Any]:
        """Execute one GRPO training step.

        Args:
            samples: batch of AgenticSample (prompt + reference for R_acc)
            step: current global step

        Returns:
            dict of metrics for this step
        """
        group_size = self.config.training.group_size
        prompts = [s.prompt for s in samples]

        # 1) Generate group_size completions per prompt
        total_gen = len(prompts) * group_size
        t_gen_start = time.time()

        if self.vllm_client is not None:
            # ── Fast path: batched generation via vLLM server ──
            expanded_prompts = []
            for prompt in prompts:
                for _ in range(group_size):
                    expanded_prompts.append(prompt)

            logger.info(
                f"    [vLLM] Sending {total_gen} prompts for batched generation..."
            )
            all_texts = self.vllm_client.generate_batch(expanded_prompts)
            gen_elapsed = time.time() - t_gen_start
            logger.info(
                f"    [vLLM] {total_gen} rollouts in {gen_elapsed:.1f}s "
                f"({gen_elapsed / max(total_gen, 1):.2f}s/rollout)"
            )
        else:
            # ── Slow path: sequential HF generation ──
            all_texts = []
            gen_idx = 0
            for prompt in prompts:
                for _ in range(group_size):
                    text = self._generate(prompt)
                    all_texts.append(text)
                    gen_idx += 1
                    if gen_idx % 8 == 0 or gen_idx == total_gen:
                        elapsed = time.time() - t_gen_start
                        logger.info(
                            f"    [gen {gen_idx}/{total_gen}] "
                            f"{elapsed:.1f}s  "
                            f"({elapsed/gen_idx:.2f}s/rollout)"
                        )
            gen_elapsed = time.time() - t_gen_start

        # 1b) Compute R_acc per-rollout by comparing to reference
        acc_tw = self.config.dataset.accuracy_tool_weight
        acc_rw = self.config.dataset.accuracy_response_weight
        all_acc = []
        idx = 0
        for sample in samples:
            for _ in range(group_size):
                result = compute_accuracy(
                    generated_text=all_texts[idx],
                    reference_response=sample.reference_response,
                    reference_tool_calls=sample.reference_tool_calls,
                    tool_weight=acc_tw,
                    response_weight=acc_rw,
                )
                all_acc.append(result.score)
                idx += 1

        # 2) Compute decomposed rewards + diagnostics
        rewards_list, reward_diags = self.reward_fn.compute_batch_with_diagnosis(
            all_texts, all_acc
        )
        r_format = torch.tensor([r.r_format for r in rewards_list], device=self.device)
        r_tool = torch.tensor([r.r_tool for r in rewards_list], device=self.device)
        r_acc = torch.tensor([r.r_acc for r in rewards_list], device=self.device)
        r_total = torch.tensor([r.r_total for r in rewards_list], device=self.device)

        # --- Verbose per-sample logging with WHY for zero rewards ---
        logger.info(
            f"[Step {step}] === REWARD DIAGNOSIS ({len(prompts)} prompts × {group_size} rollouts) ==="
        )
        idx = 0
        for pi, prompt in enumerate(prompts):
            logger.info(f"  ┌─ Prompt {pi}: {_snippet(prompt)}")
            for gi in range(group_size):
                r = rewards_list[idx]
                d = reward_diags[idx]
                logger.info(
                    f"  │  Rollout {gi}: R_fmt={r.r_format:.2f} R_tool={r.r_tool:.2f} "
                    f"R_acc={r.r_acc:.2f} R_total={r.r_total:.2f}"
                )
                logger.info(f"  │    ▸ {_snippet(all_texts[idx])}")
                # Log WHY for each zero component
                if r.r_format == 0.0 or r.r_tool == 0.0:
                    logger.info(f"  │    ✗ {d['reason']}")
                idx += 1
            logger.info(f"  └─")

        # 3) Compute advantages
        if self.config.reward.separate_baselines:
            components = {"format": r_format, "tool": r_tool, "acc": r_acc}
            advantages = self.baseline.compute_advantages(r_total, group_size, components)
        else:
            advantages = self.baseline.compute_advantages(r_total, group_size)

        # 4-9) Chunked forward pass, loss computation, and backward
        # Processing all 32 sequences at once OOMs on large-vocab MoE models
        # (32 × seq_len × 150K vocab × 2 bytes = ~5-10 GiB per logits tensor).
        # Instead, process in mini-batches and accumulate gradients.
        batch_inputs = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.training.max_seq_len,
        ).to(self.device)

        # Pre-compute coefficients (avoid double-counting from per-chunk calls)
        entropy_coeff = self.entropy_bonus.get_coeff()
        kl_coeff = 0.0
        if self.kl_anchor is not None:
            kl_coeff = self.kl_anchor.get_coeff()

        self.optimizer.zero_grad()

        fwd_chunk = 4  # sequences per forward pass chunk
        n_seqs = len(all_texts)
        n_chunks = (n_seqs + fwd_chunk - 1) // fwd_chunk

        # Accumulators for metrics (weighted by chunk fraction)
        agg_pg_loss = 0.0
        agg_entropy_bonus = 0.0
        agg_kl_penalty = 0.0
        agg_kl_raw = 0.0
        agg_raw_entropy = 0.0
        agg_te = 0.0  # token entropy metric

        for ci in range(n_chunks):
            s = ci * fwd_chunk
            e = min(s + fwd_chunk, n_seqs)
            chunk_n = e - s
            w = chunk_n / n_seqs  # weight for gradient scaling

            chunk_inputs = {k: v[s:e] for k, v in batch_inputs.items()}
            chunk_advs = advantages[s:e]

            # Policy forward (with gradients)
            outputs = self.model(**chunk_inputs)
            logits = outputs.logits

            # log_softmax (shared for entropy, KL, and PG loss)
            log_probs = F.log_softmax(logits, dim=-1)

            # Entropy bonus (manual — avoid entropy_bonus.compute which
            # would update AdaptiveEntropyBonus state per-chunk)
            probs = log_probs.exp()
            raw_entropy = -(probs * log_probs).sum(-1).mean()
            chunk_entropy_val = entropy_coeff * raw_entropy

            # KL penalty (manual — avoid kl_anchor.compute which
            # would increment _current_step per-chunk)
            chunk_kl_pen = torch.tensor(0.0, device=self.device)
            chunk_kl_raw = 0.0
            if self.kl_anchor is not None and self.ref_model is not None:
                with torch.no_grad():
                    ref_logits = self.ref_model(**chunk_inputs).logits
                ref_probs = F.softmax(ref_logits, dim=-1)
                kl = F.kl_div(log_probs, ref_probs, reduction="none").sum(dim=-1)
                mask_kl = chunk_inputs.get("attention_mask")
                if mask_kl is not None:
                    kl_val = (kl * mask_kl).sum() / mask_kl.sum().clamp(min=1)
                else:
                    kl_val = kl.mean()
                chunk_kl_pen = kl_coeff * kl_val
                chunk_kl_raw = kl_val.detach().item()
                del ref_logits, ref_probs, kl, kl_val

            # PG loss
            labels = chunk_inputs["input_ids"][:, 1:]
            token_log_probs = torch.gather(
                log_probs[:, :-1, :], 2, labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = chunk_inputs.get("attention_mask", torch.ones_like(labels))[:, 1:]
            per_sample_lp = (
                (token_log_probs * mask[:, :token_log_probs.shape[1]]).sum(dim=1)
                / mask[:, :token_log_probs.shape[1]].sum(dim=1).clamp(min=1)
            )
            chunk_pg_loss = -(per_sample_lp * chunk_advs).mean()

            # Chunk loss → backward (scaled by fraction for proper averaging)
            chunk_loss = chunk_pg_loss - chunk_entropy_val + chunk_kl_pen
            (chunk_loss * w).backward()

            # Accumulate metrics (detached)
            agg_pg_loss += chunk_pg_loss.item() * w
            agg_entropy_bonus += chunk_entropy_val.item() * w
            agg_kl_penalty += chunk_kl_pen.item() * w
            agg_kl_raw += chunk_kl_raw * w
            agg_raw_entropy += raw_entropy.item() * w

            # Token entropy metric (detached, for dashboard)
            agg_te += self.token_entropy.compute(
                logits.detach(), chunk_inputs.get("attention_mask")
            ) * w

            # Free GPU memory before next chunk
            del outputs, logits, log_probs, probs, token_log_probs
            del chunk_loss, chunk_entropy_val, chunk_kl_pen, chunk_pg_loss
            del raw_entropy, per_sample_lp
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Gradient clipping & optimizer step
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.max_grad_norm
        )
        self.optimizer.step()

        # Update entropy bonus state once per train step (not per chunk)
        if isinstance(self.entropy_bonus, AdaptiveEntropyBonus):
            self.entropy_bonus._update_entropy_state(agg_raw_entropy)
            self.entropy_bonus._step += 1

        # Update KL anchor step counter once per train step
        if self.kl_anchor is not None:
            self.kl_anchor._current_step += 1

        # Aggregated scalars for metrics / return dict
        loss_val = agg_pg_loss - agg_entropy_bonus + agg_kl_penalty
        pg_loss_val = agg_pg_loss
        entropy_bonus_val = agg_entropy_bonus
        kl_penalty_val = agg_kl_penalty
        kl_raw_val = agg_kl_raw
        te = agg_te

        # Action entropy
        import re
        actions = []
        for text in all_texts:
            found = re.findall(r"<action>\s*(\w+)", text)
            actions.extend(found if found else ["no_action"])
        ae = self.action_entropy.compute(actions)

        # Diversity
        sb = self.self_bleu.compute(all_texts)
        tu = self.traj_unique.compute(all_texts)
        pr = self.pattern_rep.compute(all_texts)

        # Advantage stats
        astats = self.adv_stats.compute(advantages.detach())

        # Update adaptive entropy with diversity signal
        if isinstance(self.entropy_bonus, AdaptiveEntropyBonus):
            self.entropy_bonus.update_diversity_signal(1.0 - sb)

        # Tool frequencies
        from collections import Counter
        tool_counts = Counter(actions)
        total_actions = sum(tool_counts.values())
        tool_freqs = {t: c / total_actions for t, c in tool_counts.items()}

        # Log
        step_log = StepLog(
            step=step,
            r_format=r_format.mean().item(),
            r_tool=r_tool.mean().item(),
            r_acc=r_acc.mean().item(),
            r_total=r_total.mean().item(),
            token_entropy=te,
            action_entropy=ae,
            entropy_coeff=self.entropy_bonus.get_coeff(),
            kl_div=kl_raw_val,
            kl_coeff=kl_coeff,
            calls_per_step=sum(1 for a in actions if a != "no_action") / len(prompts),
            tool_frequencies=tool_freqs,
            advantage_mean=astats.mean,
            advantage_std=astats.std,
            advantage_min=astats.min,
            advantage_max=astats.max,
            self_bleu=sb,
            trajectory_uniqueness=tu,
            pattern_repetition=pr,
        )
        self.reward_logger.log_step(step_log)

        # --- Stop conditions ---
        stop_signal = self.stop_agg.check(
            entropy=te,
            diversity=1.0 - sb,
            adv_mean=astats.mean,
            adv_std=astats.std,
        )

        # Build per-sample detail list for external verbose logging
        rollout_details = []
        idx = 0
        for pi, prompt in enumerate(prompts):
            for gi in range(group_size):
                r = rewards_list[idx]
                d = reward_diags[idx]
                rollout_details.append({
                    "prompt_idx": pi,
                    "rollout_idx": gi,
                    "prompt_snippet": _snippet(prompt),
                    "output_snippet": _snippet(all_texts[idx]),
                    "output_len": len(all_texts[idx]),
                    "r_format": r.r_format,
                    "r_tool": r.r_tool,
                    "r_acc": r.r_acc,
                    "r_total": r.r_total,
                    "advantage": advantages[idx].item(),
                    "diagnosis": d["reason"],
                    "has_think": d["has_think"],
                    "has_action": d["has_action"],
                    "has_answer": d["has_answer"],
                })
                idx += 1

        return {
            "loss": loss_val,
            "pg_loss": pg_loss_val,
            "entropy_bonus": entropy_bonus_val,
            "kl_penalty": kl_penalty_val,
            "token_entropy": te,
            "action_entropy": ae,
            "entropy_coeff": self.entropy_bonus.get_coeff(),
            "kl_coeff": kl_coeff,
            "kl_raw": kl_raw_val,
            "self_bleu": sb,
            "uniqueness": tu,
            "pattern_repetition": pr,
            "r_format": r_format.mean().item(),
            "r_tool": r_tool.mean().item(),
            "r_acc": r_acc.mean().item(),
            "r_total": r_total.mean().item(),
            "adv_mean": astats.mean,
            "adv_std": astats.std,
            "adv_min": astats.min,
            "adv_max": astats.max,
            "adv_skew": astats.skew,
            "adv_kurtosis": astats.kurtosis,
            "adv_frac_positive": astats.fraction_positive,
            "calls_per_step": sum(1 for a in actions if a != "no_action") / len(prompts),
            "tool_frequencies": tool_freqs,
            "n_generated": len(all_texts),
            "gen_time": gen_elapsed,
            "gen_mode": "vllm" if self.vllm_client is not None else "hf",
            "rollout_details": rollout_details,
            "should_stop": stop_signal.should_stop,
            "stop_reason": stop_signal.reason.value,
        }

    def run_eval(self, test_prompts: list[str], step: int) -> dict[str, Any]:
        """Run full eval harness."""
        results = {}

        if self.ood_eval:
            ood_results = self.ood_eval.evaluate(test_prompts)
            results["ood"] = [
                {"name": r.dataset_name, "format_rate": r.format_pass_rate, "tool_rate": r.tool_pass_rate}
                for r in ood_results
            ]
            recovery = self.ood_eval.recovery_speed()
            if recovery is not None:
                results["recovery_speed"] = recovery

        if self.metamorphic:
            meta_results = self.metamorphic.test(test_prompts)
            results["metamorphic"] = [
                {"transform": r.transform_name, "consistency": r.consistency_rate}
                for r in meta_results
            ]

        if self.redteam:
            rt_results = self.redteam.evaluate(test_prompts)
            results["redteam"] = [
                {"exploit": r.exploit_name, "rate": r.exploit_success_rate, "fp": r.false_positive_rate}
                for r in rt_results
            ]
            # Feed exploit rate to stop conditions
            self.stop_agg.check(exploit_rate=self.redteam.overall_exploit_rate)

        logger.info(f"[Step {step}] Eval results: {results}")
        return results

    def save_checkpoint(self, step: int):
        """Save model and trainer state."""
        ckpt_dir = Path(self.config.training.output_dir) / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(ckpt_dir / "model")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_dir / "tokenizer")

        # Save entropy/KL state
        state = {
            "step": step,
            "entropy_state": self.entropy_bonus.state_dict(),
        }
        if self.kl_anchor:
            state["kl_state"] = self.kl_anchor.state_dict()
        torch.save(state, ckpt_dir / "trainer_state.pt")

    def close(self):
        self.reward_logger.close()
