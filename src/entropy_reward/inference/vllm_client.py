"""Async HTTP client for vLLM's OpenAI-compatible API.

Sends batched generation requests so all rollouts in a GRPO group
run in parallel on the vLLM engine instead of sequentially through HF generate().

Typical speedup: 10-30x for batch_size*group_size=32 rollouts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for a running vLLM OpenAI-compatible server."""

    # System message for chat API — tells the model the expected output format.
    # Without this, the model never learns <think>/<action>/<answer> tags and
    # GRPO gets R=0.000 on every step.
    FORMAT_SYSTEM_MSG = (
        "You are a helpful assistant with access to tools. "
        "Always structure your response as follows:\n"
        "1. Wrap reasoning inside <think>...</think> tags.\n"
        "2. To call a tool: <action>tool_name(arg1=value1, arg2=value2)</action>\n"
        "3. To answer directly: <answer>your response</answer>\n"
        "You MUST use <think> tags before every <action> or <answer>."
    )

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model_name: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: float = 300.0,
        max_retries: int = 3,
        use_chat_api: bool = True,
        max_model_len: int = 8192,
        system_message: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_chat_api = use_chat_api
        self.max_model_len = max_model_len
        self.system_message = system_message if system_message is not None else self.FORMAT_SYSTEM_MSG
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Health check ──────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Check if the vLLM server is reachable and ready."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False

    def is_healthy(self) -> bool:
        """Synchronous wrapper for health_check."""
        return asyncio.get_event_loop().run_until_complete(self.health_check())

    def wait_until_ready(self, timeout: float = 300.0, poll_interval: float = 5.0) -> bool:
        """Block until the vLLM server responds to health checks."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                ready = loop.run_until_complete(self.health_check())
                if ready:
                    logger.info("vLLM server is ready")
                    return True
            except Exception:
                pass
            logger.info(
                f"Waiting for vLLM server at {self.base_url} "
                f"({time.time() - start:.0f}s / {timeout:.0f}s)..."
            )
            time.sleep(poll_interval)
        logger.error(f"vLLM server not ready after {timeout:.0f}s")
        return False

    # ── Single completion ─────────────────────────────────────────

    def _parse_input_tokens(self, error_body: str) -> int | None:
        """Extract input_tokens count from vLLM context-length error."""
        import re
        m = re.search(r"has (\d+) input tokens", error_body)
        return int(m.group(1)) if m else None

    async def _generate_one_chat(self, prompt: str) -> str:
        """Generate via /v1/chat/completions endpoint."""
        session = await self._get_session()
        max_tok = self.max_new_tokens

        # Build messages with system prompt for format guidance
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tok,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            try:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    if resp.status == 400:
                        body = await resp.text()
                        input_tok = self._parse_input_tokens(body)
                        if input_tok is not None:
                            available = self.max_model_len - input_tok
                            if available >= 64:
                                max_tok = available
                                logger.info(
                                    f"Prompt has {input_tok} tokens, "
                                    f"capping max_tokens to {max_tok}"
                                )
                                continue
                            else:
                                logger.warning(
                                    f"Prompt too long ({input_tok} tokens), "
                                    f"only {available} left — skipping"
                                )
                                return ""
                        logger.warning(
                            f"vLLM chat error (400): {body[:300]}"
                        )
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            f"vLLM chat completion error (status={resp.status}): {body[:300]}"
                        )
                        continue
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except asyncio.TimeoutError:
                logger.warning(f"vLLM request timeout (attempt {attempt + 1}/{self.max_retries})")
            except Exception as exc:
                logger.warning(f"vLLM request error (attempt {attempt + 1}): {exc}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
        return ""

    async def _generate_one_completion(self, prompt: str) -> str:
        """Generate via /v1/completions endpoint (raw text)."""
        session = await self._get_session()
        max_tok = self.max_new_tokens
        for attempt in range(self.max_retries):
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tok,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            try:
                async with session.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                ) as resp:
                    if resp.status == 400:
                        body = await resp.text()
                        input_tok = self._parse_input_tokens(body)
                        if input_tok is not None:
                            available = self.max_model_len - input_tok
                            if available >= 64:
                                max_tok = available
                                logger.info(
                                    f"Prompt has {input_tok} tokens, "
                                    f"capping max_tokens to {max_tok}"
                                )
                                continue
                            else:
                                logger.warning(
                                    f"Prompt too long ({input_tok} tokens), "
                                    f"only {available} left — skipping"
                                )
                                return ""
                        logger.warning(
                            f"vLLM completion error (400): {body[:300]}"
                        )
                        continue
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            f"vLLM completion error (status={resp.status}): {body[:300]}"
                        )
                        continue
                    data = await resp.json()
                    return data["choices"][0]["text"]
            except asyncio.TimeoutError:
                logger.warning(f"vLLM request timeout (attempt {attempt + 1}/{self.max_retries})")
            except Exception as exc:
                logger.warning(f"vLLM request error (attempt {attempt + 1}): {exc}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
        return ""

    # ── Batched generation ────────────────────────────────────────

    async def _generate_batch_async(self, prompts: list[str]) -> list[str]:
        """Generate completions for a batch of prompts concurrently."""
        gen_fn = self._generate_one_chat if self.use_chat_api else self._generate_one_completion
        tasks = [gen_fn(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Synchronous interface: generate completions for all prompts in parallel.

        This is the main entry point for the trainer. It sends all prompts
        to vLLM concurrently and returns completed texts.

        Args:
            prompts: list of prompt strings (already expanded for group_size,
                     i.e. len = batch_size * group_size).

        Returns:
            list of generated text strings, same length as prompts.
        """
        t0 = time.time()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        results = loop.run_until_complete(self._generate_batch_async(prompts))
        elapsed = time.time() - t0
        logger.info(
            f"vLLM batch generation: {len(prompts)} prompts in {elapsed:.1f}s "
            f"({elapsed / max(len(prompts), 1):.2f}s/prompt)"
        )
        return results

    # ── Convenience ───────────────────────────────────────────────

    def generate(self, prompt: str) -> str:
        """Generate a single completion (sync)."""
        results = self.generate_batch([prompt])
        return results[0] if results else ""
