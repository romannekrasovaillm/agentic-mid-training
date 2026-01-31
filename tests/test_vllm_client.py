"""Tests for the vLLM inference client."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from entropy_reward.inference.vllm_client import VLLMClient


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestVLLMClientInit:
    def test_defaults(self):
        client = VLLMClient()
        assert client.base_url == "http://localhost:8000"
        assert client.max_new_tokens == 512
        assert client.temperature == 0.7
        assert client.top_p == 0.9
        assert client.use_chat_api is True
        assert client.max_retries == 3

    def test_custom_params(self):
        client = VLLMClient(
            base_url="http://gpu-server:9000",
            model_name="test/model",
            max_new_tokens=256,
            temperature=0.5,
            use_chat_api=False,
        )
        assert client.base_url == "http://gpu-server:9000"
        assert client.model_name == "test/model"
        assert client.max_new_tokens == 256
        assert client.temperature == 0.5
        assert client.use_chat_api is False

    def test_trailing_slash_stripped(self):
        client = VLLMClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"


class TestVLLMClientGenerate:
    """Test generation methods using mocked HTTP responses."""

    @pytest.fixture
    def client(self):
        return VLLMClient(
            base_url="http://localhost:8000",
            model_name="test-model",
            max_new_tokens=64,
            temperature=0.7,
        )

    def test_generate_one_chat(self, client):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hello, world!"}}]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = _run(client._generate_one_chat("Test prompt"))
        assert result == "Hello, world!"

    def test_generate_one_completion(self, client):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"text": "<think>reasoning</think><answer>42</answer>"}]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = _run(client._generate_one_completion("What is 6*7?"))
        assert "42" in result

    def test_generate_batch_async(self, client):
        call_prompts = []

        async def mock_chat(prompt):
            call_prompts.append(prompt)
            return f"Reply to: {prompt}"

        client._generate_one_chat = mock_chat
        results = _run(client._generate_batch_async(
            ["Prompt 1", "Prompt 2", "Prompt 3"]
        ))
        assert len(results) == 3
        assert results[0] == "Reply to: Prompt 1"
        assert results[1] == "Reply to: Prompt 2"
        assert results[2] == "Reply to: Prompt 3"

    def test_generate_batch_sync(self, client):
        async def mock_batch(prompts):
            return [f"Reply to: {p}" for p in prompts]

        client._generate_batch_async = mock_batch
        results = client.generate_batch(["A", "B"])
        assert len(results) == 2
        assert results[0] == "Reply to: A"
        assert results[1] == "Reply to: B"

    def test_generate_single(self, client):
        async def mock_batch(prompts):
            return [f"Reply: {p}" for p in prompts]

        client._generate_batch_async = mock_batch
        result = client.generate("Hello")
        assert result == "Reply: Hello"

    def test_error_returns_empty(self, client):
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)
        client._session = mock_session
        client.max_retries = 1

        result = _run(client._generate_one_chat("prompt"))
        assert result == ""


class TestVLLMClientHealth:

    def test_health_check_success(self):
        client = VLLMClient()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False
        client._session = mock_session

        result = _run(client.health_check())
        assert result is True

    def test_health_check_failure(self):
        client = VLLMClient()
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))
        mock_session.closed = False
        client._session = mock_session

        result = _run(client.health_check())
        assert result is False


class TestVLLMConfig:
    def test_config_dataclass(self):
        from entropy_reward.utils.config_loader import VLLMConfig

        cfg = VLLMConfig()
        assert cfg.enabled is True
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.tensor_parallel_size == 1
        assert cfg.gpu_memory_utilization == 0.30
        assert cfg.max_model_len == 4096
        assert cfg.launch_server is True
        assert cfg.enforce_eager is True

    def test_config_in_experiment(self):
        from entropy_reward.utils.config_loader import ExperimentConfig

        cfg = ExperimentConfig()
        assert hasattr(cfg, "vllm")
        assert cfg.vllm.enabled is True

    def test_config_from_yaml(self, tmp_path):
        from entropy_reward.utils.config_loader import load_config

        yaml_content = """
name: "test-vllm"
vllm:
  enabled: true
  base_url: "http://gpu:9000"
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.5
"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)

        cfg = load_config(str(config_file))
        assert cfg.vllm.enabled is True
        assert cfg.vllm.base_url == "http://gpu:9000"
        assert cfg.vllm.tensor_parallel_size == 2
        assert cfg.vllm.gpu_memory_utilization == 0.5
