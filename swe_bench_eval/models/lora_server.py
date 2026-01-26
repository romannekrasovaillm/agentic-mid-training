#!/usr/bin/env python3
"""
OpenAI-Compatible Server for LoRA-adapted Models

Provides an OpenAI-compatible API endpoint for models with LoRA adapters.
Designed for use with mini-swe-agent via LiteLLM.

Usage:
    python lora_server.py \
        --base-model ai-sage/GigaChat3-10B-A1.8B-bf16 \
        --lora-path ./checkpoints/my-lora-adapter \
        --port 8080
"""

import argparse
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Generator

import torch
from flask import Flask, request, jsonify, Response, stream_with_context
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from threading import Thread

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model and tokenizer
model = None
tokenizer = None
model_name = "lora-model"


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    stop_sequences: List[str] = None


def load_model_with_lora(
    base_model_path: str,
    lora_path: str,
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    merge_lora: bool = False,
) -> tuple:
    """
    Load a base model and apply LoRA adapter.

    Args:
        base_model_path: Path or HF repo ID for the base model
        lora_path: Path to the LoRA adapter directory
        device: Device to load model on
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
        merge_lora: Merge LoRA weights into base model (faster inference, more memory)

    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_path}")
    logger.info(f"Loading LoRA adapter: {lora_path}")

    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs,
    )

    # Load and apply LoRA adapter
    logger.info("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.bfloat16,
    )

    if merge_lora:
        logger.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()

    model.eval()
    logger.info("Model loaded successfully!")

    return model, tokenizer


def format_messages(messages: List[Dict]) -> str:
    """
    Format chat messages using the tokenizer's chat template.
    Falls back to simple concatenation if no template available.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback formatting
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        formatted += "Assistant: "
        return formatted


def generate_response(
    prompt: str,
    config: GenerationConfig,
    stream: bool = False,
) -> Generator[str, None, None] | str:
    """Generate response from the model."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample or config.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if config.temperature > 0:
        generation_kwargs["temperature"] = config.temperature
        generation_kwargs["top_p"] = config.top_p

    if stream:
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs["streamer"] = streamer

        thread = Thread(target=model.generate, kwargs={**inputs, **generation_kwargs})
        thread.start()

        for text in streamer:
            yield text

        thread.join()
    else:
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.json
        messages = data.get("messages", [])
        stream = data.get("stream", False)

        config = GenerationConfig(
            max_new_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.0),
            top_p=data.get("top_p", 1.0),
            do_sample=data.get("temperature", 0.0) > 0,
        )

        prompt = format_messages(messages)

        if stream:
            def generate_stream():
                response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())

                for chunk in generate_response(prompt, config, stream=True):
                    data = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

                # Final chunk
                final_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(
                stream_with_context(generate_stream()),
                mimetype="text/event-stream",
            )
        else:
            response_text = generate_response(prompt, config, stream=False)

            return jsonify({
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(tokenizer.encode(prompt)),
                    "completion_tokens": len(tokenizer.encode(response_text)),
                    "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(response_text)),
                },
            })

    except Exception as e:
        logger.exception("Error in chat completions")
        return jsonify({"error": {"message": str(e), "type": "internal_error"}}), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models."""
    return jsonify({
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": model_name})


def main():
    global model, tokenizer, model_name

    parser = argparse.ArgumentParser(description="LoRA Model Server")
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path or HuggingFace repo ID for base model",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="lora-model",
        help="Name to use for the model in API responses",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run server on",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind server to",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA weights into base model",
    )

    args = parser.parse_args()
    model_name = args.model_name

    # Load model
    model, tokenizer = load_model_with_lora(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        merge_lora=args.merge_lora,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
