#!/usr/bin/env python3
"""
GigaChat Agent Evaluation Script
Оценка агентных способностей модели
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Результат оценки"""
    task_name: str
    score: float
    details: dict


# Тестовые задачи для оценки агентных способностей
AGENT_EVAL_TASKS = [
    {
        "name": "tool_selection",
        "description": "Правильный выбор инструмента",
        "prompts": [
            {
                "prompt": "Мне нужно узнать текущую погоду в Москве. Какой инструмент использовать?",
                "expected_tool": "get_weather",
                "available_tools": ["get_weather", "search_web", "calculate", "send_email"],
            },
            {
                "prompt": "Посчитай сумму: 234 + 567 + 891",
                "expected_tool": "calculate",
                "available_tools": ["get_weather", "search_web", "calculate", "send_email"],
            },
        ],
    },
    {
        "name": "json_format",
        "description": "Генерация валидного JSON для tool calls",
        "prompts": [
            {
                "prompt": "Вызови функцию search с аргументом query='python tutorial'",
                "expected_format": "json",
            },
        ],
    },
    {
        "name": "multi_step_reasoning",
        "description": "Многошаговое рассуждение",
        "prompts": [
            {
                "prompt": "Чтобы забронировать отель, мне нужно: 1) найти отели в городе, 2) проверить цены, 3) выбрать подходящий. Какой первый шаг?",
                "expected_step": "search_hotels",
            },
        ],
    },
    {
        "name": "error_recovery",
        "description": "Восстановление после ошибок",
        "prompts": [
            {
                "prompt": "Предыдущий вызов API вернул ошибку 429 (rate limit). Что делать дальше?",
                "expected_behavior": "retry_with_backoff",
            },
        ],
    },
]


class AgentEvaluator:
    """Класс для оценки агентных способностей"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Загрузка модели: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()

    def generate_response(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Генерация ответа модели"""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def evaluate_tool_selection(self, task: dict) -> EvalResult:
        """Оценка выбора инструмента"""
        correct = 0
        total = len(task["prompts"])
        details = []

        for prompt_data in task["prompts"]:
            tools_str = ", ".join(prompt_data["available_tools"])
            full_prompt = f"""Доступные инструменты: {tools_str}

{prompt_data["prompt"]}

Выбери один инструмент и верни его название."""

            response = self.generate_response(full_prompt)
            expected = prompt_data["expected_tool"]

            is_correct = expected.lower() in response.lower()
            if is_correct:
                correct += 1

            details.append({
                "prompt": prompt_data["prompt"],
                "expected": expected,
                "response": response[:200],
                "correct": is_correct,
            })

        score = correct / total if total > 0 else 0
        return EvalResult(
            task_name="tool_selection",
            score=score,
            details={"results": details, "correct": correct, "total": total},
        )

    def evaluate_json_format(self, task: dict) -> EvalResult:
        """Оценка генерации JSON"""
        valid_json = 0
        total = len(task["prompts"])
        details = []

        for prompt_data in task["prompts"]:
            full_prompt = f"""{prompt_data["prompt"]}

Ответ в формате JSON:"""

            response = self.generate_response(full_prompt)

            # Проверка валидности JSON
            is_valid = False
            try:
                # Пытаемся найти JSON в ответе
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    json.loads(json_match.group())
                    is_valid = True
                    valid_json += 1
            except json.JSONDecodeError:
                pass

            details.append({
                "prompt": prompt_data["prompt"],
                "response": response[:300],
                "valid_json": is_valid,
            })

        score = valid_json / total if total > 0 else 0
        return EvalResult(
            task_name="json_format",
            score=score,
            details={"results": details, "valid": valid_json, "total": total},
        )

    def evaluate_multi_step_reasoning(self, task: dict) -> EvalResult:
        """Оценка многошагового рассуждения"""
        correct = 0
        total = len(task["prompts"])
        details = []

        for prompt_data in task["prompts"]:
            response = self.generate_response(prompt_data["prompt"])
            expected = prompt_data["expected_step"]

            # Проверяем наличие ожидаемого шага в ответе
            is_correct = expected.lower() in response.lower() or "поиск" in response.lower()
            if is_correct:
                correct += 1

            details.append({
                "prompt": prompt_data["prompt"],
                "expected": expected,
                "response": response[:300],
                "correct": is_correct,
            })

        score = correct / total if total > 0 else 0
        return EvalResult(
            task_name="multi_step_reasoning",
            score=score,
            details={"results": details, "correct": correct, "total": total},
        )

    def evaluate_error_recovery(self, task: dict) -> EvalResult:
        """Оценка восстановления после ошибок"""
        correct = 0
        total = len(task["prompts"])
        details = []

        for prompt_data in task["prompts"]:
            response = self.generate_response(prompt_data["prompt"])

            # Проверяем, что модель предлагает повтор с задержкой
            recovery_keywords = ["подожд", "повтор", "retry", "backoff", "пауз", "задерж"]
            is_correct = any(kw in response.lower() for kw in recovery_keywords)
            if is_correct:
                correct += 1

            details.append({
                "prompt": prompt_data["prompt"],
                "response": response[:300],
                "suggests_retry": is_correct,
            })

        score = correct / total if total > 0 else 0
        return EvalResult(
            task_name="error_recovery",
            score=score,
            details={"results": details, "correct": correct, "total": total},
        )

    def run_evaluation(self) -> dict[str, EvalResult]:
        """Запуск полной оценки"""
        results = {}

        for task in AGENT_EVAL_TASKS:
            logger.info(f"Оценка: {task['name']} - {task['description']}")

            if task["name"] == "tool_selection":
                result = self.evaluate_tool_selection(task)
            elif task["name"] == "json_format":
                result = self.evaluate_json_format(task)
            elif task["name"] == "multi_step_reasoning":
                result = self.evaluate_multi_step_reasoning(task)
            elif task["name"] == "error_recovery":
                result = self.evaluate_error_recovery(task)
            else:
                continue

            results[task["name"]] = result
            logger.info(f"  Score: {result.score:.2%}")

        return results


def main():
    parser = argparse.ArgumentParser(description="GigaChat Agent Evaluation")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к модели")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Директория для результатов")
    args = parser.parse_args()

    # Создание директории
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Инициализация evaluator
    evaluator = AgentEvaluator(args.model_path)

    # Запуск оценки
    logger.info("Запуск агентной оценки...")
    results = evaluator.run_evaluation()

    # Сводка
    logger.info("\n" + "=" * 50)
    logger.info("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    logger.info("=" * 50)

    total_score = 0
    for name, result in results.items():
        logger.info(f"{name}: {result.score:.2%}")
        total_score += result.score

    avg_score = total_score / len(results) if results else 0
    logger.info(f"\nСредний балл: {avg_score:.2%}")

    # Сохранение результатов
    output_file = output_dir / "agent_eval_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        output_data = {
            "average_score": avg_score,
            "tasks": {
                name: {
                    "score": result.score,
                    "details": result.details,
                }
                for name, result in results.items()
            },
        }
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"\nРезультаты сохранены: {output_file}")


if __name__ == "__main__":
    main()
