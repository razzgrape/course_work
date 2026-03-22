"""Пайплайн экспериментов."""

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config.settings import settings
from data.loader import DataLoader
from llm_client.ollama_client import OllamaClient
from llm_client.mcp_bridge import McpBridge
from llm_client.prompts import get_system_prompt, make_user_message
from experiments.ambiguity import AmbiguityAnalyzer

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Результат одного прогона (один текст, один режим).

    Attributes:
        task: Задача (pos, lemma).
        mode: Режим (llm_only, tool_assisted).
        text: Исходный текст.
        expected: Эталонная разметка.
        predicted: Предсказание (распарсенный JSON или сырой текст).
        raw_response: Полный ответ LLM.
        llm_time: Время ответа LLM в секундах.
        tool_time: Время выполнения MCP-инструмента (0 для llm_only).
        success: Удалось ли получить и распарсить ответ.
        error: Сообщение об ошибке.
    """

    task: str
    mode: str
    text: str
    expected: list[dict]
    predicted: list[dict] = field(default_factory=list)
    raw_response: str = ""
    llm_time: float = 0.0
    tool_time: float = 0.0
    success: bool = True
    error: str | None = None
    ambiguity_ratio: float = 0.0
    ambiguous_tokens: list[str] = field(default_factory=list)


def parse_llm_json(text: str) -> list[dict]:
    """Попытаться распарсить JSON из ответа LLM"""
    text = text.strip()

    if not text:
        return []

    if "```" in text:
        lines = text.split("\n")
        cleaned = []
        inside_block = False
        for line in lines:
            if line.strip().startswith("```"):
                inside_block = not inside_block
                continue
            if inside_block:
                cleaned.append(line)
        text = "\n".join(cleaned).strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


class ExperimentRunner:
    """Оркестратор экспериментов.

    Прогоняет тексты через оба режима и собирает результаты.

    Example:
        >>> runner = ExperimentRunner()
        >>> results = asyncio.run(runner.run(task="pos", n=10))
        >>> print(len(results))
    """

    def __init__(
        self,
        model: str | None = None,
        python_command: str = "python3",
    ) -> None:
        self._model = model
        self._python_command = python_command
        self._loader = DataLoader()

    async def run(
        self,
        task: str,
        n: int | None = None,
        modes: list[str] | None = None,
        ambiguous_only: bool = False,
        min_ambiguous: int = 2,
    ) -> list[ExperimentResult]:
        """Запустить эксперимент.

        Args:
            task: Задача — "pos" или "lemma".
            n: Количество примеров.
            modes: Режимы — ["llm_only", "tool_assisted"] по умолчанию.
            ambiguous_only: Брать только неоднозначные предложения.
            min_ambiguous: Минимум неоднозначных токенов в предложении.

        Returns:
            Список ExperimentResult для всех примеров и режимов.
        """
        n = n or settings.max_samples
        modes = modes or ["llm_only", "tool_assisted"]

        load_n = n * 5 if ambiguous_only else n

        if task == "pos":
            samples = self._loader.load_pos_samples(n=load_n)
        elif task == "lemma":
            samples = self._loader.load_lemma_samples(n=load_n)
        else:
            raise ValueError(f"Неизвестная задача: {task}")

        ambiguity_map = {}
        if ambiguous_only:
            analyzer = AmbiguityAnalyzer()
            filtered = analyzer.filter_ambiguous(
                samples, min_ambiguous=min_ambiguous,
            )
            for sample, info in filtered:
                ambiguity_map[sample.text] = info
            samples = [sample for sample, info in filtered]

            logger.info(
                "Фильтр неоднозначности: %d -> %d предложений "
                "(min_ambiguous=%d)",
                load_n, len(samples), min_ambiguous,
            )

        if len(samples) > n:
            samples = samples[:n]

        logger.info(
            "Запуск эксперимента: task=%s, n=%d, modes=%s, ambiguous_only=%s",
            task, len(samples), modes, ambiguous_only,
        )

        results = []

        if "llm_only" in modes:
            logger.info("--- Режим: LLM-only ---")
            llm_only_results = self._run_llm_only(task, samples, ambiguity_map)
            results.extend(llm_only_results)

        if "tool_assisted" in modes:
            logger.info("--- Режим: LLM+MCP (tool_assisted) ---")
            tool_results = await self._run_tool_assisted(task, samples, ambiguity_map)
            results.extend(tool_results)

        self._save_results(results, task)

        return results

    def _run_llm_only(
        self,
        task: str,
        samples: list,
        ambiguity_map: dict | None = None,
    ) -> list[ExperimentResult]:
        """Прогнать все примеры в режиме LLM-only."""
        ambiguity_map = ambiguity_map or {}
        client_kwargs = {}
        if self._model:
            client_kwargs["model"] = self._model

        client = OllamaClient(**client_kwargs)
        system_prompt = get_system_prompt("llm_only", task)

        results = []

        for i, sample in enumerate(samples):
            user_msg = make_user_message(task, sample.text)

            amb_info = ambiguity_map.get(sample.text)
            amb_ratio = amb_info.ambiguity_ratio if amb_info else 0.0
            amb_tokens = (
                [t.token for t in amb_info.ambiguous_tokens]
                if amb_info else []
            )

            logger.info(
                "[LLM-only] %d/%d: %s...",
                i + 1, len(samples), sample.text[:50],
            )

            start = time.perf_counter()
            try:
                response = client.chat(user_msg, system_prompt)
                llm_time = time.perf_counter() - start

                predicted = parse_llm_json(response.content)

                results.append(ExperimentResult(
                    task=task,
                    mode="llm_only",
                    text=sample.text,
                    expected=sample.tokens,
                    predicted=predicted,
                    raw_response=response.content,
                    llm_time=llm_time,
                    success=len(predicted) > 0,
                    error=None if predicted else "Не удалось распарсить JSON",
                    ambiguity_ratio=amb_ratio,
                    ambiguous_tokens=amb_tokens,
                ))

            except Exception as e:
                llm_time = time.perf_counter() - start
                logger.error("[LLM-only] Ошибка: %s", e)

                results.append(ExperimentResult(
                    task=task,
                    mode="llm_only",
                    text=sample.text,
                    expected=sample.tokens,
                    llm_time=llm_time,
                    success=False,
                    error=str(e),
                ))

        client.close()
        return results

    async def _run_tool_assisted(
        self,
        task: str,
        samples: list,
        ambiguity_map: dict | None = None,
    ) -> list[ExperimentResult]:
        """Прогнать все примеры в режиме LLM+MCP."""
        ambiguity_map = ambiguity_map or {}
        client_kwargs = {}
        if self._model:
            client_kwargs["model"] = self._model

        client = OllamaClient(**client_kwargs)
        system_prompt = get_system_prompt("tool_assisted", task)

        results = []

        async with McpBridge(python_command=self._python_command) as bridge:
            tools = await bridge.list_tools()
            logger.info("MCP инструменты: %s", [t["name"] for t in tools])

            for i, sample in enumerate(samples):
                user_msg = make_user_message(task, sample.text)

                amb_info = ambiguity_map.get(sample.text)
                amb_ratio = amb_info.ambiguity_ratio if amb_info else 0.0
                amb_tokens = (
                    [t.token for t in amb_info.ambiguous_tokens]
                    if amb_info else []
                )

                logger.info(
                    "[LLM+MCP] %d/%d: %s...",
                    i + 1, len(samples), sample.text[:50],
                )

                start = time.perf_counter()
                try:
                    response = client.chat_with_tools(user_msg, system_prompt)
                    llm_time = time.perf_counter() - start

                    if response.tool_calls:
                        final, tool_results = await bridge.execute_and_respond(
                            client, response, system_prompt, user_msg,
                        )

                        tool_time = sum(
                            tr.execution_time for tr in tool_results
                        )

                        tool_output = (
                            tool_results[0].result if tool_results else ""
                        )
                        predicted = parse_llm_json(tool_output)

                        results.append(ExperimentResult(
                            task=task,
                            mode="tool_assisted",
                            text=sample.text,
                            expected=sample.tokens,
                            predicted=predicted,
                            raw_response=final.content,
                            llm_time=llm_time,
                            tool_time=tool_time,
                            success=len(predicted) > 0,
                            ambiguity_ratio=amb_ratio,
                            ambiguous_tokens=amb_tokens,
                        ))
                    else:
                        predicted = parse_llm_json(response.content)

                        results.append(ExperimentResult(
                            task=task,
                            mode="tool_assisted",
                            text=sample.text,
                            expected=sample.tokens,
                            predicted=predicted,
                            raw_response=response.content,
                            llm_time=llm_time,
                            success=False,
                            error="Модель не вызвала инструмент",
                            ambiguity_ratio=amb_ratio,
                            ambiguous_tokens=amb_tokens,
                        ))

                except Exception as e:
                    llm_time = time.perf_counter() - start
                    logger.error("[LLM+MCP] Ошибка: %s", e)

                    results.append(ExperimentResult(
                        task=task,
                        mode="tool_assisted",
                        text=sample.text,
                        expected=sample.tokens,
                        llm_time=llm_time,
                        success=False,
                        error=str(e),
                    ))

        client.close()
        return results

    def _save_results(
        self,
        results: list[ExperimentResult],
        task: str,
    ) -> None:
        """Сохранить сырые результаты в JSON."""
        settings.ensure_dirs()

        is_ambiguous = any(r.ambiguity_ratio > 0 for r in results)
        suffix = "_ambiguous" if is_ambiguous else ""
        output_path = settings.raw_results_dir / f"{task}{suffix}_results.json"

        data = [asdict(r) for r in results]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Результаты сохранены в %s", output_path)

async def main():
    parser = argparse.ArgumentParser(
        description="Запуск экспериментов LLM-only vs LLM+MCP"
    )
    parser.add_argument(
        "--task",
        choices=["pos", "lemma"],
        required=True,
        help="Задача: pos или lemma",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Количество примеров (по умолчанию 10)",
    )
    parser.add_argument(
        "--mode",
        choices=["llm_only", "tool_assisted", "both"],
        default="both",
        help="Режим: llm_only, tool_assisted или both",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Модель Ollama (по умолчанию из конфига)",
    )
    parser.add_argument(
        "--ambiguous-only",
        action="store_true",
        help="Только предложения с морфологической неоднозначностью",
    )
    parser.add_argument(
        "--min-ambiguous",
        type=int,
        default=2,
        help="Минимум неоднозначных токенов в предложении (по умолчанию 2)",
    )

    args = parser.parse_args()

    modes = (
        ["llm_only", "tool_assisted"]
        if args.mode == "both"
        else [args.mode]
    )

    runner = ExperimentRunner(model=args.model)
    results = await runner.run(
        task=args.task,
        n=args.n,
        modes=modes,
        ambiguous_only=args.ambiguous_only,
        min_ambiguous=args.min_ambiguous,
    )

    for mode in modes:
        mode_results = [r for r in results if r.mode == mode]
        success = sum(1 for r in mode_results if r.success)
        total = len(mode_results)
        avg_llm = (
            sum(r.llm_time for r in mode_results) / total
            if total
            else 0
        )

        print(f"\n{'='*50}")
        print(f"Режим: {mode}")
        print(f"Успешных: {success}/{total}")
        print(f"Среднее время LLM: {avg_llm:.2f} сек")

        if mode == "tool_assisted":
            avg_tool = (
                sum(r.tool_time for r in mode_results) / total
                if total
                else 0
            )
            print(f"Среднее время MCP: {avg_tool:.3f} сек")


if __name__ == "__main__":
    asyncio.run(main())