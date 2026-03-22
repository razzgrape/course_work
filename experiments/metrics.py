"""Модуль метрик для оценки результатов экспериментов"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TokenMetrics:
    """Метрики на уровне токенов.

    Attributes:
        total: Общее количество токенов.
        correct: Количество правильных.
        accuracy: Доля правильных (0.0 — 1.0).
    """

    total: int = 0
    correct: int = 0
    accuracy: float = 0.0


@dataclass
class SentenceMetrics:
    """Метрики на уровне предложений.

    Attributes:
        total: Общее количество предложений.
        fully_correct: Полностью правильные предложения.
        accuracy: Доля полностью правильных.
        parseable: Успешно распарсенных ответов.
        parse_rate: Доля распарсенных.
    """

    total: int = 0
    fully_correct: int = 0
    accuracy: float = 0.0
    parseable: int = 0
    parse_rate: float = 0.0


@dataclass
class ErrorExample:
    """Пример ошибки для анализа.

    Attributes:
        token: Слово, на котором ошибка.
        expected: Эталонное значение.
        predicted: Предсказанное значение.
        sentence: Предложение, в котором ошибка.
    """

    token: str
    expected: str
    predicted: str
    sentence: str


@dataclass
class EvaluationReport:
    """Полный отчёт по одному режиму и задаче.

    Attributes:
        task: Задача (pos, lemma).
        mode: Режим (llm_only, tool_assisted).
        token_metrics: Метрики на уровне токенов.
        sentence_metrics: Метрики на уровне предложений.
        errors: Примеры ошибок (первые N).
        avg_llm_time: Среднее время LLM.
        avg_tool_time: Среднее время MCP-инструмента.
    """

    task: str
    mode: str
    token_metrics: TokenMetrics = field(default_factory=TokenMetrics)
    sentence_metrics: SentenceMetrics = field(default_factory=SentenceMetrics)
    errors: list[ErrorExample] = field(default_factory=list)
    avg_llm_time: float = 0.0
    avg_tool_time: float = 0.0


class Evaluator:
    """Оценка результатов экспериментов.

    Example:
        >>> evaluator = Evaluator()
        >>> reports = evaluator.evaluate_from_file("pos")
        >>> for report in reports:
        ...     print(f"{report.mode}: accuracy={report.token_metrics.accuracy:.3f}")
    """

    def evaluate_from_file(
        self,
        task: str,
        max_errors: int = 20,
    ) -> list[EvaluationReport]:
        """Оценить результаты из сохранённого JSON-файла.

        Args:
            task: Задача — "pos" или "lemma".
            max_errors: Максимум примеров ошибок в отчёте.

        Returns:
            Список EvaluationReport — по одному на режим.
        """
        filepath = settings.raw_results_dir / f"{task}_results.json"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Файл результатов не найден: {filepath}\n"
                f"Сначала запустите эксперимент: "
                f"python -m experiments.runner --task {task}"
            )

        with open(filepath, "r", encoding="utf-8") as f:
            results = json.load(f)

        logger.info("Загружено %d результатов из %s", len(results), filepath)

        reports = []

        modes = set(r["mode"] for r in results)
        for mode in sorted(modes):
            mode_results = [r for r in results if r["mode"] == mode]
            report = self._evaluate_mode(task, mode, mode_results, max_errors)
            reports.append(report)

        return reports

    def _evaluate_mode(
        self,
        task: str,
        mode: str,
        results: list[dict],
        max_errors: int,
    ) -> EvaluationReport:
        """Оценить результаты одного режима."""
        report = EvaluationReport(task=task, mode=mode)

        total_tokens = 0
        correct_tokens = 0
        total_sentences = 0
        fully_correct_sentences = 0
        parseable = 0
        errors = []

        key = "pos" if "pos" in task else "lemma"

        llm_times = []
        tool_times = []

        for result in results:
            total_sentences += 1
            llm_times.append(result.get("llm_time", 0))
            tool_times.append(result.get("tool_time", 0))

            expected = result.get("expected", [])
            predicted = result.get("predicted", [])

            if not predicted:
                continue

            parseable += 1

            pred_map = {}
            for item in predicted:
                token = item.get("token", "")
                value = item.get(key, "")
                if token and value:
                    pred_map[token] = value

            sentence_correct = True

            for exp_item in expected:
                exp_token = exp_item.get("token", "")
                exp_value = exp_item.get(key, "")

                if not exp_token or not exp_value:
                    continue

                total_tokens += 1
                pred_value = pred_map.get(exp_token, "")

                if pred_value.lower() == exp_value.lower():
                    correct_tokens += 1
                else:
                    sentence_correct = False
                    if len(errors) < max_errors:
                        errors.append(ErrorExample(
                            token=exp_token,
                            expected=exp_value,
                            predicted=pred_value or "(отсутствует)",
                            sentence=result.get("text", "")[:80],
                        ))

            if sentence_correct:
                fully_correct_sentences += 1

        report.token_metrics = TokenMetrics(
            total=total_tokens,
            correct=correct_tokens,
            accuracy=correct_tokens / total_tokens if total_tokens > 0 else 0.0,
        )

        report.sentence_metrics = SentenceMetrics(
            total=total_sentences,
            fully_correct=fully_correct_sentences,
            accuracy=(
                fully_correct_sentences / total_sentences
                if total_sentences > 0
                else 0.0
            ),
            parseable=parseable,
            parse_rate=(
                parseable / total_sentences
                if total_sentences > 0
                else 0.0
            ),
        )

        report.errors = errors

        report.avg_llm_time = (
            sum(llm_times) / len(llm_times) if llm_times else 0.0
        )
        report.avg_tool_time = (
            sum(tool_times) / len(tool_times) if tool_times else 0.0
        )

        return report


def print_report(report: EvaluationReport) -> None:
    """Вывести отчёт в читаемом формате."""
    print(f"\n{'='*60}")
    print(f"  Задача: {report.task.upper()}  |  Режим: {report.mode}")
    print(f"{'='*60}")

    tm = report.token_metrics
    sm = report.sentence_metrics

    print(f"\n  Токенная accuracy:     {tm.accuracy:.4f}  ({tm.correct}/{tm.total})")
    print(f"  Предложений полностью: {sm.accuracy:.4f}  ({sm.fully_correct}/{sm.total})")
    print(f"  Распарсено ответов:    {sm.parse_rate:.4f}  ({sm.parseable}/{sm.total})")
    print(f"  Среднее время LLM:     {report.avg_llm_time:.2f} сек")

    if report.avg_tool_time > 0:
        print(f"  Среднее время MCP:     {report.avg_tool_time:.3f} сек")

    if report.errors:
        print(f"\n  Примеры ошибок (первые {len(report.errors)}):")
        print(f"  {'Токен':<20} {'Ожидание':<12} {'Предсказание':<12}")
        print(f"  {'-'*44}")
        for err in report.errors[:10]:
            print(f"  {err.token:<20} {err.expected:<12} {err.predicted:<12}")


def print_comparison(reports: list[EvaluationReport]) -> None:
    """Вывести сравнительную таблицу режимов."""
    if len(reports) < 2:
        for r in reports:
            print_report(r)
        return

    print(f"\n{'='*60}")
    print(f"  СРАВНЕНИЕ: {reports[0].task.upper()}")
    print(f"{'='*60}")
    print(f"\n  {'Метрика':<30} {'LLM-only':<15} {'LLM+MCP':<15}")
    print(f"  {'-'*60}")

    llm_only = next((r for r in reports if r.mode == "llm_only"), None)
    tool = next((r for r in reports if r.mode == "tool_assisted"), None)

    if llm_only and tool:
        print(f"  {'Токенная accuracy':<30} {llm_only.token_metrics.accuracy:<15.4f} {tool.token_metrics.accuracy:<15.4f}")
        print(f"  {'Предложений полностью':<30} {llm_only.sentence_metrics.accuracy:<15.4f} {tool.sentence_metrics.accuracy:<15.4f}")
        print(f"  {'Распарсено ответов':<30} {llm_only.sentence_metrics.parse_rate:<15.4f} {tool.sentence_metrics.parse_rate:<15.4f}")
        print(f"  {'Среднее время LLM (сек)':<30} {llm_only.avg_llm_time:<15.2f} {tool.avg_llm_time:<15.2f}")
        print(f"  {'Среднее время MCP (сек)':<30} {'-':<15} {tool.avg_tool_time:<15.3f}")

        delta = tool.token_metrics.accuracy - llm_only.token_metrics.accuracy
        direction = "+" if delta >= 0 else ""
        print(f"\n  Разница accuracy: {direction}{delta:.4f}")

    for r in reports:
        print_report(r)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Оценка результатов")
    parser.add_argument(
        "--task",
        choices=["pos", "lemma", "pos_ambiguous", "lemma_ambiguous"],
        required=True,
        help="Задача: pos или lemma",
    )
    args = parser.parse_args()

    evaluator = Evaluator()
    reports = evaluator.evaluate_from_file(args.task)
    print_comparison(reports)