"""Анализатор морфологической неоднозначности.

Определяет, какие слова в предложении pymorphy3 не может
однозначно классифицировать (несколько вариантов разбора
с высокой уверенностью).

Используется для фильтрации датасета: выбираем предложения
с большим количеством неоднозначных токенов, чтобы проверить,
сможет ли LLM снять неоднозначность с помощью контекста.
"""

import logging
from dataclasses import dataclass, field

import pymorphy3
from razdel import tokenize as razdel_tokenize

logger = logging.getLogger(__name__)


@dataclass
class AmbiguousToken:
    """Неоднозначный токен.

    Attributes:
        token: Слово из текста.
        variants: Список вариантов разбора [(pos, lemma, score), ...].
        top_score: Уверенность лучшего варианта.
        n_variants: Количество вариантов с score > threshold.
    """

    token: str
    variants: list[tuple[str, str, float]] = field(default_factory=list)
    top_score: float = 0.0
    n_variants: int = 0


@dataclass
class AmbiguityInfo:
    """Информация о неоднозначности предложения.

    Attributes:
        text: Текст предложения.
        total_tokens: Общее количество токенов (без пунктуации).
        ambiguous_tokens: Список неоднозначных токенов.
        ambiguity_ratio: Доля неоднозначных токенов.
    """

    text: str
    total_tokens: int = 0
    ambiguous_tokens: list[AmbiguousToken] = field(default_factory=list)
    ambiguity_ratio: float = 0.0


class AmbiguityAnalyzer:
    """Анализатор морфологической неоднозначности.

    Для каждого слова проверяет, сколько вариантов разбора
    pymorphy3 считает правдоподобными. Если несколько вариантов
    имеют высокий score — слово неоднозначное.

    Example:
        >>> analyzer = AmbiguityAnalyzer()
        >>> info = analyzer.analyze("Рабочие стали выходить из цеха")
        >>> for tok in info.ambiguous_tokens:
        ...     print(f"{tok.token}: {tok.variants}")
    """

    def __init__(self, score_threshold: float = 0.1) -> None:
        """
        Args:
            score_threshold: Минимальный score варианта, чтобы
                считать его правдоподобным. Варианты с score
                ниже этого порога игнорируются.
        """
        self._morph = pymorphy3.MorphAnalyzer()
        self._score_threshold = score_threshold

    def analyze(self, text: str) -> AmbiguityInfo:
        """Проанализировать неоднозначность предложения.

        Args:
            text: Текст предложения.

        Returns:
            AmbiguityInfo с информацией о неоднозначных токенах.
        """
        tokens = list(razdel_tokenize(text))

        total = 0
        ambiguous = []

        for tok in tokens:
            word = tok.text

            if not word.isalpha():
                continue

            total += 1
            parses = self._morph.parse(word)

            variants = []
            for p in parses:
                if p.score >= self._score_threshold:
                    pos = str(p.tag.POS) if p.tag.POS else "UNKNOWN"
                    variants.append((pos, p.normal_form, round(p.score, 4)))

            # Неоднозначное слово — если 2+ варианта
            # с разными частями речи
            unique_pos = set(v[0] for v in variants)
            if len(unique_pos) >= 2:
                ambiguous.append(AmbiguousToken(
                    token=word,
                    variants=variants,
                    top_score=variants[0][2] if variants else 0.0,
                    n_variants=len(unique_pos),
                ))

        ratio = len(ambiguous) / total if total > 0 else 0.0

        return AmbiguityInfo(
            text=text,
            total_tokens=total,
            ambiguous_tokens=ambiguous,
            ambiguity_ratio=ratio,
        )

    def filter_ambiguous(
        self,
        sentences: list,
        min_ambiguous: int = 2,
        min_ratio: float = 0.0,
    ) -> list[tuple[object, AmbiguityInfo]]:
        """Отфильтровать предложения с высокой неоднозначностью.

        Args:
            sentences: Список объектов с атрибутом .text
                (PosSample, LemmaSample и т.д.).
            min_ambiguous: Минимум неоднозначных токенов.
            min_ratio: Минимальная доля неоднозначных токенов.

        Returns:
            Список кортежей (sample, AmbiguityInfo),
            отсортированный по ambiguity_ratio (убывание).
        """
        results = []

        for sample in sentences:
            info = self.analyze(sample.text)

            if (
                len(info.ambiguous_tokens) >= min_ambiguous
                and info.ambiguity_ratio >= min_ratio
            ):
                results.append((sample, info))

        results.sort(key=lambda x: x[1].ambiguity_ratio, reverse=True)

        logger.info(
            "Отфильтровано %d предложений с >= %d неоднозначными токенами",
            len(results),
            min_ambiguous,
        )

        return results

if __name__ == "__main__":
    analyzer = AmbiguityAnalyzer()

    test_sentences = [
        "Рабочие стали выходить из цеха",
        "Мы печь пироги не стали",
        "Больной быстро стал поправляться",
        "Три стали использовались в производстве",
        "Владимир Путин посетил Москву",
    ]

    for sent in test_sentences:
        info = analyzer.analyze(sent)
        print(f"\n\"{sent}\"")
        print(f"  Токенов: {info.total_tokens}, неоднозначных: {len(info.ambiguous_tokens)}, ratio: {info.ambiguity_ratio:.2f}")

        for tok in info.ambiguous_tokens:
            variants_str = ", ".join(
                f"{pos}({lemma}, {score})"
                for pos, lemma, score in tok.variants
            )
            print(f"  {tok.token}: {variants_str}")