"""Heuristic hallucination flag based on entities and numbers absent from context."""

from __future__ import annotations

import re

from astraea.core.types import Response
from astraea.datasets.sample import Sample
from astraea.exceptions import MetricError
from astraea.metrics.base import Metric, MetricResult

_NUMBER_RE = re.compile(r"\d[\d.,]*")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


class HallucinationFlag(Metric):
    """Heuristic flag for entities and numbers in the response absent from the context.

    The metric extracts numbers and capitalized-word sequences (a coarse
    proxy for proper nouns) from the response, then checks whether each
    token also appears in the joined context. The score is
    ``1 - hallucinated / extracted``. When nothing is extracted, the score
    is ``1.0``: an answer that asserts no entity or number cannot have
    hallucinated one.

    The check is case-insensitive by default, which trades a few false
    negatives ("apple" matching "Apple Inc.") for fewer false alarms on
    surface-form mismatches. This is a cheap proxy, not a substitute for
    LLM-judged faithfulness; use both.

    :param normalize_case: Compare tokens in a case-insensitive way.
    :type normalize_case: bool
    """

    def __init__(self, *, normalize_case: bool = True) -> None:
        self.name = "hallucination_flag"
        self._normalize_case = normalize_case

    def score(self, sample: Sample, response: Response) -> MetricResult:
        """Detect tokens in the response that do not appear in the context.

        :param sample: Evaluation example. Must have non-empty ``context``.
        :type sample: Sample
        :param response: Model response under evaluation.
        :type response: Response
        :returns: Score in ``[0, 1]`` along with the list of suspect tokens.
        :rtype: MetricResult
        :raises astraea.exceptions.MetricError: When ``sample.context``
            is empty; the metric has nothing to check against.
        """
        if not sample.context:
            raise MetricError("HallucinationFlag requires sample.context to be non-empty.")

        haystack = " ".join(sample.context)
        if self._normalize_case:
            haystack = haystack.casefold()

        extracted = _extract_tokens(response.text)
        if not extracted:
            return MetricResult(
                metric=self.name,
                score=1.0,
                reason="No entities or numbers extracted from the answer.",
                metadata={"extracted_count": 0, "hallucinated": []},
            )

        hallucinated: list[str] = []
        for token in extracted:
            needle = token.casefold() if self._normalize_case else token
            if needle not in haystack:
                hallucinated.append(token)

        score = 1.0 - len(hallucinated) / len(extracted)
        reason = (
            f"{len(hallucinated)}/{len(extracted)} tokens not found in context."
            if hallucinated
            else "All extracted tokens found in context."
        )
        return MetricResult(
            metric=self.name,
            score=score,
            reason=reason,
            metadata={
                "extracted_count": len(extracted),
                "hallucinated": hallucinated,
            },
        )


def _extract_tokens(text: str) -> list[str]:
    """Extract numbers and proper-noun sequences from ``text``.

    Numbers cover digits with optional decimal points, commas, or both
    (``42``, ``3.14``, ``1,200``). Proper nouns cover one or more
    consecutive capitalized words (``Paris``, ``New York City``). The two
    lists are concatenated; duplicates are preserved so a token mentioned
    multiple times is checked multiple times.

    :param text: Source text to scan.
    :type text: str
    :returns: List of extracted tokens in the order they appeared.
    :rtype: list[str]
    """
    return [*_NUMBER_RE.findall(text), *_PROPER_NOUN_RE.findall(text)]
