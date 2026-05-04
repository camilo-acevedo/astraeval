"""Tests for :mod:`astraeval.metrics.hallucination`."""

from __future__ import annotations

import pytest

from astraeval.core.types import Response
from astraeval.datasets.sample import Sample
from astraeval.exceptions import MetricError
from astraeval.metrics.hallucination import HallucinationFlag


def _response(text: str) -> Response:
    """Build a minimal response carrying just the text under evaluation.

    :param text: Text the metric will scan for entities and numbers.
    :type text: str
    :returns: Minimal :class:`Response`.
    :rtype: Response
    """
    return Response(text=text, model="m", provider="fake")


def test_all_tokens_found_yields_perfect_score() -> None:
    """When every extracted token appears in the context the score is 1.0."""
    metric = HallucinationFlag()
    sample = Sample(
        input="Q",
        context=("The Eiffel Tower in Paris is 330 meters tall.",),
    )

    result = metric.score(sample, _response("Paris has the Eiffel Tower at 330 meters."))

    assert result.score == 1.0
    assert result.metadata["hallucinated"] == []


def test_unsupported_number_lowers_score() -> None:
    """A number absent from the context counts as a hallucination."""
    metric = HallucinationFlag()
    sample = Sample(input="Q", context=("Paris is the capital of France.",))

    result = metric.score(sample, _response("Paris has 12 million tourists."))

    assert result.score == pytest.approx(1 / 2)
    assert "12" in result.metadata["hallucinated"]


def test_unsupported_proper_noun_lowers_score() -> None:
    """A proper noun absent from the context counts as a hallucination."""
    metric = HallucinationFlag()
    sample = Sample(input="Q", context=("Lima is the capital of Peru.",))

    result = metric.score(sample, _response("Lima and Madrid are capitals."))

    assert result.score == pytest.approx(1 / 2)
    assert "Madrid" in result.metadata["hallucinated"]


def test_no_entities_or_numbers_yields_one() -> None:
    """A response without any extractable tokens cannot hallucinate."""
    metric = HallucinationFlag()
    sample = Sample(input="Q", context=("anything",))

    result = metric.score(sample, _response("yes it does work well"))

    assert result.score == 1.0
    assert result.metadata["extracted_count"] == 0


def test_case_insensitive_matching_by_default() -> None:
    """``Apple`` in the answer matches ``apple`` in the context with default settings."""
    metric = HallucinationFlag()
    sample = Sample(input="Q", context=("apple is a fruit",))

    result = metric.score(sample, _response("Apple is good."))

    assert result.score == 1.0


def test_strict_case_matching_flags_case_mismatch() -> None:
    """With ``normalize_case=False`` case differences register as hallucinations."""
    metric = HallucinationFlag(normalize_case=False)
    sample = Sample(input="Q", context=("apple is a fruit",))

    result = metric.score(sample, _response("Apple is good."))

    assert result.score == 0.0
    assert "Apple" in result.metadata["hallucinated"]


def test_multi_word_proper_nouns_extracted_as_one_token() -> None:
    """Consecutive capitalized words form a single token, e.g. ``New York``."""
    metric = HallucinationFlag()
    sample = Sample(input="Q", context=("San Francisco is in California.",))

    result = metric.score(sample, _response("San Francisco hosts events."))

    assert result.score == 1.0
    assert result.metadata["extracted_count"] == 1


def test_decimal_and_thousand_separators() -> None:
    """Numbers like ``3.14`` and ``1,200`` are extracted as single tokens."""
    metric = HallucinationFlag()
    sample = Sample(input="Q", context=("pi is 3.14 and the count was 1,200",))

    result = metric.score(sample, _response("3.14 and 1,200 are mentioned."))

    assert result.score == 1.0


def test_missing_context_raises_metric_error() -> None:
    """``HallucinationFlag`` cannot evaluate without a context to compare against."""
    metric = HallucinationFlag()

    with pytest.raises(MetricError, match="context"):
        metric.score(Sample(input="Q"), _response("Paris"))
