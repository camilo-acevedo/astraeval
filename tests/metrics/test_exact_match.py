"""Tests for :mod:`astraeval.metrics.exact_match`."""

from __future__ import annotations

import pytest

from astraeval.core.types import Response
from astraeval.datasets.sample import Sample
from astraeval.exceptions import MetricError
from astraeval.metrics.exact_match import ExactMatch


def _response(text: str) -> Response:
    """Build a response object carrying only the fields the metric inspects.

    :param text: Text the metric will compare against ``sample.expected``.
    :type text: str
    :returns: Minimal :class:`Response`.
    :rtype: Response
    """
    return Response(text=text, model="m", provider="fake")


def test_normalized_match_ignores_case_and_whitespace() -> None:
    """With ``normalize=True`` (default), surface differences do not penalize."""
    metric = ExactMatch()
    sample = Sample(input="Q", expected="Hello World")

    result = metric.score(sample, _response("  hello world  "))

    assert result.score == 1.0
    assert result.reason == "match"


def test_strict_match_rejects_case_differences() -> None:
    """With ``normalize=False``, comparison is byte-for-byte."""
    metric = ExactMatch(normalize=False)
    sample = Sample(input="Q", expected="Hello")

    assert metric.score(sample, _response("hello")).score == 0.0
    assert metric.score(sample, _response("Hello")).score == 1.0


def test_mismatch_returns_zero_with_reason() -> None:
    """A non-matching response produces score 0.0 and the ``"mismatch"`` reason."""
    metric = ExactMatch()
    sample = Sample(input="Q", expected="cat")

    result = metric.score(sample, _response("dog"))

    assert result.score == 0.0
    assert result.reason == "mismatch"


def test_missing_expected_raises_metric_error() -> None:
    """``ExactMatch`` requires reference text and reports the precondition clearly."""
    metric = ExactMatch()
    sample = Sample(input="Q")

    with pytest.raises(MetricError, match="expected"):
        metric.score(sample, _response("anything"))


def test_metadata_records_normalization_choice() -> None:
    """The configured ``normalize`` flag is reflected in ``metadata``."""
    strict = ExactMatch(normalize=False)
    lenient = ExactMatch(normalize=True)
    sample = Sample(input="Q", expected="x")

    assert strict.score(sample, _response("x")).metadata == {"normalize": False}
    assert lenient.score(sample, _response("x")).metadata == {"normalize": True}
