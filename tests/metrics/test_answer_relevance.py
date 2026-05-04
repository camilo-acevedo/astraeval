"""Tests for :mod:`llm_evals.metrics.answer_relevance`."""

from __future__ import annotations

import json

import pytest

from llm_evals.core.types import Response
from llm_evals.datasets.sample import Sample
from llm_evals.exceptions import MetricError
from llm_evals.metrics.answer_relevance import AnswerRelevance
from llm_evals.metrics.llm_judge import LLMJudge
from llm_evals.providers.fake import FakeProvider


def _build_metric(judge_text: str) -> AnswerRelevance:
    """Construct an :class:`AnswerRelevance` whose judge returns ``judge_text``.

    :param judge_text: Canned response the judge produces on its single call.
    :type judge_text: str
    :returns: Metric instance ready to score one sample.
    :rtype: AnswerRelevance
    """
    provider = FakeProvider([judge_text])
    return AnswerRelevance(LLMJudge(provider, model="m"))


def _response(text: str = "answer") -> Response:
    """Build a minimal response carrying just the text under evaluation.

    :param text: Text the metric will score.
    :type text: str
    :returns: Minimal :class:`Response`.
    :rtype: Response
    """
    return Response(text=text, model="m", provider="fake")


def test_high_score_passes_through() -> None:
    """A judge returning ``1.0`` produces a score of ``1.0``."""
    metric = _build_metric(json.dumps({"score": 1.0, "reason": "perfect"}))

    result = metric.score(Sample(input="Q"), _response())

    assert result.score == 1.0
    assert result.reason == "perfect"


def test_partial_score_passes_through() -> None:
    """Mid-range scores are preserved."""
    metric = _build_metric(json.dumps({"score": 0.5}))

    result = metric.score(Sample(input="Q"), _response())

    assert result.score == 0.5
    assert result.reason is None


def test_score_above_one_is_clamped() -> None:
    """A misbehaving judge returning above 1.0 is clamped."""
    metric = _build_metric(json.dumps({"score": 1.7, "reason": "x"}))

    result = metric.score(Sample(input="Q"), _response())

    assert result.score == 1.0
    assert result.metadata["raw_score"] == 1.7


def test_score_below_zero_is_clamped() -> None:
    """A negative score is clamped to ``0.0``."""
    metric = _build_metric(json.dumps({"score": -0.4}))

    result = metric.score(Sample(input="Q"), _response())

    assert result.score == 0.0
    assert result.metadata["raw_score"] == -0.4


def test_integer_score_is_accepted() -> None:
    """JSON integers are coerced to floats so ``{"score": 1}`` is valid."""
    metric = _build_metric(json.dumps({"score": 1}))

    result = metric.score(Sample(input="Q"), _response())

    assert result.score == 1.0


def test_boolean_score_is_rejected() -> None:
    """JSON booleans (which are ``int`` subclasses in Python) are rejected explicitly."""
    metric = _build_metric(json.dumps({"score": True}))

    with pytest.raises(MetricError, match="numeric"):
        metric.score(Sample(input="Q"), _response())


def test_missing_score_raises_metric_error() -> None:
    """A response without ``score`` surfaces clearly."""
    metric = _build_metric(json.dumps({"reason": "x"}))

    with pytest.raises(MetricError, match="numeric"):
        metric.score(Sample(input="Q"), _response())


def test_does_not_require_context() -> None:
    """``AnswerRelevance`` is question/answer only and ignores empty context."""
    metric = _build_metric(json.dumps({"score": 0.8}))

    result = metric.score(Sample(input="Q", context=()), _response())

    assert result.score == 0.8
