"""Tests for :mod:`astraeval.metrics.context_precision`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from astraeval.core.types import Response
from astraeval.datasets.sample import Sample
from astraeval.exceptions import MetricError
from astraeval.metrics.context_precision import ContextPrecision
from astraeval.metrics.llm_judge import LLMJudge
from astraeval.providers.fake import FakeProvider


def _build_metric(judge_text: str) -> ContextPrecision:
    """Construct a :class:`ContextPrecision` whose judge returns ``judge_text``.

    :param judge_text: Canned response the judge produces on its single call.
    :type judge_text: str
    :returns: Metric instance ready to score one sample.
    :rtype: ContextPrecision
    """
    return ContextPrecision(LLMJudge(FakeProvider([judge_text]), model="m"))


def _response(text: str = "answer") -> Response:
    """Build a minimal response carrying just the text under evaluation.

    :param text: Text the metric will score.
    :type text: str
    :returns: Minimal :class:`Response`.
    :rtype: Response
    """
    return Response(text=text, model="m", provider="fake")


def test_all_chunks_used_yields_one() -> None:
    """When the judge marks every chunk used the score is exactly 1.0."""
    payload = json.dumps(
        {
            "chunks": [
                {"index": 0, "used": True},
                {"index": 1, "used": True},
            ]
        }
    )
    metric = _build_metric(payload)
    sample = Sample(input="Q", context=("c0", "c1"))

    result = metric.score(sample, _response())

    assert result.score == 1.0
    assert result.metadata["used_count"] == 2


def test_partial_use_produces_fraction() -> None:
    """Score equals used / total."""
    payload = json.dumps(
        {
            "chunks": [
                {"index": 0, "used": True},
                {"index": 1, "used": False},
                {"index": 2, "used": False},
                {"index": 3, "used": True},
            ]
        }
    )
    metric = _build_metric(payload)
    sample = Sample(input="Q", context=("c0", "c1", "c2", "c3"))

    result = metric.score(sample, _response())

    assert result.score == 0.5
    assert result.metadata["used_count"] == 2


def test_empty_chunks_list_raises_metric_error() -> None:
    """A judge that returns zero verdicts is reported, not silently scored 0/0."""
    metric = _build_metric(json.dumps({"chunks": []}))
    sample = Sample(input="Q", context=("c0",))

    with pytest.raises(MetricError, match="chunks"):
        metric.score(sample, _response())


def test_missing_context_raises_metric_error() -> None:
    """``ContextPrecision`` requires retrieved chunks to evaluate."""
    metric = _build_metric(json.dumps({"chunks": [{"used": True}]}))

    with pytest.raises(MetricError, match="context"):
        metric.score(Sample(input="Q"), _response())


def test_malformed_chunk_entries_count_as_unused() -> None:
    """Non-dict chunk entries do not raise; they are simply not counted."""
    payload = json.dumps({"chunks": [{"used": True}, "garbage", 42, {"used": True}]})
    metric = _build_metric(payload)
    sample = Sample(input="Q", context=("c0", "c1", "c2", "c3"))

    result = metric.score(sample, _response())

    assert result.score == 0.5


def test_prompt_includes_indexed_chunks() -> None:
    """Each chunk reaches the judge prefixed with its index for unambiguous reference."""
    seen_prompts: list[str] = []

    def handler(prompt: str, model: str, params: Mapping[str, Any]) -> str:
        seen_prompts.append(prompt)
        return json.dumps({"chunks": [{"used": True}]})

    metric = ContextPrecision(LLMJudge(FakeProvider(handler=handler), model="m"))
    sample = Sample(
        input="Q",
        context=("first chunk", "second chunk"),
    )

    metric.score(sample, _response())

    assert "[0] first chunk" in seen_prompts[0]
    assert "[1] second chunk" in seen_prompts[0]
