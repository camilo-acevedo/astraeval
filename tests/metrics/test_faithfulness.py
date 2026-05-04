"""Tests for :mod:`astraea.metrics.faithfulness`."""

from __future__ import annotations

import json

import pytest

from astraea.core.types import Response
from astraea.datasets.sample import Sample
from astraea.exceptions import MetricError
from astraea.metrics.faithfulness import Faithfulness
from astraea.metrics.llm_judge import LLMJudge
from astraea.providers.fake import FakeProvider


def _build_metric(judge_text: str) -> Faithfulness:
    """Construct a :class:`Faithfulness` whose judge returns ``judge_text``.

    :param judge_text: Canned response the judge produces on its single call.
    :type judge_text: str
    :returns: Metric instance ready to score one sample.
    :rtype: Faithfulness
    """
    provider = FakeProvider([judge_text])
    judge = LLMJudge(provider, model="judge-m")
    return Faithfulness(judge)


def _response(text: str = "answer") -> Response:
    """Build a minimal response carrying just the text under evaluation.

    :param text: Text the metric will score.
    :type text: str
    :returns: Minimal :class:`Response`.
    :rtype: Response
    """
    return Response(text=text, model="m", provider="fake")


def test_all_claims_supported_yields_one() -> None:
    """When every claim is supported the score is exactly 1.0."""
    judge_payload = json.dumps(
        {
            "claims": [
                {"text": "claim a", "supported": True},
                {"text": "claim b", "supported": True},
            ]
        }
    )
    metric = _build_metric(judge_payload)
    sample = Sample(input="Q", context=("ctx",))

    result = metric.score(sample, _response())

    assert result.score == 1.0
    assert result.metadata["claim_count"] == 2
    assert result.metadata["supported_count"] == 2


def test_partial_support_produces_fraction() -> None:
    """Score equals supported / total."""
    judge_payload = json.dumps(
        {
            "claims": [
                {"text": "a", "supported": True},
                {"text": "b", "supported": False},
                {"text": "c", "supported": True},
                {"text": "d", "supported": False},
            ]
        }
    )
    metric = _build_metric(judge_payload)
    sample = Sample(input="Q", context=("ctx",))

    result = metric.score(sample, _response())

    assert result.score == 0.5
    assert result.metadata["supported_count"] == 2


def test_empty_claim_list_yields_perfect_score() -> None:
    """An answer with no extractable claims cannot hallucinate; score is 1.0."""
    metric = _build_metric(json.dumps({"claims": []}))
    sample = Sample(input="Q", context=("ctx",))

    result = metric.score(sample, _response())

    assert result.score == 1.0
    assert result.metadata["claim_count"] == 0


def test_judge_response_with_code_fence_is_accepted() -> None:
    """Common LLM behaviour of wrapping JSON in fences is tolerated."""
    judge_payload = "```json\n" + json.dumps({"claims": [{"supported": True}]}) + "\n```"
    metric = _build_metric(judge_payload)
    sample = Sample(input="Q", context=("ctx",))

    result = metric.score(sample, _response())

    assert result.score == 1.0


def test_malformed_claim_entries_count_as_unsupported() -> None:
    """Non-dict claim entries are treated as unsupported rather than raising."""
    judge_payload = json.dumps({"claims": [{"supported": True}, "garbage", 42]})
    metric = _build_metric(judge_payload)
    sample = Sample(input="Q", context=("ctx",))

    result = metric.score(sample, _response())

    assert result.score == pytest.approx(1 / 3)


def test_missing_context_raises_metric_error() -> None:
    """``Faithfulness`` requires retrieved context to evaluate against."""
    metric = _build_metric(json.dumps({"claims": []}))
    sample = Sample(input="Q")

    with pytest.raises(MetricError, match="context"):
        metric.score(sample, _response())


def test_missing_claims_field_raises_metric_error() -> None:
    """A judge response without a 'claims' list is reported, not silently ignored."""
    metric = _build_metric(json.dumps({"unrelated": []}))
    sample = Sample(input="Q", context=("ctx",))

    with pytest.raises(MetricError, match="claims"):
        metric.score(sample, _response())


def test_prompt_includes_question_answer_and_context() -> None:
    """The prompt sent to the judge embeds every input the metric receives."""
    from collections.abc import Mapping
    from typing import Any

    seen_prompts: list[str] = []

    def handler(prompt: str, model: str, params: Mapping[str, Any]) -> str:
        seen_prompts.append(prompt)
        return json.dumps({"claims": []})

    provider = FakeProvider(handler=handler)
    judge = LLMJudge(provider, model="m")
    metric = Faithfulness(judge)
    sample = Sample(
        input="What color is the sky?",
        context=("The sky is blue during the day.",),
    )

    metric.score(sample, _response("Blue."))

    assert len(seen_prompts) == 1
    prompt = seen_prompts[0]
    assert "What color is the sky?" in prompt
    assert "The sky is blue during the day." in prompt
    assert "Blue." in prompt
