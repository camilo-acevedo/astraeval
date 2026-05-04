"""Faithfulness metric: fraction of answer claims supported by the context."""

from __future__ import annotations

from typing import Any

from astraeval.core.types import Response
from astraeval.datasets.sample import Sample
from astraeval.exceptions import MetricError
from astraeval.metrics.base import Metric, MetricResult
from astraeval.metrics.llm_judge import LLMJudge, parse_json_object

_PROMPT_TEMPLATE = """\
You are a strict evaluator. Decompose the answer into atomic factual claims and \
decide whether each claim is supported by the provided context.

Return a single JSON object with this shape:
{{
  "claims": [
    {{"text": "...", "supported": true, "reason": "..."}},
    ...
  ]
}}

Respond with JSON only. Do not include any additional commentary.

Context:
{context}

Question:
{question}

Answer:
{answer}
"""


class Faithfulness(Metric):
    """LLM-as-judge metric scoring how well an answer is grounded in context.

    The metric asks the judge to decompose the answer into atomic factual
    claims and to label each one as supported or unsupported by the
    context. The score is the fraction of supported claims.

    :param judge: Configured :class:`LLMJudge` used to perform the
        evaluation call.
    :type judge: LLMJudge
    """

    def __init__(self, judge: LLMJudge) -> None:
        self.name = "faithfulness"
        self._judge = judge

    def score(self, sample: Sample, response: Response) -> MetricResult:
        """Decompose the answer into claims and score the supported fraction.

        :param sample: Evaluation example. Must have non-empty ``context``.
        :type sample: Sample
        :param response: Model response under evaluation.
        :type response: Response
        :returns: Score in ``[0, 1]`` along with extracted claims as metadata.
        :rtype: MetricResult
        :raises astraeval.exceptions.MetricError: When ``sample.context`` is
            empty, when the judge does not return valid JSON, or when the
            ``claims`` field is missing or not a list.
        """
        if not sample.context:
            raise MetricError("Faithfulness requires sample.context to be non-empty.")

        prompt = _PROMPT_TEMPLATE.format(
            context="\n---\n".join(sample.context),
            question=sample.input,
            answer=response.text,
        )
        payload = parse_json_object(self._judge.ask(prompt).text)
        claims = payload.get("claims")
        if not isinstance(claims, list):
            raise MetricError("Judge response is missing a 'claims' list.")

        if not claims:
            return MetricResult(
                metric=self.name,
                score=1.0,
                reason="No claims extracted from the answer.",
                metadata={"claim_count": 0, "supported_count": 0, "claims": []},
            )

        supported_count = _count_supported(claims)
        total = len(claims)
        score = supported_count / total
        return MetricResult(
            metric=self.name,
            score=score,
            reason=f"{supported_count}/{total} claims supported by context.",
            metadata={
                "claim_count": total,
                "supported_count": supported_count,
                "claims": claims,
            },
        )


def _count_supported(claims: list[Any]) -> int:
    """Count entries in ``claims`` whose ``supported`` field is ``True``.

    Items that are not mappings or that lack the field are treated as
    unsupported rather than raising, so a single malformed entry does not
    invalidate the whole evaluation.

    :param claims: List of claim objects produced by the judge.
    :type claims: list[Any]
    :returns: Number of supported claims.
    :rtype: int
    """
    supported = 0
    for claim in claims:
        if isinstance(claim, dict) and claim.get("supported") is True:
            supported += 1
    return supported
