"""Context precision metric: fraction of retrieved chunks the answer relied on."""

from __future__ import annotations

from typing import Any

from astraea.core.types import Response
from astraea.datasets.sample import Sample
from astraea.exceptions import MetricError
from astraea.metrics.base import Metric, MetricResult
from astraea.metrics.llm_judge import LLMJudge, parse_json_object

_PROMPT_TEMPLATE = """\
You are evaluating the quality of a retrieval system. For each context \
chunk listed below, decide whether the answer relied on it.

A chunk is "used" only if the answer would be incomplete or wrong without \
it. Background or duplicated information that did not influence the answer \
is "not used".

Return a single JSON object with one entry per chunk:
{{
  "chunks": [
    {{"index": 0, "used": true, "reason": "..."}},
    {{"index": 1, "used": false, "reason": "..."}},
    ...
  ]
}}

Respond with JSON only. Do not include any additional commentary.

Question:
{question}

Answer:
{answer}

Chunks:
{chunks}
"""


class ContextPrecision(Metric):
    """LLM-as-judge metric measuring how many retrieved chunks were useful.

    For each chunk in ``sample.context`` the judge labels it ``used`` or
    not. Score equals the fraction of chunks marked used. A high score
    means the retriever surfaced little noise; a low score means it
    over-retrieved.

    :param judge: Configured :class:`LLMJudge` used to perform the
        evaluation call.
    :type judge: LLMJudge
    """

    def __init__(self, judge: LLMJudge) -> None:
        self.name = "context_precision"
        self._judge = judge

    def score(self, sample: Sample, response: Response) -> MetricResult:
        """Evaluate per-chunk usefulness for one ``(sample, response)`` pair.

        :param sample: Evaluation example. Must have non-empty ``context``.
        :type sample: Sample
        :param response: Model response under evaluation.
        :type response: Response
        :returns: Score in ``[0, 1]`` along with the per-chunk verdicts.
        :rtype: MetricResult
        :raises astraea.exceptions.MetricError: When ``sample.context`` is
            empty, when the judge does not return valid JSON, or when the
            ``chunks`` field is missing or not a list.
        """
        if not sample.context:
            raise MetricError("ContextPrecision requires sample.context to be non-empty.")

        chunks_block = "\n".join(f"[{index}] {chunk}" for index, chunk in enumerate(sample.context))
        prompt = _PROMPT_TEMPLATE.format(
            question=sample.input,
            answer=response.text,
            chunks=chunks_block,
        )
        payload = parse_json_object(self._judge.ask(prompt))
        chunks = payload.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise MetricError("Judge response is missing a non-empty 'chunks' list.")

        used_count = _count_used(chunks)
        total = len(chunks)
        score = used_count / total
        return MetricResult(
            metric=self.name,
            score=score,
            reason=f"{used_count}/{total} chunks used by the answer.",
            metadata={
                "chunk_count": total,
                "used_count": used_count,
                "chunks": chunks,
            },
        )


def _count_used(chunks: list[Any]) -> int:
    """Count entries in ``chunks`` whose ``used`` field is ``True``.

    Items that are not mappings or that lack the field are treated as not
    used so a single malformed entry does not derail the evaluation.

    :param chunks: List of chunk verdicts produced by the judge.
    :type chunks: list[Any]
    :returns: Number of chunks marked as used.
    :rtype: int
    """
    used = 0
    for chunk in chunks:
        if isinstance(chunk, dict) and chunk.get("used") is True:
            used += 1
    return used
