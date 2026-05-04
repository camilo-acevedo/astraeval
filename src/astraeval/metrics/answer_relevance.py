"""Answer relevance metric: how directly the response addresses the question."""

from __future__ import annotations

from astraeval.core.types import Response
from astraeval.datasets.sample import Sample
from astraeval.exceptions import MetricError
from astraeval.metrics.base import Metric, MetricResult
from astraeval.metrics.llm_judge import LLMJudge, parse_json_object

_PROMPT_TEMPLATE = """\
You are evaluating whether an answer addresses the user's question.

Score the answer on a scale from 0.0 to 1.0:
- 1.0: directly and completely addresses the question.
- 0.5: partially addresses the question or addresses an adjacent question.
- 0.0: does not address the question at all.

Return a single JSON object:
{{"score": 0.0, "reason": "one short sentence"}}

Respond with JSON only. Do not include any additional commentary.

Question:
{question}

Answer:
{answer}
"""


class AnswerRelevance(Metric):
    """LLM-as-judge metric scoring how directly an answer addresses the question.

    The judge returns a continuous score in ``[0, 1]``. Out-of-range values
    are clamped so a misbehaving judge cannot push aggregates outside the
    interval all metrics agree on.

    :param judge: Configured :class:`LLMJudge` used to perform the
        evaluation call.
    :type judge: LLMJudge
    """

    def __init__(self, judge: LLMJudge) -> None:
        self.name = "answer_relevance"
        self._judge = judge

    def score(self, sample: Sample, response: Response) -> MetricResult:
        """Ask the judge whether ``response`` addresses ``sample.input``.

        :param sample: Evaluation example whose ``input`` is the question.
        :type sample: Sample
        :param response: Model response under evaluation.
        :type response: Response
        :returns: Score in ``[0, 1]`` along with the judge's reason.
        :rtype: MetricResult
        :raises astraeval.exceptions.MetricError: When the judge does not
            return valid JSON or the ``score`` field is missing or
            non-numeric.
        """
        prompt = _PROMPT_TEMPLATE.format(question=sample.input, answer=response.text)
        payload = parse_json_object(self._judge.ask(prompt).text)

        raw_score = payload.get("score")
        if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
            raise MetricError("Judge response is missing a numeric 'score' field.")
        score = max(0.0, min(1.0, float(raw_score)))

        reason_field = payload.get("reason")
        reason = reason_field if isinstance(reason_field, str) else None

        return MetricResult(
            metric=self.name,
            score=score,
            reason=reason,
            metadata={"raw_score": raw_score},
        )
