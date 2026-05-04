"""Heuristic exact-match metric."""

from __future__ import annotations

from astraeval.core.types import Response
from astraeval.datasets.sample import Sample
from astraeval.exceptions import MetricError
from astraeval.metrics.base import Metric, MetricResult


class ExactMatch(Metric):
    """Score ``1.0`` when the response equals the expected answer, else ``0.0``.

    By default both strings are stripped of leading and trailing whitespace
    and lower-cased before comparison. Disable :paramref:`normalize` to
    require byte-for-byte equality.

    :param normalize: When ``True``, both strings are stripped and
        case-folded before comparison.
    :type normalize: bool
    """

    def __init__(self, *, normalize: bool = True) -> None:
        self.name = "exact_match"
        self._normalize = normalize

    def score(self, sample: Sample, response: Response) -> MetricResult:
        """Compare ``response.text`` to ``sample.expected``.

        :param sample: Evaluation example. Must define ``expected``.
        :type sample: Sample
        :param response: Model response under evaluation.
        :type response: Response
        :returns: ``MetricResult`` with score 1.0 on match, 0.0 otherwise.
        :rtype: MetricResult
        :raises astraeval.exceptions.MetricError: When ``sample.expected`` is
            ``None``.
        """
        if sample.expected is None:
            raise MetricError("ExactMatch requires sample.expected to be set.")
        a = sample.expected
        b = response.text
        if self._normalize:
            a = a.strip().casefold()
            b = b.strip().casefold()
        is_match = a == b
        return MetricResult(
            metric=self.name,
            score=1.0 if is_match else 0.0,
            reason="match" if is_match else "mismatch",
            metadata={"normalize": self._normalize},
        )
