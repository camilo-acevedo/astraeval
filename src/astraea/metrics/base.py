"""Abstract :class:`Metric` interface and :class:`MetricResult` dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from astraea.core.types import Response
from astraea.datasets.sample import Sample


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Outcome of evaluating one metric against one ``(sample, response)`` pair.

    :ivar metric: Name of the metric that produced the score.
    :vartype metric: str
    :ivar score: Score in the closed interval ``[0, 1]``.
    :vartype score: float
    :ivar reason: Optional human-readable explanation, useful for debugging
        and surfacing in HTML reports.
    :vartype reason: str | None
    :ivar metadata: Auxiliary information emitted by the metric, such as the
        number of supported claims for faithfulness.
    :vartype metadata: collections.abc.Mapping[str, Any]
    """

    metric: str
    score: float
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    Concrete subclasses assign :attr:`name` in their constructor and
    implement :meth:`score`. Metrics are expected to be deterministic given
    their configuration and the inputs.

    :ivar name: Stable identifier used in reports and aggregate summaries.
    :vartype name: str
    """

    name: str

    @abstractmethod
    def score(self, sample: Sample, response: Response) -> MetricResult:
        """Score a single ``(sample, response)`` pair.

        :param sample: Evaluation example fed to the model.
        :type sample: Sample
        :param response: Response produced by the model under test.
        :type response: Response
        :returns: Structured score for this metric on the given pair.
        :rtype: MetricResult
        :raises astraea.exceptions.MetricError: When the inputs do not
            satisfy the preconditions of this metric (for example, a
            reference-based metric receiving a sample without ``expected``).
        """
