"""Evaluation run orchestrator combining a provider, dataset, and metrics."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from llm_evals.core.manifest import (
    RunManifest,
    hash_dataset,
    hash_params,
    utc_now_iso,
)
from llm_evals.core.types import Response
from llm_evals.datasets.sample import Sample
from llm_evals.metrics.base import Metric, MetricResult
from llm_evals.providers.base import Provider


@dataclass(frozen=True, slots=True)
class SampleResult:
    """Per-sample result aggregating the model response and every metric score.

    :ivar sample: The evaluation example.
    :vartype sample: Sample
    :ivar response: Response produced by the model under test.
    :vartype response: Response
    :ivar metrics: Tuple of metric results in the order metrics were
        configured on the run.
    :vartype metrics: tuple[MetricResult, ...]
    """

    sample: Sample
    response: Response
    metrics: tuple[MetricResult, ...]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Full result of an :class:`EvalRun` execution.

    :ivar samples: Per-sample results in dataset order.
    :vartype samples: tuple[SampleResult, ...]
    :ivar summary: Mapping from metric name to mean score across samples.
    :vartype summary: collections.abc.Mapping[str, float]
    :ivar manifest: Reproducibility manifest describing the run.
    :vartype manifest: RunManifest
    """

    samples: tuple[SampleResult, ...]
    summary: Mapping[str, float]
    manifest: RunManifest


class EvalRun:
    """Run a model over a dataset and score every response with each metric.

    :param provider: Provider used to produce model responses.
    :type provider: Provider
    :param dataset: Iterable of :class:`Sample` instances.
    :type dataset: collections.abc.Iterable[Sample]
    :param metrics: Sequence of metrics applied to every response. The order
        is preserved in result objects and the manifest.
    :type metrics: collections.abc.Sequence[Metric]
    :param model: Model identifier forwarded to ``Provider.complete``.
    :type model: str
    :param params: Optional provider parameters (temperature, max_tokens,
        etc.) forwarded on every call. Defaults to an empty mapping.
    :type params: collections.abc.Mapping[str, Any] | None
    """

    def __init__(
        self,
        provider: Provider,
        dataset: Iterable[Sample],
        metrics: Sequence[Metric],
        *,
        model: str,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self._provider = provider
        self._dataset = dataset
        self._metrics = tuple(metrics)
        self._model = model
        self._params: Mapping[str, Any] = dict(params) if params is not None else {}

    def execute(self) -> RunResult:
        """Run the evaluation end to end.

        Materializes the dataset (so the manifest can hash it), submits each
        sample to the provider, scores the response with every metric, and
        produces an aggregate summary alongside a :class:`RunManifest`.

        :returns: Aggregate result of the run.
        :rtype: RunResult
        """
        samples = tuple(self._dataset)
        started_at = utc_now_iso()

        results: list[SampleResult] = []
        per_metric: dict[str, list[float]] = {m.name: [] for m in self._metrics}

        for sample in samples:
            response = self._provider.complete(
                sample.input,
                model=self._model,
                **self._params,
            )
            metric_results = tuple(metric.score(sample, response) for metric in self._metrics)
            for result in metric_results:
                per_metric[result.metric].append(result.score)
            results.append(SampleResult(sample=sample, response=response, metrics=metric_results))

        finished_at = utc_now_iso()
        summary = {
            name: (sum(scores) / len(scores)) if scores else 0.0
            for name, scores in per_metric.items()
        }

        dataset_hash = hash_dataset([asdict(s) for s in samples])
        params_hash = hash_params(self._params)
        run_id = _make_run_id(
            provider=self._provider.name,
            model=self._model,
            dataset_hash=dataset_hash,
            params_hash=params_hash,
            started_at=started_at,
        )
        manifest = RunManifest(
            run_id=run_id,
            provider=self._provider.name,
            model=self._model,
            metric_names=tuple(m.name for m in self._metrics),
            sample_count=len(samples),
            dataset_hash=dataset_hash,
            params_hash=params_hash,
            started_at=started_at,
            finished_at=finished_at,
            summary=summary,
        )
        return RunResult(samples=tuple(results), summary=summary, manifest=manifest)


def _make_run_id(
    *,
    provider: str,
    model: str,
    dataset_hash: str,
    params_hash: str,
    started_at: str,
) -> str:
    """Build a short identifier mixing the inputs that determined the run.

    :param provider: Provider name.
    :type provider: str
    :param model: Model identifier.
    :type model: str
    :param dataset_hash: SHA-256 of the dataset contents.
    :type dataset_hash: str
    :param params_hash: SHA-256 of the forwarded parameters.
    :type params_hash: str
    :param started_at: ISO-8601 start timestamp.
    :type started_at: str
    :returns: First 16 hex characters of the combined SHA-256 digest.
    :rtype: str
    """
    payload = f"{provider}|{model}|{dataset_hash}|{params_hash}|{started_at}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
