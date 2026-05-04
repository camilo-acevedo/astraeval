"""Tests for :mod:`llm_evals.core.eval_run`."""

from __future__ import annotations

from llm_evals.core.eval_run import EvalRun
from llm_evals.datasets.sample import Sample
from llm_evals.metrics.exact_match import ExactMatch
from llm_evals.providers.fake import FakeProvider


def test_run_produces_sample_results_in_order() -> None:
    """The run yields one :class:`SampleResult` per sample, preserving order."""
    provider = FakeProvider(["yes", "no"])
    samples = [
        Sample(input="Q1", expected="yes"),
        Sample(input="Q2", expected="yes"),
    ]

    result = EvalRun(provider, samples, [ExactMatch()], model="m").execute()

    assert len(result.samples) == 2
    assert result.samples[0].response.text == "yes"
    assert result.samples[1].response.text == "no"


def test_summary_is_mean_of_per_metric_scores() -> None:
    """The summary maps each metric name to the arithmetic mean of its scores."""
    provider = FakeProvider(["yes", "wrong"])
    samples = [
        Sample(input="Q1", expected="yes"),
        Sample(input="Q2", expected="yes"),
    ]

    result = EvalRun(provider, samples, [ExactMatch()], model="m").execute()

    assert result.summary == {"exact_match": 0.5}


def test_manifest_records_run_metadata() -> None:
    """The manifest carries provider, model, metric names, and sample count."""
    provider = FakeProvider(["yes"])
    samples = [Sample(input="Q1", expected="yes")]

    result = EvalRun(
        provider,
        samples,
        [ExactMatch()],
        model="claude-test",
        params={"temperature": 0.0},
    ).execute()

    manifest = result.manifest
    assert manifest.provider == "fake"
    assert manifest.model == "claude-test"
    assert manifest.metric_names == ("exact_match",)
    assert manifest.sample_count == 1
    assert manifest.summary == {"exact_match": 1.0}
    assert manifest.started_at <= manifest.finished_at
    assert len(manifest.run_id) == 16


def test_dataset_iterable_is_only_consumed_once() -> None:
    """Generators may be passed in; the run materializes them internally."""

    def gen() -> list[Sample]:
        return [Sample(input="Q", expected="yes")]

    provider = FakeProvider(["yes"])
    result = EvalRun(provider, gen(), [ExactMatch()], model="m").execute()

    assert result.summary["exact_match"] == 1.0


def test_empty_metric_list_still_produces_results() -> None:
    """Running without metrics returns sample responses and an empty summary."""
    provider = FakeProvider(["whatever"])
    samples = [Sample(input="Q")]

    result = EvalRun(provider, samples, [], model="m").execute()

    assert result.summary == {}
    assert result.samples[0].response.text == "whatever"


def test_params_are_forwarded_to_provider() -> None:
    """Provider params declared on the run reach the underlying ``complete`` call."""
    from collections.abc import Mapping
    from typing import Any

    seen: list[dict[str, Any]] = []

    def handler(prompt: str, model: str, params: Mapping[str, Any]) -> str:
        seen.append(dict(params))
        return "ok"

    provider = FakeProvider(handler=handler)
    samples = [Sample(input="Q", expected="ok")]

    EvalRun(
        provider,
        samples,
        [ExactMatch()],
        model="m",
        params={"temperature": 0.7},
    ).execute()

    assert seen == [{"temperature": 0.7}]
