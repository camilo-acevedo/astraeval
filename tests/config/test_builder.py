"""Tests for :mod:`llm_evals.config.builder`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_evals.config.builder import (
    build_dataset,
    build_eval_run,
    build_judge,
    build_metric,
    build_provider,
)
from llm_evals.config.schema import (
    CacheConfig,
    DatasetConfig,
    JudgeConfig,
    MetricConfig,
    ProviderConfig,
    RunConfig,
)
from llm_evals.exceptions import ConfigError
from llm_evals.metrics.answer_relevance import AnswerRelevance
from llm_evals.metrics.context_precision import ContextPrecision
from llm_evals.metrics.exact_match import ExactMatch
from llm_evals.metrics.faithfulness import Faithfulness
from llm_evals.metrics.hallucination import HallucinationFlag
from llm_evals.metrics.llm_judge import LLMJudge
from llm_evals.providers.fake import FakeProvider


def test_build_provider_fake_requires_responses() -> None:
    """A ``fake`` provider config without responses fails fast."""
    cfg = ProviderConfig(type="fake", model="m")

    with pytest.raises(ConfigError, match="responses"):
        build_provider(cfg)


def test_build_provider_fake_with_responses() -> None:
    """``fake`` configurations build a :class:`FakeProvider` with the canned list."""
    cfg = ProviderConfig(type="fake", model="m", responses=("a", "b"))

    provider = build_provider(cfg)

    assert isinstance(provider, FakeProvider)
    assert provider.complete("p", model="m").text == "a"


def test_build_dataset_jsonl(tmp_path: Path) -> None:
    """The JSONL loader is wired to the configured path."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text('{"input": "Q1"}\n{"input": "Q2"}\n', encoding="utf-8")
    cfg = DatasetConfig(type="jsonl", path=str(fixture))

    samples = list(build_dataset(cfg))

    assert [s.input for s in samples] == ["Q1", "Q2"]


def test_build_judge_uses_provider_model() -> None:
    """The judge inherits the model from its provider configuration."""
    cfg = JudgeConfig(
        provider=ProviderConfig(type="fake", model="judge-m", responses=("ok",)),
        params={"temperature": 0.0},
    )

    judge = build_judge(cfg)

    assert isinstance(judge, LLMJudge)
    assert judge.model == "judge-m"


def test_build_metric_concrete_classes() -> None:
    """Each metric type produces the matching concrete metric instance."""
    judge = LLMJudge(FakeProvider(["ok"]), model="m")

    assert isinstance(build_metric(MetricConfig(type="exact_match")), ExactMatch)
    assert isinstance(build_metric(MetricConfig(type="hallucination_flag")), HallucinationFlag)
    assert isinstance(build_metric(MetricConfig(type="faithfulness"), judge=judge), Faithfulness)
    assert isinstance(
        build_metric(MetricConfig(type="answer_relevance"), judge=judge), AnswerRelevance
    )
    assert isinstance(
        build_metric(MetricConfig(type="context_precision"), judge=judge), ContextPrecision
    )


def test_build_metric_judge_required_for_llm_judges() -> None:
    """LLM-as-judge metrics refuse to build without a judge."""
    with pytest.raises(ConfigError, match="judge"):
        build_metric(MetricConfig(type="faithfulness"))


def test_build_metric_options_propagate() -> None:
    """Non-``type`` keys reach the metric constructor as keyword arguments."""
    metric = build_metric(MetricConfig(type="exact_match", options={"normalize": False}))

    assert isinstance(metric, ExactMatch)


def test_build_metric_invalid_options_surface_as_config_error() -> None:
    """Constructor TypeError on bad kwargs surfaces as :class:`ConfigError`."""
    with pytest.raises(ConfigError, match="options"):
        build_metric(MetricConfig(type="exact_match", options={"unknown_kwarg": 1}))


def test_build_eval_run_wraps_provider_when_cache_enabled(tmp_path: Path) -> None:
    """With cache enabled, the provider exposed to ``EvalRun`` is a :class:`CachedProvider`."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text('{"input": "Q", "expected": "ok"}\n', encoding="utf-8")
    cfg = RunConfig(
        provider=ProviderConfig(type="fake", model="m", responses=("ok",)),
        dataset=DatasetConfig(type="jsonl", path=str(fixture)),
        metrics=(MetricConfig(type="exact_match"),),
        cache=CacheConfig(enabled=True, path=str(tmp_path / "c.sqlite")),
    )

    run = build_eval_run(cfg)
    result = run.execute()

    assert result.summary["exact_match"] == 1.0
    assert (tmp_path / "c.sqlite").exists()


def test_build_eval_run_skips_cache_when_disabled(tmp_path: Path) -> None:
    """With cache disabled, no SQLite database is created."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text('{"input": "Q", "expected": "ok"}\n', encoding="utf-8")
    cfg = RunConfig(
        provider=ProviderConfig(type="fake", model="m", responses=("ok",)),
        dataset=DatasetConfig(type="jsonl", path=str(fixture)),
        metrics=(MetricConfig(type="exact_match"),),
        cache=CacheConfig(enabled=False, path=str(tmp_path / "should-not-exist.sqlite")),
    )

    build_eval_run(cfg).execute()

    assert not (tmp_path / "should-not-exist.sqlite").exists()


def test_build_eval_run_with_llm_judge_metric(tmp_path: Path) -> None:
    """A run configuring an LLM-as-judge metric receives a shared judge."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text(
        '{"input": "Q", "context": ["ctx"]}\n',
        encoding="utf-8",
    )
    judge_response = json.dumps({"claims": [{"supported": True}]})
    cfg = RunConfig(
        provider=ProviderConfig(type="fake", model="m", responses=("answer",)),
        dataset=DatasetConfig(type="jsonl", path=str(fixture)),
        metrics=(MetricConfig(type="faithfulness"),),
        cache=CacheConfig(enabled=False),
        judge=JudgeConfig(
            provider=ProviderConfig(type="fake", model="judge", responses=(judge_response,))
        ),
    )

    result = build_eval_run(cfg).execute()

    assert result.summary["faithfulness"] == 1.0
