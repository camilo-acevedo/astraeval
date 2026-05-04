"""Factories that turn :mod:`astraea.config.schema` dataclasses into runtime objects."""

from __future__ import annotations

from collections.abc import Iterable

from astraea.config.schema import (
    DatasetConfig,
    JudgeConfig,
    MetricConfig,
    ProviderConfig,
    RunConfig,
)
from astraea.core.cache import Cache
from astraea.core.eval_run import EvalRun
from astraea.datasets.jsonl import load_jsonl
from astraea.datasets.sample import Sample
from astraea.exceptions import ConfigError
from astraea.metrics.answer_relevance import AnswerRelevance
from astraea.metrics.base import Metric
from astraea.metrics.context_precision import ContextPrecision
from astraea.metrics.exact_match import ExactMatch
from astraea.metrics.faithfulness import Faithfulness
from astraea.metrics.hallucination import HallucinationFlag
from astraea.metrics.llm_judge import LLMJudge
from astraea.providers.anthropic_provider import AnthropicProvider
from astraea.providers.base import Provider
from astraea.providers.cached import CachedProvider
from astraea.providers.fake import FakeProvider
from astraea.providers.ollama_provider import OllamaProvider
from astraea.providers.openai_provider import OpenAIProvider

_JUDGE_REQUIRED = {"faithfulness", "answer_relevance", "context_precision"}


def build_provider(cfg: ProviderConfig) -> Provider:
    """Construct the concrete provider described by ``cfg``.

    :param cfg: Validated provider configuration.
    :type cfg: ProviderConfig
    :returns: Concrete :class:`Provider` ready to issue completion calls.
    :rtype: Provider
    :raises astraea.exceptions.ConfigError: When a ``fake`` provider is
        requested without canned responses.
    """
    if cfg.type == "anthropic":
        return AnthropicProvider(
            api_key=cfg.api_key,
            default_max_tokens=cfg.default_max_tokens or 1024,
        )
    if cfg.type == "openai":
        return OpenAIProvider(api_key=cfg.api_key, base_url=cfg.base_url)
    if cfg.type == "ollama":
        return OllamaProvider(host=cfg.host)
    if cfg.type == "fake":
        if not cfg.responses:
            raise ConfigError("Fake provider requires a non-empty 'responses' list.")
        return FakeProvider(list(cfg.responses))
    raise ConfigError(f"Unknown provider type: {cfg.type!r}")


def build_dataset(cfg: DatasetConfig) -> Iterable[Sample]:
    """Construct an iterable of :class:`Sample` from a dataset configuration.

    :param cfg: Validated dataset configuration.
    :type cfg: DatasetConfig
    :returns: Iterable of samples ready to feed an :class:`EvalRun`.
    :rtype: collections.abc.Iterable[Sample]
    :raises astraea.exceptions.ConfigError: When the dataset type is
        unknown.
    """
    if cfg.type == "jsonl":
        return load_jsonl(cfg.path)
    raise ConfigError(f"Unknown dataset type: {cfg.type!r}")


def build_judge(cfg: JudgeConfig) -> LLMJudge:
    """Construct an :class:`LLMJudge` from a judge configuration.

    :param cfg: Validated judge configuration.
    :type cfg: JudgeConfig
    :returns: Configured judge ready for use by LLM-as-judge metrics.
    :rtype: LLMJudge
    """
    return LLMJudge(
        build_provider(cfg.provider),
        model=cfg.provider.model,
        params=cfg.params,
    )


def build_metric(cfg: MetricConfig, *, judge: LLMJudge | None = None) -> Metric:
    """Construct a metric from a single metric configuration entry.

    :param cfg: Validated metric configuration.
    :type cfg: MetricConfig
    :param judge: Judge required by LLM-as-judge metrics. ``None`` is only
        valid for heuristic metrics.
    :type judge: LLMJudge | None
    :returns: Concrete metric instance.
    :rtype: Metric
    :raises astraea.exceptions.ConfigError: When an LLM-as-judge metric
        is requested without a judge or when constructor options are not
        accepted by the metric.
    """
    if cfg.type in _JUDGE_REQUIRED and judge is None:
        raise ConfigError(f"Metric {cfg.type!r} requires a 'judge' configuration block.")
    try:
        if cfg.type == "exact_match":
            return ExactMatch(**dict(cfg.options))
        if cfg.type == "faithfulness":
            assert judge is not None
            return Faithfulness(judge)
        if cfg.type == "answer_relevance":
            assert judge is not None
            return AnswerRelevance(judge)
        if cfg.type == "context_precision":
            assert judge is not None
            return ContextPrecision(judge)
        if cfg.type == "hallucination_flag":
            return HallucinationFlag(**dict(cfg.options))
    except TypeError as exc:
        raise ConfigError(f"Invalid options for metric {cfg.type!r}: {exc}") from exc
    raise ConfigError(f"Unknown metric type: {cfg.type!r}")


def build_eval_run(cfg: RunConfig) -> EvalRun:
    """Compose a full :class:`EvalRun` from a validated :class:`RunConfig`.

    Wraps the model provider in a :class:`CachedProvider` when
    ``cache.enabled`` is true. The judge, if any, is constructed once and
    shared across LLM-as-judge metrics.

    :param cfg: Validated run configuration.
    :type cfg: RunConfig
    :returns: Ready-to-execute evaluation run.
    :rtype: EvalRun
    :raises astraea.exceptions.ConfigError: When metric configuration
        cannot be honoured (for example LLM-as-judge metrics without a
        ``judge`` block).
    """
    base_provider = build_provider(cfg.provider)
    provider: Provider = (
        CachedProvider(base_provider, Cache(cfg.cache.path)) if cfg.cache.enabled else base_provider
    )
    judge = build_judge(cfg.judge) if cfg.judge is not None else None
    metrics = [build_metric(m, judge=judge) for m in cfg.metrics]
    return EvalRun(
        provider,
        build_dataset(cfg.dataset),
        metrics,
        model=cfg.provider.model,
        params=cfg.provider.params,
    )
