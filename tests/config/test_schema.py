"""Tests for :mod:`llm_evals.config.schema`."""

from __future__ import annotations

import pytest

from llm_evals.config.schema import (
    CacheConfig,
    DatasetConfig,
    JudgeConfig,
    OutputConfig,
    ProviderConfig,
    RunConfig,
)
from llm_evals.exceptions import ConfigError

_MIN_RUN: dict[str, object] = {
    "provider": {"type": "fake", "model": "m", "responses": ["x"]},
    "dataset": {"type": "jsonl", "path": "data.jsonl"},
    "metrics": [{"type": "exact_match"}],
}


def test_run_config_minimum_valid_input() -> None:
    """The smallest acceptable YAML produces a fully populated :class:`RunConfig`."""
    config = RunConfig.from_dict(_MIN_RUN)

    assert config.provider.type == "fake"
    assert config.dataset.path == "data.jsonl"
    assert len(config.metrics) == 1
    assert config.metrics[0].type == "exact_match"
    assert config.cache.enabled is True
    assert config.judge is None
    assert config.thresholds == {}
    assert config.output.formats == ("json",)


def test_run_config_full_input_round_trip() -> None:
    """All optional sections populate the resulting dataclass exactly."""
    raw = {
        **_MIN_RUN,
        "cache": {"enabled": False, "path": "x.sqlite"},
        "judge": {
            "type": "fake",
            "model": "judge-m",
            "responses": ['{"claims": []}'],
            "params": {"temperature": 0.2},
        },
        "thresholds": {"exact_match": 0.9},
        "output": {"dir": "out", "formats": ["json", "html"]},
    }

    config = RunConfig.from_dict(raw)

    assert config.cache.enabled is False
    assert config.cache.path == "x.sqlite"
    assert isinstance(config.judge, JudgeConfig)
    assert config.judge.provider.type == "fake"
    assert config.judge.params == {"temperature": 0.2}
    assert config.thresholds == {"exact_match": 0.9}
    assert config.output.formats == ("json", "html")


def test_provider_unknown_type_rejected() -> None:
    """Unknown provider types surface with the allowed set in the error message."""
    raw = {**_MIN_RUN, "provider": {"type": "unknown", "model": "m"}}

    with pytest.raises(ConfigError, match=r"provider\.type"):
        RunConfig.from_dict(raw)


def test_metric_unknown_type_rejected() -> None:
    """Unknown metric types surface with the allowed set in the error message."""
    raw = {**_MIN_RUN, "metrics": [{"type": "made_up"}]}

    with pytest.raises(ConfigError, match=r"metrics\[0\]\.type"):
        RunConfig.from_dict(raw)


def test_metric_options_pass_through() -> None:
    """Non-``type`` keys in a metric entry land in :attr:`MetricConfig.options`."""
    raw = {**_MIN_RUN, "metrics": [{"type": "exact_match", "normalize": False}]}

    config = RunConfig.from_dict(raw)

    assert config.metrics[0].options == {"normalize": False}


def test_thresholds_must_be_numeric() -> None:
    """Booleans, strings, and other non-numeric values are rejected explicitly."""
    raw = {**_MIN_RUN, "thresholds": {"exact_match": "high"}}

    with pytest.raises(ConfigError, match="numeric"):
        RunConfig.from_dict(raw)


def test_thresholds_boolean_rejected() -> None:
    """``True``/``False`` are rejected even though Python treats them as ints."""
    raw = {**_MIN_RUN, "thresholds": {"exact_match": True}}

    with pytest.raises(ConfigError, match="numeric"):
        RunConfig.from_dict(raw)


def test_empty_metrics_list_rejected() -> None:
    """An empty ``metrics`` list is treated as a configuration error."""
    raw = {**_MIN_RUN, "metrics": []}

    with pytest.raises(ConfigError, match="metrics"):
        RunConfig.from_dict(raw)


def test_output_unknown_format_rejected() -> None:
    """Formats outside ``{json, html}`` are rejected."""
    raw = {**_MIN_RUN, "output": {"formats": ["pdf"]}}

    with pytest.raises(ConfigError, match="formats"):
        RunConfig.from_dict(raw)


def test_provider_config_optional_fields_default_to_none() -> None:
    """Optional fields default to ``None`` rather than empty strings."""
    cfg = ProviderConfig.from_dict({"type": "openai", "model": "gpt"})

    assert cfg.api_key is None
    assert cfg.base_url is None
    assert cfg.host is None
    assert cfg.default_max_tokens is None


def test_dataset_config_unknown_loader_rejected() -> None:
    """Unknown dataset loaders surface clearly."""
    with pytest.raises(ConfigError, match=r"dataset\.type"):
        DatasetConfig.from_dict({"type": "csv", "path": "x.csv"})


def test_cache_config_defaults() -> None:
    """Defaults match the documentation (enabled, sibling SQLite file)."""
    cfg = CacheConfig.from_dict({})

    assert cfg.enabled is True
    assert cfg.path == ".llm-evals-cache.sqlite"


def test_output_config_defaults() -> None:
    """Defaults pick a ``runs`` directory and JSON-only output."""
    cfg = OutputConfig.from_dict({})

    assert cfg.dir == "runs"
    assert cfg.formats == ("json",)


def test_metric_config_index_appears_in_error() -> None:
    """The error message tells the user which list entry was wrong."""
    raw = {**_MIN_RUN, "metrics": [{"type": "exact_match"}, {"type": "bogus"}]}

    with pytest.raises(ConfigError, match=r"metrics\[1\]"):
        RunConfig.from_dict(raw)
