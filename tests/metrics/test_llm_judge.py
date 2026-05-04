"""Tests for :mod:`astraea.metrics.llm_judge`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from astraea.exceptions import MetricError
from astraea.metrics.llm_judge import LLMJudge, parse_json_object
from astraea.providers.fake import FakeProvider


def test_ask_forwards_prompt_and_default_temperature() -> None:
    """``ask`` calls the provider with the configured model and ``temperature=0``."""
    seen_params: list[Mapping[str, Any]] = []

    def handler(prompt: str, model: str, params: Mapping[str, Any]) -> str:
        seen_params.append(dict(params))
        return f"echo:{prompt}|{model}"

    provider = FakeProvider(handler=handler)
    judge = LLMJudge(provider, model="judge-m")

    text = judge.ask("hello")

    assert text == "echo:hello|judge-m"
    assert seen_params == [{"temperature": 0.0}]


def test_explicit_params_override_default() -> None:
    """User-provided params replace the temperature default entirely."""
    seen_params: list[Mapping[str, Any]] = []

    def handler(prompt: str, model: str, params: Mapping[str, Any]) -> str:
        seen_params.append(dict(params))
        return "ok"

    judge = LLMJudge(
        FakeProvider(handler=handler),
        model="m",
        params={"temperature": 0.7, "max_tokens": 64},
    )
    judge.ask("x")

    assert seen_params == [{"temperature": 0.7, "max_tokens": 64}]


def test_provider_and_model_properties() -> None:
    """The configured provider and model are exposed read-only."""
    provider = FakeProvider(["x"])
    judge = LLMJudge(provider, model="judge-m")

    assert judge.provider is provider
    assert judge.model == "judge-m"


def test_parse_json_object_plain() -> None:
    """A plain JSON object parses unchanged."""
    text = '{"a": 1, "b": "two"}'

    assert parse_json_object(text) == {"a": 1, "b": "two"}


def test_parse_json_object_strips_fence_with_language_tag() -> None:
    """Triple-backtick fences with a language tag are stripped before parsing."""
    text = '```json\n{"a": 1}\n```'

    assert parse_json_object(text) == {"a": 1}


def test_parse_json_object_strips_fence_without_language_tag() -> None:
    """Plain triple-backtick fences are stripped before parsing."""
    text = '```\n{"k": true}\n```'

    assert parse_json_object(text) == {"k": True}


def test_parse_json_object_rejects_invalid_json() -> None:
    """Invalid JSON surfaces as :class:`MetricError`, not :class:`ValueError`."""
    with pytest.raises(MetricError, match="valid JSON"):
        parse_json_object("not json at all")


def test_parse_json_object_rejects_non_object() -> None:
    """Top-level lists or scalars are rejected with a descriptive message."""
    with pytest.raises(MetricError, match="non-object"):
        parse_json_object("[1, 2, 3]")
