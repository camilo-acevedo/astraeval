"""Tests for :mod:`astraeval.metrics.llm_judge`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from astraeval.core.types import Response
from astraeval.exceptions import MetricError
from astraeval.metrics.llm_judge import LLMJudge, parse_json_object, parse_judge_response
from astraeval.providers.fake import FakeProvider


def test_ask_forwards_prompt_and_default_temperature() -> None:
    """``ask`` calls the provider with the configured model and ``temperature=0``."""
    seen_params: list[Mapping[str, Any]] = []

    def handler(prompt: str, model: str, params: Mapping[str, Any]) -> str:
        seen_params.append(dict(params))
        return f"echo:{prompt}|{model}"

    provider = FakeProvider(handler=handler)
    judge = LLMJudge(provider, model="judge-m")

    response = judge.ask("hello")

    assert response.text == "echo:hello|judge-m"
    assert response.model == "judge-m"
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
    """Garbage text surfaces as :class:`MetricError`, not :class:`ValueError`."""
    with pytest.raises(MetricError, match="valid JSON object"):
        parse_json_object("not json at all")


def test_parse_json_object_rejects_non_object() -> None:
    """Top-level arrays are rejected because metrics expect a mapping."""
    with pytest.raises(MetricError, match="valid JSON object"):
        parse_json_object("[1, 2, 3]")


def test_parse_json_object_extracts_object_from_prose() -> None:
    """A JSON object embedded in commentary is recovered."""
    text = 'Sure, here is my analysis: {"score": 0.8, "reason": "looks good"} -- end.'

    assert parse_json_object(text) == {"score": 0.8, "reason": "looks good"}


def test_parse_json_object_handles_braces_inside_strings() -> None:
    """Brace tracking does not get confused by braces inside string literals."""
    text = '{"text": "this { is not } a brace boundary", "ok": true}'

    assert parse_json_object(text) == {
        "text": "this { is not } a brace boundary",
        "ok": True,
    }


def test_parse_json_object_handles_nested_objects() -> None:
    """Nested objects inside prose are extracted correctly."""
    text = 'Result follows: {"outer": {"inner": [1, 2]}, "k": "v"} done.'

    assert parse_json_object(text) == {"outer": {"inner": [1, 2]}, "k": "v"}


def test_parse_judge_response_passes_through_valid_payload() -> None:
    """``parse_judge_response`` returns the parsed object on the happy path."""
    response = Response(
        text='{"score": 0.7}',
        model="judge",
        provider="fake",
        finish_reason="end_turn",
    )

    assert parse_judge_response(response) == {"score": 0.7}


def test_parse_judge_response_flags_truncation() -> None:
    """A ``max_tokens`` finish reason produces a truncation-specific message."""
    response = Response(
        text='{"claims": [{"text": "a", "supported": tru',  # cut mid-token
        model="judge",
        provider="fake",
        finish_reason="max_tokens",
    )

    with pytest.raises(MetricError, match="truncated"):
        parse_judge_response(response)


def test_parse_judge_response_passes_through_unrelated_errors() -> None:
    """Errors unrelated to truncation surface unchanged."""
    response = Response(
        text="not json at all",
        model="judge",
        provider="fake",
        finish_reason="end_turn",
    )

    with pytest.raises(MetricError, match="valid JSON object"):
        parse_judge_response(response)
