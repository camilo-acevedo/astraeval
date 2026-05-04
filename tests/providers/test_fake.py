"""Tests for :mod:`astraea.providers.fake`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from astraea.providers.fake import FakeProvider


def test_static_responses_are_returned_in_order() -> None:
    """Successive calls consume the canned response list sequentially."""
    provider = FakeProvider(["one", "two", "three"])
    assert provider.complete("a", model="m").text == "one"
    assert provider.complete("b", model="m").text == "two"
    assert provider.complete("c", model="m").text == "three"


def test_exhausted_static_responses_raise() -> None:
    """Calling beyond the canned list raises a clear error."""
    provider = FakeProvider(["only"])
    provider.complete("p", model="m")
    with pytest.raises(RuntimeError, match="exhausted"):
        provider.complete("p", model="m")


def test_handler_is_invoked_with_prompt_model_and_params() -> None:
    """The handler callable receives the call inputs verbatim."""
    seen: list[tuple[str, str, Mapping[str, Any]]] = []

    def handler(prompt: str, model: str, params: Mapping[str, Any]) -> str:
        seen.append((prompt, model, params))
        return f"echo:{prompt}"

    provider = FakeProvider(handler=handler)
    response = provider.complete("hello", model="m1", temperature=0.5)

    assert response.text == "echo:hello"
    assert response.model == "m1"
    assert response.provider == "fake"
    assert seen == [("hello", "m1", {"temperature": 0.5})]


def test_constructor_requires_exactly_one_source() -> None:
    """Providing neither or both ``responses`` and ``handler`` raises."""
    with pytest.raises(ValueError):
        FakeProvider()
    with pytest.raises(ValueError):
        FakeProvider(["x"], handler=lambda p, m, params: "y")


def test_response_records_latency() -> None:
    """The returned :class:`Response` carries a non-negative latency."""
    provider = FakeProvider(["x"])
    response = provider.complete("p", model="m")
    assert response.latency_ms is not None
    assert response.latency_ms >= 0.0
