"""Tests for :mod:`astraea.providers.base`."""

from __future__ import annotations

from astraea.providers.base import Provider, request_key


def test_request_key_is_deterministic() -> None:
    """The same inputs in different dict orders yield the same key."""
    a = request_key(
        provider="fake",
        model="m",
        prompt="hello",
        params={"temperature": 0.0, "max_tokens": 10},
    )
    b = request_key(
        provider="fake",
        model="m",
        prompt="hello",
        params={"max_tokens": 10, "temperature": 0.0},
    )
    assert a == b
    assert len(a) == 64


def test_request_key_changes_with_inputs() -> None:
    """Any change to the inputs produces a different key."""
    base = request_key(provider="fake", model="m", prompt="p", params={"t": 0})
    assert base != request_key(provider="other", model="m", prompt="p", params={"t": 0})
    assert base != request_key(provider="fake", model="m2", prompt="p", params={"t": 0})
    assert base != request_key(provider="fake", model="m", prompt="p2", params={"t": 0})
    assert base != request_key(provider="fake", model="m", prompt="p", params={"t": 1})


def test_provider_declares_complete_as_abstract() -> None:
    """``Provider`` requires concrete subclasses to implement ``complete``."""
    assert "complete" in Provider.__abstractmethods__
