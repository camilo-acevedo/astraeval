"""Tests for :mod:`astraeval.providers.base`."""

from __future__ import annotations

from astraeval.providers.base import Provider, hash_request


def test_hash_request_is_deterministic() -> None:
    """The same inputs in different dict orders yield the same hash."""
    a = hash_request(
        provider="fake",
        model="m",
        prompt="hello",
        params={"temperature": 0.0, "max_tokens": 10},
    )
    b = hash_request(
        provider="fake",
        model="m",
        prompt="hello",
        params={"max_tokens": 10, "temperature": 0.0},
    )
    assert a == b
    assert len(a) == 64


def test_hash_request_changes_with_inputs() -> None:
    """Any change to the inputs produces a different hash."""
    base = hash_request(provider="fake", model="m", prompt="p", params={"t": 0})
    assert base != hash_request(provider="other", model="m", prompt="p", params={"t": 0})
    assert base != hash_request(provider="fake", model="m2", prompt="p", params={"t": 0})
    assert base != hash_request(provider="fake", model="m", prompt="p2", params={"t": 0})
    assert base != hash_request(provider="fake", model="m", prompt="p", params={"t": 1})


def test_provider_declares_complete_as_abstract() -> None:
    """``Provider`` requires concrete subclasses to implement ``complete``."""
    assert "complete" in Provider.__abstractmethods__
