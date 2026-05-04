"""Tests for :mod:`astraeval.datasets.sample`."""

from __future__ import annotations

from astraeval.datasets.sample import Sample


def test_default_fields() -> None:
    """Optional fields default to ``None`` and empty container values."""
    sample = Sample(input="What is 2+2?")

    assert sample.input == "What is 2+2?"
    assert sample.expected is None
    assert sample.context == ()
    assert sample.metadata == {}


def test_full_construction() -> None:
    """All fields can be provided and round-trip without mutation."""
    sample = Sample(
        input="Q",
        expected="A",
        context=("ctx1", "ctx2"),
        metadata={"id": 7},
    )

    assert sample.expected == "A"
    assert sample.context == ("ctx1", "ctx2")
    assert sample.metadata == {"id": 7}


def test_equality() -> None:
    """Two samples constructed with identical fields compare equal."""
    a = Sample(input="x", expected="y")
    b = Sample(input="x", expected="y")

    assert a == b
