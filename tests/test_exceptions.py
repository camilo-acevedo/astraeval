"""Tests for :mod:`astraeval.exceptions`."""

from __future__ import annotations

from astraeval.exceptions import (
    AstraevalError,
    CacheError,
    DatasetError,
    MetricError,
    ProviderError,
)


def test_all_subclasses_inherit_from_base() -> None:
    """Every domain-specific error subclasses :class:`AstraevalError`."""
    for cls in (ProviderError, CacheError, DatasetError, MetricError):
        assert issubclass(cls, AstraevalError)
