"""Tests for :mod:`astraea.exceptions`."""

from __future__ import annotations

from astraea.exceptions import (
    AstraeaError,
    CacheError,
    DatasetError,
    MetricError,
    ProviderError,
)


def test_all_subclasses_inherit_from_base() -> None:
    """Every domain-specific error subclasses :class:`AstraeaError`."""
    for cls in (ProviderError, CacheError, DatasetError, MetricError):
        assert issubclass(cls, AstraeaError)
