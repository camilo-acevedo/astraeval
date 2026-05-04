"""Tests for :mod:`llm_evals.exceptions`."""

from __future__ import annotations

from llm_evals.exceptions import (
    CacheError,
    DatasetError,
    LLMEvalsError,
    MetricError,
    ProviderError,
)


def test_all_subclasses_inherit_from_base() -> None:
    """Every domain-specific error subclasses :class:`LLMEvalsError`."""
    for cls in (ProviderError, CacheError, DatasetError, MetricError):
        assert issubclass(cls, LLMEvalsError)
