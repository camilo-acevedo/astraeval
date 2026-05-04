"""Tests for :mod:`astraeval.metrics.base`."""

from __future__ import annotations

from astraeval.metrics.base import Metric, MetricResult


def test_metric_declares_score_as_abstract() -> None:
    """``Metric`` requires concrete subclasses to implement ``score``."""
    assert "score" in Metric.__abstractmethods__


def test_metric_result_defaults() -> None:
    """Optional fields default to ``None`` and an empty mapping."""
    result = MetricResult(metric="m", score=0.5)

    assert result.reason is None
    assert result.metadata == {}


def test_metric_result_equality() -> None:
    """Two results constructed with identical fields compare equal."""
    a = MetricResult(metric="m", score=1.0, reason="ok")
    b = MetricResult(metric="m", score=1.0, reason="ok")

    assert a == b
