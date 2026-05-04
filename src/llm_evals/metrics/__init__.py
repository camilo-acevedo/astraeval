"""Evaluation metrics: faithfulness, answer relevance, context precision, hallucination flag."""

from llm_evals.metrics.base import Metric, MetricResult
from llm_evals.metrics.exact_match import ExactMatch

__all__ = ["ExactMatch", "Metric", "MetricResult"]
