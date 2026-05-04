"""Reproducible LLM evaluation harness with run manifests, prompt hashing, and SQLite caching."""

from importlib.metadata import PackageNotFoundError, version

from astraeval.core.eval_run import EvalRun, RunResult, SampleResult
from astraeval.core.manifest import RunManifest
from astraeval.core.types import Response
from astraeval.datasets.sample import Sample
from astraeval.exceptions import (
    AstraevalError,
    CacheError,
    ConfigError,
    DatasetError,
    MetricError,
    ProviderError,
    ThresholdError,
)
from astraeval.metrics.base import Metric, MetricResult
from astraeval.providers.base import Provider

try:
    __version__ = version("astraeval")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "AstraevalError",
    "CacheError",
    "ConfigError",
    "DatasetError",
    "EvalRun",
    "Metric",
    "MetricError",
    "MetricResult",
    "Provider",
    "ProviderError",
    "Response",
    "RunManifest",
    "RunResult",
    "Sample",
    "SampleResult",
    "ThresholdError",
    "__version__",
]
