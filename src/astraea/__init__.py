"""Reproducible LLM evaluation harness with run manifests, prompt hashing, and SQLite caching."""

from importlib.metadata import PackageNotFoundError, version

from astraea.core.eval_run import EvalRun, RunResult, SampleResult
from astraea.core.manifest import RunManifest
from astraea.core.types import Response
from astraea.datasets.sample import Sample
from astraea.exceptions import (
    AstraeaError,
    CacheError,
    ConfigError,
    DatasetError,
    MetricError,
    ProviderError,
    ThresholdError,
)
from astraea.metrics.base import Metric, MetricResult
from astraea.providers.base import Provider

try:
    __version__ = version("astraea")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "AstraeaError",
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
