"""Exception hierarchy for the ``llm_evals`` package."""

from __future__ import annotations


class LLMEvalsError(Exception):
    """Base class for every error raised by ``llm_evals``."""


class ProviderError(LLMEvalsError):
    """Raised when a provider fails to complete a request."""


class CacheError(LLMEvalsError):
    """Raised when reading from or writing to the request cache fails."""


class DatasetError(LLMEvalsError):
    """Raised when a dataset cannot be loaded or parsed."""


class MetricError(LLMEvalsError):
    """Raised when a metric cannot evaluate the inputs it was given."""


class ConfigError(LLMEvalsError):
    """Raised when a YAML or programmatic configuration is invalid."""


class ThresholdError(LLMEvalsError):
    """Raised by the CLI when one or more metric thresholds were violated."""
