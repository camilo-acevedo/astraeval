"""Exception hierarchy for the ``astraeval`` package."""

from __future__ import annotations


class AstraevalError(Exception):
    """Base class for every error raised by ``astraeval``."""


class ProviderError(AstraevalError):
    """Raised when a provider fails to complete a request."""


class CacheError(AstraevalError):
    """Raised when reading from or writing to the request cache fails."""


class DatasetError(AstraevalError):
    """Raised when a dataset cannot be loaded or parsed."""


class MetricError(AstraevalError):
    """Raised when a metric cannot evaluate the inputs it was given."""


class ConfigError(AstraevalError):
    """Raised when a YAML or programmatic configuration is invalid."""


class ThresholdError(AstraevalError):
    """Raised by the CLI when one or more metric thresholds were violated."""
