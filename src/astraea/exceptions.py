"""Exception hierarchy for the ``astraea`` package."""

from __future__ import annotations


class AstraeaError(Exception):
    """Base class for every error raised by ``astraea``."""


class ProviderError(AstraeaError):
    """Raised when a provider fails to complete a request."""


class CacheError(AstraeaError):
    """Raised when reading from or writing to the request cache fails."""


class DatasetError(AstraeaError):
    """Raised when a dataset cannot be loaded or parsed."""


class MetricError(AstraeaError):
    """Raised when a metric cannot evaluate the inputs it was given."""


class ConfigError(AstraeaError):
    """Raised when a YAML or programmatic configuration is invalid."""


class ThresholdError(AstraeaError):
    """Raised by the CLI when one or more metric thresholds were violated."""
