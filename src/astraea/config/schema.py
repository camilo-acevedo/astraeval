"""Dataclasses describing the YAML run configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from astraea.exceptions import ConfigError

_PROVIDER_TYPES = {"anthropic", "openai", "ollama", "fake"}
_DATASET_TYPES = {"jsonl"}
_METRIC_TYPES = {
    "exact_match",
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "hallucination_flag",
}
_OUTPUT_FORMATS = {"json", "html"}


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Configuration for the model under evaluation.

    :ivar type: Provider name, one of ``anthropic``, ``openai``, ``ollama``,
        or ``fake``.
    :vartype type: str
    :ivar model: Model identifier passed to the provider's ``complete``.
    :vartype model: str
    :ivar api_key: Optional API key. Ignored by ``ollama`` and ``fake``.
    :vartype api_key: str | None
    :ivar base_url: Optional base URL override for OpenAI-compatible
        endpoints.
    :vartype base_url: str | None
    :ivar host: Optional Ollama daemon URL.
    :vartype host: str | None
    :ivar default_max_tokens: Default ``max_tokens`` for Anthropic, which
        requires the field on every call.
    :vartype default_max_tokens: int | None
    :ivar responses: Canned responses for the ``fake`` provider.
    :vartype responses: tuple[str, ...]
    :ivar params: Provider parameters forwarded on every call.
    :vartype params: collections.abc.Mapping[str, Any]
    """

    type: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    host: str | None = None
    default_max_tokens: int | None = None
    responses: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, key: str = "provider") -> ProviderConfig:
        """Validate and construct a :class:`ProviderConfig` from a mapping.

        :param data: Raw mapping from the parsed YAML.
        :type data: collections.abc.Mapping[str, Any]
        :param key: Section name used in error messages.
        :type key: str
        :returns: Validated configuration.
        :rtype: ProviderConfig
        :raises astraea.exceptions.ConfigError: When required fields are
            missing or values fall outside the expected domain.
        """
        provider_type = _require_str(data, "type", section=key)
        if provider_type not in _PROVIDER_TYPES:
            raise ConfigError(
                f"{key}.type must be one of {sorted(_PROVIDER_TYPES)}, got {provider_type!r}."
            )
        model = _require_str(data, "model", section=key)
        responses_raw = data.get("responses", [])
        if not isinstance(responses_raw, list):
            raise ConfigError(f"{key}.responses must be a list of strings.")
        params = _require_mapping(data, "params", default={}, section=key)
        return cls(
            type=provider_type,
            model=model,
            api_key=_optional_str(data, "api_key", section=key),
            base_url=_optional_str(data, "base_url", section=key),
            host=_optional_str(data, "host", section=key),
            default_max_tokens=_optional_int(data, "default_max_tokens", section=key),
            responses=tuple(str(r) for r in responses_raw),
            params=dict(params),
        )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Configuration for the dataset under evaluation.

    :ivar type: Loader name, currently only ``jsonl``.
    :vartype type: str
    :ivar path: Path to the dataset file.
    :vartype path: str
    """

    type: str
    path: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetConfig:
        """Validate and construct a :class:`DatasetConfig`.

        :param data: Raw mapping from the parsed YAML.
        :type data: collections.abc.Mapping[str, Any]
        :returns: Validated configuration.
        :rtype: DatasetConfig
        :raises astraea.exceptions.ConfigError: When the dataset section
            is malformed or references an unknown loader.
        """
        dataset_type = _require_str(data, "type", section="dataset")
        if dataset_type not in _DATASET_TYPES:
            raise ConfigError(
                f"dataset.type must be one of {sorted(_DATASET_TYPES)}, got {dataset_type!r}."
            )
        return cls(type=dataset_type, path=_require_str(data, "path", section="dataset"))


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Configuration for the SQLite request cache.

    :ivar enabled: Toggle for the cache. When ``False`` calls bypass it.
    :vartype enabled: bool
    :ivar path: Filesystem path to the SQLite database file.
    :vartype path: str
    """

    enabled: bool = True
    path: str = ".astraea-cache.sqlite"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CacheConfig:
        """Validate and construct a :class:`CacheConfig`.

        :param data: Raw mapping from the parsed YAML.
        :type data: collections.abc.Mapping[str, Any]
        :returns: Validated configuration.
        :rtype: CacheConfig
        :raises astraea.exceptions.ConfigError: When values are present
            but of the wrong type.
        """
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError("cache.enabled must be a boolean.")
        path = data.get("path", ".astraea-cache.sqlite")
        if not isinstance(path, str):
            raise ConfigError("cache.path must be a string.")
        return cls(enabled=enabled, path=path)


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    """Configuration for the LLM judge used by faithfulness-style metrics.

    Has the same shape as :class:`ProviderConfig` but is optional and only
    used by metrics that require a judge.

    :ivar provider: Underlying provider configuration for the judge.
    :vartype provider: ProviderConfig
    :ivar params: Provider parameters used on every judge call. When
        omitted defaults to ``{"temperature": 0.0}``.
    :vartype params: collections.abc.Mapping[str, Any]
    """

    provider: ProviderConfig
    params: Mapping[str, Any] = field(default_factory=lambda: {"temperature": 0.0})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> JudgeConfig:
        """Validate and construct a :class:`JudgeConfig`.

        :param data: Raw mapping from the parsed YAML.
        :type data: collections.abc.Mapping[str, Any]
        :returns: Validated configuration.
        :rtype: JudgeConfig
        :raises astraea.exceptions.ConfigError: When required fields are
            missing.
        """
        provider = ProviderConfig.from_dict(data, key="judge")
        params_raw = _require_mapping(data, "params", default={"temperature": 0.0}, section="judge")
        return cls(provider=provider, params=dict(params_raw))


@dataclass(frozen=True, slots=True)
class MetricConfig:
    """Configuration entry for a single metric.

    :ivar type: Metric identifier such as ``faithfulness`` or
        ``exact_match``.
    :vartype type: str
    :ivar options: Constructor keyword arguments specific to the metric.
    :vartype options: collections.abc.Mapping[str, Any]
    """

    type: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int) -> MetricConfig:
        """Validate and construct a :class:`MetricConfig`.

        :param data: Raw mapping from the parsed YAML.
        :type data: collections.abc.Mapping[str, Any]
        :param index: Position in the metrics list, used in error messages.
        :type index: int
        :returns: Validated configuration.
        :rtype: MetricConfig
        :raises astraea.exceptions.ConfigError: When ``type`` is missing
            or unknown.
        """
        section = f"metrics[{index}]"
        metric_type = _require_str(data, "type", section=section)
        if metric_type not in _METRIC_TYPES:
            raise ConfigError(
                f"{section}.type must be one of {sorted(_METRIC_TYPES)}, got {metric_type!r}."
            )
        options = {k: v for k, v in data.items() if k != "type"}
        return cls(type=metric_type, options=options)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Configuration for run artifacts.

    :ivar dir: Directory under which a per-run subdirectory is created.
    :vartype dir: str
    :ivar formats: Set of report formats to emit. Subset of ``{"json",
        "html"}``.
    :vartype formats: tuple[str, ...]
    """

    dir: str = "runs"
    formats: tuple[str, ...] = ("json",)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OutputConfig:
        """Validate and construct an :class:`OutputConfig`.

        :param data: Raw mapping from the parsed YAML.
        :type data: collections.abc.Mapping[str, Any]
        :returns: Validated configuration.
        :rtype: OutputConfig
        :raises astraea.exceptions.ConfigError: When ``formats`` is not a
            list of recognised values.
        """
        out_dir = data.get("dir", "runs")
        if not isinstance(out_dir, str):
            raise ConfigError("output.dir must be a string.")
        formats_raw = data.get("formats", ["json"])
        if not isinstance(formats_raw, list) or not all(isinstance(f, str) for f in formats_raw):
            raise ConfigError("output.formats must be a list of strings.")
        unknown = set(formats_raw) - _OUTPUT_FORMATS
        if unknown:
            raise ConfigError(
                f"output.formats contains unknown values: {sorted(unknown)}. "
                f"Allowed: {sorted(_OUTPUT_FORMATS)}."
            )
        return cls(dir=out_dir, formats=tuple(formats_raw))


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Top-level run configuration assembled from a YAML document.

    :ivar provider: Provider for the model under evaluation.
    :vartype provider: ProviderConfig
    :ivar dataset: Dataset to score against.
    :vartype dataset: DatasetConfig
    :ivar metrics: Metrics to apply, in order.
    :vartype metrics: tuple[MetricConfig, ...]
    :ivar cache: Optional cache configuration. Defaults to enabled.
    :vartype cache: CacheConfig
    :ivar judge: Optional judge configuration for LLM-as-judge metrics.
    :vartype judge: JudgeConfig | None
    :ivar thresholds: Mapping from metric name to minimum aggregate score.
    :vartype thresholds: collections.abc.Mapping[str, float]
    :ivar output: Output directory and format selection.
    :vartype output: OutputConfig
    """

    provider: ProviderConfig
    dataset: DatasetConfig
    metrics: tuple[MetricConfig, ...]
    cache: CacheConfig = field(default_factory=CacheConfig)
    judge: JudgeConfig | None = None
    thresholds: Mapping[str, float] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        """Validate and construct a :class:`RunConfig` from a mapping.

        :param data: Raw mapping from the parsed YAML.
        :type data: collections.abc.Mapping[str, Any]
        :returns: Validated run configuration.
        :rtype: RunConfig
        :raises astraea.exceptions.ConfigError: When required sections
            are missing or any subsection fails to validate.
        """
        if not isinstance(data, Mapping):
            raise ConfigError("Top-level configuration must be a mapping.")

        provider_data = _require_mapping(data, "provider")
        dataset_data = _require_mapping(data, "dataset")
        metrics_raw = data.get("metrics")
        if not isinstance(metrics_raw, list) or not metrics_raw:
            raise ConfigError("metrics must be a non-empty list.")

        metrics = tuple(
            MetricConfig.from_dict(m, index=i)
            for i, m in enumerate(_iter_mappings(metrics_raw, key="metrics"))
        )

        cache = CacheConfig.from_dict(data["cache"]) if "cache" in data else CacheConfig()
        judge = JudgeConfig.from_dict(data["judge"]) if "judge" in data else None
        output = OutputConfig.from_dict(data["output"]) if "output" in data else OutputConfig()
        thresholds = _build_thresholds(data.get("thresholds", {}))

        return cls(
            provider=ProviderConfig.from_dict(provider_data),
            dataset=DatasetConfig.from_dict(dataset_data),
            metrics=metrics,
            cache=cache,
            judge=judge,
            thresholds=thresholds,
            output=output,
        )


def _build_thresholds(raw: Any) -> dict[str, float]:
    """Validate and coerce the thresholds mapping.

    :param raw: Raw value of the ``thresholds`` field.
    :type raw: Any
    :returns: Mapping from metric name to a finite float threshold.
    :rtype: dict[str, float]
    :raises astraea.exceptions.ConfigError: When the structure is wrong
        or any value is not numeric.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError("thresholds must be a mapping of metric_name to number.")
    result: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ConfigError("thresholds keys must be strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"thresholds[{key!r}] must be numeric; got {type(value).__name__}.")
        result[key] = float(value)
    return result


def _iter_mappings(items: Sequence[Any], *, key: str) -> Sequence[Mapping[str, Any]]:
    """Validate that every entry in ``items`` is a mapping.

    :param items: Raw list to validate.
    :type items: collections.abc.Sequence[Any]
    :param key: Section name used in error messages.
    :type key: str
    :returns: The same sequence, narrowed in type.
    :rtype: collections.abc.Sequence[collections.abc.Mapping[str, Any]]
    :raises astraea.exceptions.ConfigError: When any entry is not a
        mapping.
    """
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ConfigError(f"{key}[{index}] must be a mapping.")
    return items


def _require_str(data: Mapping[str, Any], field_name: str, *, section: str) -> str:
    """Return ``data[field_name]`` ensuring it is a non-empty string.

    :param data: Source mapping.
    :type data: collections.abc.Mapping[str, Any]
    :param field_name: Name of the field to read.
    :type field_name: str
    :param section: Section name used in error messages.
    :type section: str
    :returns: The string value.
    :rtype: str
    :raises astraea.exceptions.ConfigError: When the field is missing,
        empty, or not a string.
    """
    if field_name not in data:
        raise ConfigError(f"{section}.{field_name} is required.")
    value = data[field_name]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{section}.{field_name} must be a non-empty string.")
    return value


def _optional_str(data: Mapping[str, Any], field_name: str, *, section: str) -> str | None:
    """Return ``data[field_name]`` when present and a string, else ``None``.

    :param data: Source mapping.
    :type data: collections.abc.Mapping[str, Any]
    :param field_name: Name of the field to read.
    :type field_name: str
    :param section: Section name used in error messages.
    :type section: str
    :returns: The string value or ``None``.
    :rtype: str | None
    :raises astraea.exceptions.ConfigError: When the field is present
        but not a string.
    """
    if field_name not in data:
        return None
    value = data[field_name]
    if not isinstance(value, str):
        raise ConfigError(f"{section}.{field_name} must be a string when provided.")
    return value


def _optional_int(data: Mapping[str, Any], field_name: str, *, section: str) -> int | None:
    """Return ``data[field_name]`` when present and an integer, else ``None``.

    :param data: Source mapping.
    :type data: collections.abc.Mapping[str, Any]
    :param field_name: Name of the field to read.
    :type field_name: str
    :param section: Section name used in error messages.
    :type section: str
    :returns: The integer value or ``None``.
    :rtype: int | None
    :raises astraea.exceptions.ConfigError: When the field is present
        but not an integer.
    """
    if field_name not in data:
        return None
    value = data[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{section}.{field_name} must be an integer when provided.")
    return int(value)


def _require_mapping(
    data: Mapping[str, Any],
    field_name: str,
    *,
    default: Mapping[str, Any] | None = None,
    section: str | None = None,
) -> Mapping[str, Any]:
    """Return ``data[field_name]`` as a mapping, applying ``default`` when missing.

    :param data: Source mapping.
    :type data: collections.abc.Mapping[str, Any]
    :param field_name: Name of the field to read.
    :type field_name: str
    :param default: Default value when the field is absent. ``None`` means
        the field is required.
    :type default: collections.abc.Mapping[str, Any] | None
    :param section: Section name used in error messages. Defaults to
        ``field_name``.
    :type section: str | None
    :returns: The mapping value.
    :rtype: collections.abc.Mapping[str, Any]
    :raises astraea.exceptions.ConfigError: When the field is missing
        without a default or is not a mapping.
    """
    label = section if section is not None else field_name
    if field_name not in data:
        if default is None:
            raise ConfigError(f"{label}.{field_name} is required.")
        return default
    value = data[field_name]
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label}.{field_name} must be a mapping.")
    return value
