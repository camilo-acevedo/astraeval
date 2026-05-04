"""YAML loader returning a validated :class:`RunConfig`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from astraeval.config.schema import RunConfig
from astraeval.exceptions import ConfigError


def load_yaml(path: str | Path) -> RunConfig:
    """Read a YAML run configuration from ``path`` and validate it.

    :param path: Filesystem path to the YAML configuration file.
    :type path: str | pathlib.Path
    :returns: Validated :class:`RunConfig`.
    :rtype: RunConfig
    :raises astraeval.exceptions.ConfigError: When the file cannot be read,
        does not parse as YAML, or fails schema validation.
    """
    file_path = Path(path)
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            raw: Any = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {file_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML at {file_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"Configuration file is empty: {file_path}")
    if not isinstance(raw, dict):
        raise ConfigError(
            f"Top-level YAML in {file_path} must be a mapping, got {type(raw).__name__}."
        )
    return RunConfig.from_dict(raw)
