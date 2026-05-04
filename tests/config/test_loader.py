"""Tests for :mod:`astraea.config.loader`."""

from __future__ import annotations

from pathlib import Path

import pytest

from astraea.config.loader import load_yaml
from astraea.exceptions import ConfigError


def test_load_minimal_yaml(tmp_path: Path) -> None:
    """A minimal YAML file parses into a :class:`RunConfig`."""
    fixture = tmp_path / "config.yaml"
    fixture.write_text(
        """
provider:
  type: fake
  model: m
  responses: [hello]
dataset:
  type: jsonl
  path: ./data.jsonl
metrics:
  - type: exact_match
""",
        encoding="utf-8",
    )

    config = load_yaml(fixture)

    assert config.provider.type == "fake"
    assert config.dataset.path == "./data.jsonl"
    assert config.metrics[0].type == "exact_match"


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    """A missing config path produces :class:`ConfigError`, not :class:`FileNotFoundError`."""
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "absent.yaml")


def test_invalid_yaml_raises_config_error(tmp_path: Path) -> None:
    """Malformed YAML is reported with file context."""
    fixture = tmp_path / "broken.yaml"
    fixture.write_text("provider: [unclosed", encoding="utf-8")

    with pytest.raises(ConfigError, match="parse YAML"):
        load_yaml(fixture)


def test_empty_file_raises_config_error(tmp_path: Path) -> None:
    """A file containing only whitespace is rejected explicitly."""
    fixture = tmp_path / "empty.yaml"
    fixture.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="empty"):
        load_yaml(fixture)


def test_top_level_list_rejected(tmp_path: Path) -> None:
    """A top-level YAML list is rejected with a descriptive message."""
    fixture = tmp_path / "list.yaml"
    fixture.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(fixture)
