"""Tests for :mod:`astraeval.datasets.jsonl`."""

from __future__ import annotations

from pathlib import Path

import pytest

from astraeval.datasets.jsonl import load_jsonl
from astraeval.exceptions import DatasetError


def test_loads_minimal_records(tmp_path: Path) -> None:
    """A file with only ``input`` fields produces samples with default extras."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text(
        '{"input": "Q1"}\n{"input": "Q2"}\n',
        encoding="utf-8",
    )

    samples = list(load_jsonl(fixture))

    assert [s.input for s in samples] == ["Q1", "Q2"]
    assert all(s.expected is None for s in samples)
    assert all(s.context == () for s in samples)


def test_loads_full_records(tmp_path: Path) -> None:
    """All optional fields populate the resulting :class:`Sample`."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text(
        '{"input": "Q", "expected": "A", "context": ["c1", "c2"], "metadata": {"id": 1}}\n',
        encoding="utf-8",
    )

    samples = list(load_jsonl(fixture))

    assert len(samples) == 1
    assert samples[0].expected == "A"
    assert samples[0].context == ("c1", "c2")
    assert samples[0].metadata == {"id": 1}


def test_skips_blank_lines(tmp_path: Path) -> None:
    """Empty lines and trailing whitespace are ignored, not treated as records."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text(
        '\n{"input": "Q1"}\n\n   \n{"input": "Q2"}\n',
        encoding="utf-8",
    )

    samples = list(load_jsonl(fixture))

    assert [s.input for s in samples] == ["Q1", "Q2"]


def test_invalid_json_raises_dataset_error(tmp_path: Path) -> None:
    """Malformed JSON surfaces with file path and line number context."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text(
        '{"input": "Q1"}\n{not valid json}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match=":2"):
        list(load_jsonl(fixture))


def test_missing_input_raises_dataset_error(tmp_path: Path) -> None:
    """Records without an ``input`` field are rejected explicitly."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text('{"expected": "A"}\n', encoding="utf-8")

    with pytest.raises(DatasetError, match="input"):
        list(load_jsonl(fixture))


def test_non_list_context_raises_dataset_error(tmp_path: Path) -> None:
    """``context`` values that are not lists are rejected."""
    fixture = tmp_path / "data.jsonl"
    fixture.write_text(
        '{"input": "Q", "context": "not a list"}\n',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="context"):
        list(load_jsonl(fixture))
