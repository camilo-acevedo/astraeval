"""Tests for :mod:`astraeval.cli`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astraeval.cli import EXIT_ERROR, EXIT_OK, EXIT_THRESHOLD, main

_CONFIG_TEMPLATE = """\
provider:
  type: fake
  model: m
  responses:
    - "yes"
    - "no"
dataset:
  type: jsonl
  path: {dataset_path}
metrics:
  - type: exact_match
cache:
  enabled: false
output:
  dir: {output_dir}
  formats: [json, html]
{extra}
"""


def _write_config(
    tmp_path: Path,
    *,
    extra: str = "",
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Create a JSONL dataset and a YAML config pointing at it.

    :param tmp_path: Directory in which to materialise the fixtures.
    :type tmp_path: pathlib.Path
    :param extra: Extra YAML to append after the standard sections.
    :type extra: str
    :param dataset_path: Override for the dataset file location.
    :type dataset_path: pathlib.Path | None
    :param output_dir: Override for the output directory.
    :type output_dir: pathlib.Path | None
    :returns: Path to the YAML config.
    :rtype: pathlib.Path
    """
    dataset = dataset_path or (tmp_path / "data.jsonl")
    dataset.write_text(
        '{"input": "Q1", "expected": "yes"}\n{"input": "Q2", "expected": "yes"}\n',
        encoding="utf-8",
    )
    out_dir = output_dir or (tmp_path / "runs")
    config = tmp_path / "config.yaml"
    config.write_text(
        _CONFIG_TEMPLATE.format(
            dataset_path=dataset.as_posix(),
            output_dir=out_dir.as_posix(),
            extra=extra,
        ),
        encoding="utf-8",
    )
    return config


def test_version_flag_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """``astraeval --version`` prints the package version and exits 0."""
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    captured = capsys.readouterr()
    assert info.value.code == 0
    assert "astraeval" in captured.out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Invoking with no subcommand prints help and exits 0."""
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == EXIT_OK
    assert "run" in captured.out


def test_run_command_writes_reports_and_exits_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A passing run produces JSON + HTML reports and returns exit code 0."""
    config = _write_config(tmp_path)

    exit_code = main(["run", str(config)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_OK
    assert "exact_match" in captured.out
    runs_root = tmp_path / "runs"
    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "samples.jsonl").is_file()
    assert (run_dir / "report.html").is_file()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"] == {"exact_match": 0.5}


def test_threshold_violation_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run that misses a configured threshold returns exit code 1."""
    extra = "thresholds:\n  exact_match: 0.9\n"
    config = _write_config(tmp_path, extra=extra)

    exit_code = main(["run", str(config)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_THRESHOLD
    assert "[FAIL]" in captured.out
    assert "threshold" in captured.err.lower()


def test_threshold_pass_exits_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A satisfied threshold leaves the exit code at 0 and prints ``[OK]``."""
    extra = "thresholds:\n  exact_match: 0.4\n"
    config = _write_config(tmp_path, extra=extra)

    exit_code = main(["run", str(config)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_OK
    assert "[OK]" in captured.out


def test_unknown_metric_in_thresholds_is_a_violation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A threshold pointing at a metric not produced by the run is reported clearly."""
    extra = "thresholds:\n  faithfulness: 0.5\n"
    config = _write_config(tmp_path, extra=extra)

    exit_code = main(["run", str(config)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_THRESHOLD
    assert "faithfulness" in captured.err
    assert "not produced" in captured.err


def test_missing_config_file_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    """A missing config path exits with the generic error code (2)."""
    exit_code = main(["run", "nope.yaml"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_ERROR
    assert "not found" in captured.err.lower()


def test_output_dir_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--output-dir`` overrides the directory declared in the config."""
    config = _write_config(tmp_path)
    override = tmp_path / "custom-out"

    exit_code = main(["run", str(config), "--output-dir", str(override)])
    capsys.readouterr()

    assert exit_code == EXIT_OK
    assert override.is_dir()
    assert any(override.iterdir())
    assert not (tmp_path / "runs").exists()


def test_no_cache_flag_skips_cache_creation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--no-cache`` overrides config.cache.enabled so no SQLite file is written."""
    extra = "cache:\n  enabled: true\n  path: " + (tmp_path / "c.sqlite").as_posix() + "\n"
    config = _write_config(tmp_path, extra=extra)

    exit_code = main(["run", str(config), "--no-cache"])
    capsys.readouterr()

    assert exit_code == EXIT_OK
    assert not (tmp_path / "c.sqlite").exists()
