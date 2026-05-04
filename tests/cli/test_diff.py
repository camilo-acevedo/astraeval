"""Tests for the ``llm-evals diff`` subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_evals.cli import EXIT_ERROR, EXIT_OK, EXIT_THRESHOLD, main


def _write_run_dir(
    base: Path,
    *,
    run_id: str,
    metrics: dict[str, float],
    sample_count: int = 10,
    provider: str = "fake",
    model: str = "m",
) -> Path:
    """Create a minimal run output directory with a ``summary.json``.

    :param base: Parent directory under which to create the run dir.
    :type base: pathlib.Path
    :param run_id: Identifier embedded in the summary.
    :type run_id: str
    :param metrics: Mapping from metric name to aggregate score.
    :type metrics: dict[str, float]
    :param sample_count: Sample count recorded in the summary.
    :type sample_count: int
    :param provider: Provider name recorded in the summary.
    :type provider: str
    :param model: Model identifier recorded in the summary.
    :type model: str
    :returns: Path to the new run directory.
    :rtype: pathlib.Path
    """
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "sample_count": sample_count,
        "metrics": metrics,
        "started_at": "2026-05-04T00:00:00+00:00",
        "finished_at": "2026-05-04T00:00:01+00:00",
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def test_diff_prints_side_by_side_table(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The diff output shows both runs and a delta column."""
    a = _write_run_dir(tmp_path, run_id="aaa", metrics={"exact_match": 0.5})
    b = _write_run_dir(tmp_path, run_id="bbb", metrics={"exact_match": 0.7})

    exit_code = main(["diff", str(a), str(b)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_OK
    assert "aaa" in captured.out
    assert "bbb" in captured.out
    assert "exact_match" in captured.out
    assert "+0.200" in captured.out


def test_diff_handles_metric_only_in_one_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Metrics present in only one run render a placeholder in the missing column."""
    a = _write_run_dir(tmp_path, run_id="aaa", metrics={"exact_match": 0.5})
    b = _write_run_dir(tmp_path, run_id="bbb", metrics={"exact_match": 0.5, "faithfulness": 0.9})

    exit_code = main(["diff", str(a), str(b)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_OK
    assert "faithfulness" in captured.out


def test_max_regression_passes_when_within_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A small drop within the allowed budget exits 0."""
    a = _write_run_dir(tmp_path, run_id="aaa", metrics={"exact_match": 0.80})
    b = _write_run_dir(tmp_path, run_id="bbb", metrics={"exact_match": 0.78})

    exit_code = main(["diff", str(a), str(b), "--max-regression", "0.05"])
    capsys.readouterr()

    assert exit_code == EXIT_OK


def test_max_regression_fails_when_exceeded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A drop greater than the budget exits 1 (threshold violation)."""
    a = _write_run_dir(tmp_path, run_id="aaa", metrics={"exact_match": 0.80})
    b = _write_run_dir(tmp_path, run_id="bbb", metrics={"exact_match": 0.60})

    exit_code = main(["diff", str(a), str(b), "--max-regression", "0.05"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_THRESHOLD
    assert "regressed" in captured.err
    assert "exact_match" in captured.err


def test_disappearing_metric_counts_as_regression(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A metric absent from run B counts as a full regression of its baseline value."""
    a = _write_run_dir(tmp_path, run_id="aaa", metrics={"faithfulness": 0.9})
    b = _write_run_dir(tmp_path, run_id="bbb", metrics={})

    exit_code = main(["diff", str(a), str(b), "--max-regression", "0.05"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_THRESHOLD
    assert "faithfulness" in captured.err


def test_missing_run_directory_exits_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-existent run directory surfaces as a generic error."""
    a = _write_run_dir(tmp_path, run_id="aaa", metrics={"exact_match": 0.5})

    exit_code = main(["diff", str(a), str(tmp_path / "missing")])
    captured = capsys.readouterr()

    assert exit_code == EXIT_ERROR
    assert "not found" in captured.err.lower()


def test_run_directory_without_summary_exits_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A directory missing ``summary.json`` is reported clearly."""
    incomplete = tmp_path / "no-summary"
    incomplete.mkdir()
    a = _write_run_dir(tmp_path, run_id="aaa", metrics={"exact_match": 0.5})

    exit_code = main(["diff", str(a), str(incomplete)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_ERROR
    assert "summary.json" in captured.err
