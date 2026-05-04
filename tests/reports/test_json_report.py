"""Tests for :mod:`astraeval.reports.json_report`."""

from __future__ import annotations

import json
from pathlib import Path

from astraeval.core.eval_run import EvalRun, RunResult
from astraeval.datasets.sample import Sample
from astraeval.metrics.exact_match import ExactMatch
from astraeval.providers.fake import FakeProvider
from astraeval.reports.json_report import (
    write_manifest,
    write_run,
    write_samples,
    write_summary,
)


def _execute_run() -> RunResult:
    """Run a tiny end-to-end evaluation for the reports under test.

    :returns: A fully populated :class:`RunResult`.
    :rtype: RunResult
    """
    provider = FakeProvider(["yes", "no"])
    samples = [
        Sample(input="Q1", expected="yes"),
        Sample(input="Q2", expected="yes"),
    ]
    return EvalRun(provider, samples, [ExactMatch()], model="m").execute()


def test_write_run_creates_canonical_layout(tmp_path: Path) -> None:
    """``write_run`` emits manifest, summary, and samples under a per-run subdir."""
    result = _execute_run()

    run_dir = write_run(result, tmp_path)

    assert run_dir.parent == tmp_path
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "samples.jsonl").is_file()


def test_run_subdir_name_is_filesystem_safe(tmp_path: Path) -> None:
    """The per-run directory has no characters that break Windows filesystems."""
    result = _execute_run()

    run_dir = write_run(result, tmp_path)

    assert ":" not in run_dir.name
    assert run_dir.name.endswith(result.manifest.run_id)


def test_manifest_json_round_trips(tmp_path: Path) -> None:
    """``manifest.json`` parses back into the same fields the run produced."""
    result = _execute_run()
    out = tmp_path / "manifest.json"

    write_manifest(result, out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["run_id"] == result.manifest.run_id
    assert payload["provider"] == "fake"
    assert payload["metric_names"] == ["exact_match"]
    assert payload["sample_count"] == 2


def test_summary_json_carries_aggregate_scores(tmp_path: Path) -> None:
    """``summary.json`` includes per-metric aggregate scores for CI parsing."""
    result = _execute_run()
    out = tmp_path / "summary.json"

    write_summary(result, out)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["metrics"] == {"exact_match": 0.5}
    assert payload["sample_count"] == 2
    assert "started_at" in payload
    assert "finished_at" in payload


def test_samples_jsonl_one_line_per_sample(tmp_path: Path) -> None:
    """``samples.jsonl`` has exactly one JSON object per line."""
    result = _execute_run()
    out = tmp_path / "samples.jsonl"

    write_samples(result, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["sample"]["input"] == "Q1"
    assert parsed[0]["response"]["text"] == "yes"
    assert parsed[0]["metrics"][0]["metric"] == "exact_match"
    assert parsed[0]["metrics"][0]["score"] == 1.0
    assert parsed[1]["metrics"][0]["score"] == 0.0


def test_write_run_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """The base output directory is created when missing."""
    result = _execute_run()
    nested = tmp_path / "deeply" / "nested" / "runs"

    run_dir = write_run(result, nested)

    assert nested.exists()
    assert run_dir.exists()
