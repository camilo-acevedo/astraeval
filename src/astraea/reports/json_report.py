"""JSON report writer producing the canonical on-disk layout for a run."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from astraea.core.eval_run import RunResult


def write_run(result: RunResult, output_dir: str | Path) -> Path:
    """Persist a :class:`RunResult` to disk under a per-run subdirectory.

    Creates ``output_dir/<iso-timestamp>_<run_id>/`` containing:

    - ``manifest.json``: full :class:`astraea.core.manifest.RunManifest`
    - ``summary.json``: compact aggregate scores designed for CI parsing
    - ``samples.jsonl``: one JSON object per line, one line per sample,
      with the sample, response, and per-metric verdicts

    :param result: Aggregate run result to persist.
    :type result: RunResult
    :param output_dir: Base directory under which the per-run subdirectory
        is created. Created recursively if missing.
    :type output_dir: str | pathlib.Path
    :returns: Path to the per-run subdirectory.
    :rtype: pathlib.Path
    """
    base = Path(output_dir)
    run_dir = base / _run_subdir_name(result)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(result, run_dir / "manifest.json")
    write_summary(result, run_dir / "summary.json")
    write_samples(result, run_dir / "samples.jsonl")

    return run_dir


def run_subdir_name(result: RunResult) -> str:
    """Compute the per-run subdirectory name used by :func:`write_run`.

    Exposed so callers (the CLI, downstream tooling) can predict where a
    run will be written without re-implementing the same naming scheme.

    :param result: Run result whose manifest carries the timestamp and ID.
    :type result: RunResult
    :returns: Filesystem-safe directory name component.
    :rtype: str
    """
    return _run_subdir_name(result)


def write_manifest(result: RunResult, path: Path) -> None:
    """Write the :class:`RunManifest` as a single JSON document.

    :param result: Aggregate run result whose manifest should be written.
    :type result: RunResult
    :param path: Destination file path.
    :type path: pathlib.Path
    """
    path.write_text(result.manifest.to_json(indent=2), encoding="utf-8")


def write_summary(result: RunResult, path: Path) -> None:
    """Write a compact summary suitable for CI gates and dashboards.

    The summary contains the run identity, provider/model used, sample
    count, per-metric aggregate scores, and ISO-8601 timestamps.

    :param result: Aggregate run result.
    :type result: RunResult
    :param path: Destination file path.
    :type path: pathlib.Path
    """
    payload: dict[str, Any] = {
        "run_id": result.manifest.run_id,
        "provider": result.manifest.provider,
        "model": result.manifest.model,
        "sample_count": result.manifest.sample_count,
        "metrics": dict(result.summary),
        "started_at": result.manifest.started_at,
        "finished_at": result.manifest.finished_at,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_samples(result: RunResult, path: Path) -> None:
    """Write one JSON object per line, one line per sample.

    The JSONL format keeps each row independently parseable so it can be
    streamed, filtered with ``jq``, or loaded one-at-a-time when the run
    is large.

    :param result: Aggregate run result.
    :type result: RunResult
    :param path: Destination file path. Existing content is overwritten.
    :type path: pathlib.Path
    """
    with path.open("w", encoding="utf-8") as stream:
        for sample_result in result.samples:
            line = {
                "sample": asdict(sample_result.sample),
                "response": asdict(sample_result.response),
                "metrics": [asdict(metric) for metric in sample_result.metrics],
            }
            stream.write(json.dumps(line, sort_keys=True))
            stream.write("\n")


def _run_subdir_name(result: RunResult) -> str:
    """Compose the per-run directory name as ``<filesystem-safe-ts>_<run_id>``.

    Replaces colons with hyphens and the trailing ``+00:00`` with ``Z`` so
    the name is safe on Windows.

    :param result: Run result whose manifest carries the timestamp and ID.
    :type result: RunResult
    :returns: Directory name component.
    :rtype: str
    """
    safe_ts = result.manifest.started_at.replace(":", "-").replace("+00-00", "Z")
    return f"{safe_ts}_{result.manifest.run_id}"
