"""Implementation of the ``astraeval diff`` subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from astraeval.exceptions import AstraevalError, ThresholdError


def diff_command(args: argparse.Namespace) -> int:
    """Compare two run output directories and report aggregate deltas.

    Reads ``summary.json`` from each directory, prints a side-by-side
    comparison, and (when ``--max-regression`` is set) raises
    :class:`ThresholdError` if any metric drops by more than the allowed
    amount.

    :param args: Parsed command-line arguments produced by
        :func:`astraeval.cli.build_parser`.
    :type args: argparse.Namespace
    :returns: Process exit code.
    :rtype: int
    :raises astraeval.exceptions.ThresholdError: When any metric regressed
        by more than ``args.max_regression``.
    :raises astraeval.exceptions.AstraevalError: When either run directory
        cannot be loaded.
    """
    summary_a = _load_summary(args.run_a)
    summary_b = _load_summary(args.run_b)

    print_diff(summary_a, summary_b)

    if args.max_regression is not None:
        regressions = check_regressions(summary_a, summary_b, args.max_regression)
        if regressions:
            raise ThresholdError(_format_regressions(regressions, args.max_regression))
    return 0


def print_diff(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Print a side-by-side aggregate comparison of two run summaries.

    :param a: Decoded ``summary.json`` for run A (the baseline).
    :type a: dict[str, Any]
    :param b: Decoded ``summary.json`` for run B (the comparison).
    :type b: dict[str, Any]
    """
    metrics_a: dict[str, float] = dict(a.get("metrics", {}))
    metrics_b: dict[str, float] = dict(b.get("metrics", {}))
    all_names = sorted(set(metrics_a) | set(metrics_b))

    name_width = max((len(n) for n in all_names), default=0)
    name_width = max(name_width, len("Metric"))

    print(f"A: {a.get('run_id', '<unknown>')}  ({a.get('provider')}/{a.get('model')})")
    print(f"B: {b.get('run_id', '<unknown>')}  ({b.get('provider')}/{b.get('model')})")
    print(f"Samples: A={a.get('sample_count')}  B={b.get('sample_count')}")
    print()
    header = f"  {'Metric'.ljust(name_width)}  {'A':>8}  {'B':>8}  {'Delta':>8}"
    print(header)
    print(f"  {'-' * name_width}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for name in all_names:
        score_a = metrics_a.get(name)
        score_b = metrics_b.get(name)
        delta = score_b - score_a if score_a is not None and score_b is not None else None
        print(
            f"  {name.ljust(name_width)}  "
            f"{_fmt_score(score_a):>8}  "
            f"{_fmt_score(score_b):>8}  "
            f"{_fmt_delta(delta):>8}"
        )


def check_regressions(
    a: dict[str, Any],
    b: dict[str, Any],
    max_regression: float,
) -> list[tuple[str, float]]:
    """Return ``(metric, drop)`` pairs for metrics that fell more than allowed.

    A metric present in ``a`` but absent in ``b`` is treated as a full
    regression of magnitude ``a[metric]`` so disappearing metrics are not
    silently ignored.

    :param a: Decoded ``summary.json`` for run A (the baseline).
    :type a: dict[str, Any]
    :param b: Decoded ``summary.json`` for run B (the comparison).
    :type b: dict[str, Any]
    :param max_regression: Maximum allowed drop. Must be non-negative.
    :type max_regression: float
    :returns: One entry per regressing metric, sorted by metric name.
    :rtype: list[tuple[str, float]]
    """
    metrics_a: dict[str, float] = dict(a.get("metrics", {}))
    metrics_b: dict[str, float] = dict(b.get("metrics", {}))
    regressions: list[tuple[str, float]] = []
    for name, score_a in sorted(metrics_a.items()):
        score_b = metrics_b.get(name)
        drop = score_a if score_b is None else score_a - score_b
        if drop > max_regression:
            regressions.append((name, drop))
    return regressions


def _load_summary(run_dir: str) -> dict[str, Any]:
    """Load ``summary.json`` from a run output directory.

    :param run_dir: Filesystem path to a per-run output directory produced
        by :func:`astraeval.reports.json_report.write_run`.
    :type run_dir: str
    :returns: Parsed summary JSON document.
    :rtype: dict[str, Any]
    :raises astraeval.exceptions.AstraevalError: When the directory or
        summary file cannot be located, or when the JSON is malformed.
    """
    path = Path(run_dir)
    if not path.is_dir():
        raise AstraevalError(f"Run directory not found: {run_dir}")
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise AstraevalError(f"summary.json missing in {run_dir}")
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AstraevalError(f"Invalid JSON in {summary_path}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise AstraevalError(f"summary.json in {run_dir} must be a JSON object.")
    return data


def _fmt_score(score: float | None) -> str:
    """Render a score for the diff table, using ``-`` for missing values.

    :param score: Aggregate score in ``[0, 1]`` or ``None`` when absent.
    :type score: float | None
    :returns: Right-aligned formatted string.
    :rtype: str
    """
    return "-" if score is None else f"{score:.3f}"


def _fmt_delta(delta: float | None) -> str:
    """Render a delta with a leading sign, or ``-`` when one side is missing.

    :param delta: Difference between B and A or ``None`` when one summary
        omitted the metric.
    :type delta: float | None
    :returns: Right-aligned formatted string.
    :rtype: str
    """
    if delta is None:
        return "-"
    return f"{delta:+.3f}"


def _format_regressions(
    regressions: list[tuple[str, float]],
    max_regression: float,
) -> str:
    """Format the list of regressions as a single multi-line message.

    :param regressions: Pairs of ``(metric, drop)`` produced by
        :func:`check_regressions`.
    :type regressions: list[tuple[str, float]]
    :param max_regression: Threshold the regressions exceeded.
    :type max_regression: float
    :returns: Multi-line string suitable for inclusion in an exception.
    :rtype: str
    """
    header = f"{len(regressions)} metric(s) regressed by more than {max_regression:+.3f}:"
    body = "\n".join(f"  - {name}: -{drop:.3f}" for name, drop in regressions)
    return f"{header}\n{body}"
