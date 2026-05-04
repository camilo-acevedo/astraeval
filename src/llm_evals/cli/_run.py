"""Implementation of the ``llm-evals run`` subcommand."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from llm_evals.config.builder import build_eval_run
from llm_evals.config.loader import load_yaml
from llm_evals.config.schema import RunConfig
from llm_evals.core.eval_run import RunResult
from llm_evals.exceptions import ThresholdError
from llm_evals.reports.html_report import write_html
from llm_evals.reports.json_report import write_run


def run_command(args: argparse.Namespace) -> int:
    """Execute the ``llm-evals run`` workflow and return an exit code.

    Loads the YAML configuration referenced by ``args.config``, applies any
    CLI overrides (``--output-dir``, ``--no-cache``), executes the run,
    persists configured reports under the output directory, prints a
    human-readable summary, and reports threshold violations as a
    :class:`ThresholdError` so the top-level entry point can emit exit
    code 1.

    :param args: Parsed command-line arguments produced by
        :func:`llm_evals.cli.build_parser`.
    :type args: argparse.Namespace
    :returns: Process exit code.
    :rtype: int
    :raises llm_evals.exceptions.ThresholdError: When at least one
        configured threshold was not met.
    :raises llm_evals.exceptions.LLMEvalsError: For any other domain error
        encountered during loading or execution.
    """
    config = load_yaml(args.config)
    config = _apply_overrides(config, args)

    run = build_eval_run(config)
    result = run.execute()

    output_dir = Path(config.output.dir)
    run_dir = write_run(result, output_dir)
    if "html" in config.output.formats:
        write_html(result, run_dir / "report.html")

    failures = check_thresholds(result.summary, config.thresholds)
    _print_summary(result, config, run_dir, failures)
    if failures:
        raise ThresholdError(_format_failures(failures))
    return 0


def check_thresholds(
    summary: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> list[str]:
    """Return a list of human-readable threshold-violation messages.

    A missing metric in ``summary`` is treated as a violation so the user
    learns about typos in the threshold map immediately.

    :param summary: Aggregate metric scores produced by the run.
    :type summary: collections.abc.Mapping[str, float]
    :param thresholds: Mapping from metric name to minimum acceptable score.
    :type thresholds: collections.abc.Mapping[str, float]
    :returns: One message per failing metric. Empty list when every
        threshold is satisfied.
    :rtype: list[str]
    """
    failures: list[str] = []
    for metric_name, minimum in thresholds.items():
        score = summary.get(metric_name)
        if score is None:
            failures.append(
                f"{metric_name}: not produced by this run (cannot compare to {minimum:.3f})"
            )
        elif score < minimum:
            failures.append(f"{metric_name}: {score:.3f} < {minimum:.3f}")
    return failures


def _apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    """Return ``config`` with CLI overrides applied.

    :param config: Original configuration parsed from YAML.
    :type config: RunConfig
    :param args: Parsed CLI arguments.
    :type args: argparse.Namespace
    :returns: Configuration with overrides applied.
    :rtype: RunConfig
    """
    output = config.output
    if args.output_dir is not None:
        output = replace(output, dir=args.output_dir)

    cache = config.cache
    if args.no_cache:
        cache = replace(cache, enabled=False)

    return replace(config, output=output, cache=cache)


def _print_summary(
    result: RunResult,
    config: RunConfig,
    run_dir: Path,
    failures: list[str],
) -> None:
    """Print a concise human-readable summary to stdout.

    :param result: Aggregate run result.
    :type result: RunResult
    :param config: Resolved run configuration including thresholds.
    :type config: RunConfig
    :param run_dir: Directory where the run artifacts were written.
    :type run_dir: pathlib.Path
    :param failures: Threshold-violation messages.
    :type failures: list[str]
    """
    manifest = result.manifest
    print(
        f"Run {manifest.run_id} ({manifest.provider}/{manifest.model}, "
        f"{manifest.sample_count} samples)"
    )
    width = max((len(name) for name in result.summary), default=0)
    failing = {f.split(":", 1)[0] for f in failures}
    for metric_name in sorted(result.summary):
        score = result.summary[metric_name]
        status = ""
        if metric_name in config.thresholds:
            status = "[FAIL]" if metric_name in failing else "[OK]"
        print(f"  {metric_name.ljust(width)}  {score:.3f}  {status}".rstrip())
    print(f"Reports written to {run_dir}")
    if failures:
        print(_format_failures(failures), file=sys.stderr)


def _format_failures(failures: list[str]) -> str:
    """Format a list of threshold-violation messages as a single string.

    :param failures: Messages produced by :func:`check_thresholds`.
    :type failures: list[str]
    :returns: Multi-line string ready to print or wrap in an exception.
    :rtype: str
    """
    header = f"{len(failures)} threshold violation(s):"
    body = "\n".join(f"  - {msg}" for msg in failures)
    return f"{header}\n{body}"
