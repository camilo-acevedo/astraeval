"""Command-line interface for the ``astraeval`` package."""

from __future__ import annotations

import argparse
import sys

from astraeval import __version__
from astraeval.cli._diff import diff_command
from astraeval.cli._run import run_command
from astraeval.exceptions import AstraevalError, ThresholdError

EXIT_OK = 0
EXIT_THRESHOLD = 1
EXIT_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with every subcommand wired in.

    :returns: Configured parser ready to call :meth:`parse_args`.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="astraeval",
        description="Astraeval: reproducible LLM evaluation harness.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"astraeval {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    run = subparsers.add_parser("run", help="Execute an evaluation run from a YAML config.")
    run.add_argument("config", help="Path to a YAML run configuration file.")
    run.add_argument(
        "--output-dir",
        default=None,
        help="Override the output directory declared in the config.",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the SQLite request cache regardless of config.",
    )
    run.set_defaults(handler=run_command)

    diff = subparsers.add_parser(
        "diff",
        help="Compare two run output directories and report metric deltas.",
    )
    diff.add_argument("run_a", help="Baseline run directory (output of astraeval run).")
    diff.add_argument("run_b", help="Comparison run directory.")
    diff.add_argument(
        "--max-regression",
        type=float,
        default=None,
        help=(
            "Exit with code 1 when any metric in run B drops by more than this "
            "amount compared to run A."
        ),
    )
    diff.set_defaults(handler=diff_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point used by both the console script and ``python -m astraeval``.

    :param argv: Argument vector to parse. When ``None``, falls back to
        :data:`sys.argv` excluding the program name.
    :type argv: list[str] | None
    :returns: Process exit code. ``0`` on success, ``1`` when a threshold
        was violated, ``2`` on any other error.
    :rtype: int
    """
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK

    try:
        return int(args.handler(args))
    except ThresholdError as exc:
        print(f"threshold violation: {exc}", file=sys.stderr)
        return EXIT_THRESHOLD
    except AstraevalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
