"""Command-line entry point for the ``llm-evals`` package."""

from __future__ import annotations

import sys

from llm_evals import __version__


def main(argv: list[str] | None = None) -> int:
    """Run the ``llm-evals`` command-line interface.

    :param argv: Command-line arguments to parse. When ``None``, the function
        falls back to :data:`sys.argv` excluding the program name.
    :type argv: list[str] | None
    :returns: Process exit code, where ``0`` indicates success.
    :rtype: int
    """
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in {"-V", "--version"}:
        print(f"llm-evals {__version__}")
        return 0
    print(f"llm-evals {__version__} - CLI not yet implemented.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
