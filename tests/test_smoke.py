"""Smoke tests verifying that the package imports and exposes a version string."""

from __future__ import annotations

import pytest

import llm_evals
from llm_evals.cli import main


def test_version_is_string() -> None:
    """Verify ``llm_evals.__version__`` resolves to a non-empty string."""
    assert isinstance(llm_evals.__version__, str)
    assert llm_evals.__version__ != ""


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the CLI prints the version when invoked with ``--version``.

    :param capsys: Pytest fixture capturing standard output and standard error.
    :type capsys: pytest.CaptureFixture[str]
    """
    exit_code = main(["--version"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "llm-evals" in captured.out


def test_cli_no_args(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the CLI emits a placeholder message when invoked without arguments.

    :param capsys: Pytest fixture capturing standard output and standard error.
    :type capsys: pytest.CaptureFixture[str]
    """
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "llm-evals" in captured.out
