"""Smoke tests verifying that the package imports and exposes a version string."""

from __future__ import annotations

import llm_evals


def test_version_is_string() -> None:
    """Verify ``llm_evals.__version__`` resolves to a non-empty string."""
    assert isinstance(llm_evals.__version__, str)
    assert llm_evals.__version__ != ""
