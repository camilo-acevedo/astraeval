"""Smoke tests verifying that the package imports and exposes a version string."""

from __future__ import annotations

import astraeval


def test_version_is_string() -> None:
    """Verify ``astraeval.__version__`` resolves to a non-empty string."""
    assert isinstance(astraeval.__version__, str)
    assert astraeval.__version__ != ""
