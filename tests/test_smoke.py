"""Smoke tests verifying that the package imports and exposes a version string."""

from __future__ import annotations

import astraea


def test_version_is_string() -> None:
    """Verify ``astraea.__version__`` resolves to a non-empty string."""
    assert isinstance(astraea.__version__, str)
    assert astraea.__version__ != ""
