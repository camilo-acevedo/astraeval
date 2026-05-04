"""Tests for :mod:`astraea.core.cache`."""

from __future__ import annotations

from pathlib import Path

import pytest

from astraea.core.cache import Cache
from astraea.core.types import Response
from astraea.exceptions import CacheError


def _make_response(text: str = "hi") -> Response:
    """Construct a minimal :class:`Response` for cache tests.

    :param text: Text to embed in the response.
    :type text: str
    :returns: Response object with deterministic fields.
    :rtype: Response
    """
    return Response(text=text, model="m", provider="fake", latency_ms=1.0)


def test_set_and_get_round_trip(tmp_path: Path) -> None:
    """A response written to the cache is retrievable by key with all fields preserved."""
    cache = Cache(tmp_path / "calls.sqlite")
    response = _make_response("hello")

    cache.set("k1", response)
    fetched = cache.get("k1")

    assert fetched == response


def test_get_returns_none_on_miss(tmp_path: Path) -> None:
    """An unknown key yields ``None`` rather than raising."""
    cache = Cache(tmp_path / "calls.sqlite")
    assert cache.get("missing") is None


def test_set_replaces_existing_entry(tmp_path: Path) -> None:
    """A second ``set`` for the same key overwrites the previous value."""
    cache = Cache(tmp_path / "calls.sqlite")

    cache.set("k", _make_response("old"))
    cache.set("k", _make_response("new"))

    fetched = cache.get("k")
    assert fetched is not None
    assert fetched.text == "new"
    assert len(cache) == 1


def test_clear_removes_every_entry(tmp_path: Path) -> None:
    """``clear`` empties the cache."""
    cache = Cache(tmp_path / "calls.sqlite")
    cache.set("a", _make_response())
    cache.set("b", _make_response())
    assert len(cache) == 2

    cache.clear()
    assert len(cache) == 0


def test_contains_operator(tmp_path: Path) -> None:
    """The ``in`` operator returns ``True`` only for keys that were set."""
    cache = Cache(tmp_path / "calls.sqlite")
    cache.set("present", _make_response())

    assert "present" in cache
    assert "absent" not in cache
    assert 123 not in cache


def test_corrupted_payload_raises_cache_error(tmp_path: Path) -> None:
    """Manually injected garbage payload surfaces as :class:`CacheError`."""
    import sqlite3

    cache = Cache(tmp_path / "calls.sqlite")
    cache.set("k", _make_response())

    conn = sqlite3.connect(cache.path)
    try:
        conn.execute("UPDATE responses SET payload = ? WHERE key = ?", ("not json", "k"))
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(CacheError):
        cache.get("k")


def test_creates_parent_directory(tmp_path: Path) -> None:
    """The cache eagerly creates missing parent directories."""
    nested = tmp_path / "deeply" / "nested" / "calls.sqlite"
    cache = Cache(nested)
    cache.set("k", _make_response())

    assert nested.exists()
    assert cache.get("k") is not None
