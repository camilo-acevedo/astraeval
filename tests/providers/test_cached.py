"""Tests for :mod:`llm_evals.providers.cached`."""

from __future__ import annotations

from pathlib import Path

from llm_evals.core.cache import Cache
from llm_evals.providers.cached import CachedProvider
from llm_evals.providers.fake import FakeProvider


def test_cache_miss_then_hit_calls_inner_only_once(tmp_path: Path) -> None:
    """The second call for the same prompt is served from the cache."""
    inner = FakeProvider(["first", "second"])
    cache = Cache(tmp_path / "calls.sqlite")
    cached = CachedProvider(inner, cache)

    a = cached.complete("p", model="m")
    b = cached.complete("p", model="m")

    assert a.text == "first"
    assert b.text == "first"
    assert len(cache) == 1


def test_different_prompts_use_different_cache_keys(tmp_path: Path) -> None:
    """Distinct prompts both reach the inner provider and produce different keys."""
    inner = FakeProvider(["one", "two"])
    cache = Cache(tmp_path / "calls.sqlite")
    cached = CachedProvider(inner, cache)

    a = cached.complete("first", model="m")
    b = cached.complete("second", model="m")

    assert a.text == "one"
    assert b.text == "two"
    assert len(cache) == 2


def test_different_params_bypass_cache(tmp_path: Path) -> None:
    """Identical prompt with different params still routes to the inner provider."""
    inner = FakeProvider(["cold", "warm"])
    cache = Cache(tmp_path / "calls.sqlite")
    cached = CachedProvider(inner, cache)

    a = cached.complete("p", model="m", temperature=0.0)
    b = cached.complete("p", model="m", temperature=1.0)

    assert a.text == "cold"
    assert b.text == "warm"
    assert len(cache) == 2


def test_name_reflects_inner_provider(tmp_path: Path) -> None:
    """The wrapper exposes the wrapped provider's name unchanged."""
    inner = FakeProvider(["x"])
    cached = CachedProvider(inner, Cache(tmp_path / "c.sqlite"))

    assert cached.name == inner.name == "fake"


def test_cache_persists_across_provider_instances(tmp_path: Path) -> None:
    """Recreating the wrapper against the same cache reuses prior responses."""
    cache_path = tmp_path / "calls.sqlite"

    cached_one = CachedProvider(FakeProvider(["seeded"]), Cache(cache_path))
    cached_one.complete("p", model="m")

    cached_two = CachedProvider(FakeProvider([]), Cache(cache_path))
    response = cached_two.complete("p", model="m")

    assert response.text == "seeded"
