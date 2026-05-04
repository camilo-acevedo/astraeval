"""Tests for :mod:`astraea.core.manifest`."""

from __future__ import annotations

import json

from astraea.core.manifest import (
    RunManifest,
    hash_dataset,
    hash_params,
    utc_now_iso,
)


def test_hash_dataset_is_order_dependent() -> None:
    """Sample order is part of the dataset hash."""
    a = hash_dataset([{"input": "Q1"}, {"input": "Q2"}])
    b = hash_dataset([{"input": "Q2"}, {"input": "Q1"}])

    assert a != b


def test_hash_dataset_is_deterministic_across_dict_orders() -> None:
    """Within a sample, key insertion order does not affect the hash."""
    a = hash_dataset([{"input": "Q", "expected": "A"}])
    b = hash_dataset([{"expected": "A", "input": "Q"}])

    assert a == b


def test_hash_params_is_order_independent() -> None:
    """Hashing a parameter dict is independent of insertion order."""
    a = hash_params({"temperature": 0.0, "max_tokens": 16})
    b = hash_params({"max_tokens": 16, "temperature": 0.0})

    assert a == b


def test_hash_params_changes_with_value() -> None:
    """Changing any parameter value yields a different hash."""
    base = hash_params({"temperature": 0.0})
    changed = hash_params({"temperature": 0.7})

    assert base != changed


def test_utc_now_iso_returns_iso8601() -> None:
    """``utc_now_iso`` produces a parsable ISO-8601 string."""
    timestamp = utc_now_iso()
    assert "T" in timestamp
    assert timestamp.endswith("+00:00")


def test_run_manifest_to_json_round_trip() -> None:
    """The JSON document produced by ``to_json`` is parseable and complete."""
    manifest = RunManifest(
        run_id="abc123",
        provider="fake",
        model="m",
        metric_names=("exact_match",),
        sample_count=2,
        dataset_hash="d",
        params_hash="p",
        started_at="2026-05-04T00:00:00+00:00",
        finished_at="2026-05-04T00:00:01+00:00",
        summary={"exact_match": 0.5},
    )

    payload = json.loads(manifest.to_json())

    assert payload["run_id"] == "abc123"
    assert payload["metric_names"] == ["exact_match"]
    assert payload["summary"] == {"exact_match": 0.5}
