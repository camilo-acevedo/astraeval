"""Run manifest for reproducibility and auditing of evaluation runs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable record describing a single :class:`EvalRun` execution.

    The manifest captures every input that influenced the run so the result
    can be audited and reproduced. Hashes cover dataset content and the
    forwarded provider parameters so that semantically identical re-runs
    share an identifier.

    :ivar run_id: Stable identifier for the run, derived from its inputs.
    :vartype run_id: str
    :ivar provider: Provider name as exposed by ``Provider.name``.
    :vartype provider: str
    :ivar model: Model identifier passed to ``Provider.complete``.
    :vartype model: str
    :ivar metric_names: Names of every metric scored during the run, in the
        order they were configured.
    :vartype metric_names: tuple[str, ...]
    :ivar sample_count: Number of samples processed.
    :vartype sample_count: int
    :ivar dataset_hash: SHA-256 over the canonical JSON of every sample.
    :vartype dataset_hash: str
    :ivar params_hash: SHA-256 over the canonical JSON of the provider
        params dictionary forwarded on each call.
    :vartype params_hash: str
    :ivar started_at: ISO-8601 timestamp marking the start of the run.
    :vartype started_at: str
    :ivar finished_at: ISO-8601 timestamp marking completion.
    :vartype finished_at: str
    :ivar summary: Mapping from metric name to aggregate (mean) score.
    :vartype summary: collections.abc.Mapping[str, float]
    """

    run_id: str
    provider: str
    model: str
    metric_names: tuple[str, ...]
    sample_count: int
    dataset_hash: str
    params_hash: str
    started_at: str
    finished_at: str
    summary: Mapping[str, float] = field(default_factory=dict)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the manifest as a JSON document.

        :param indent: Number of spaces to indent nested elements. ``None``
            yields the most compact representation.
        :type indent: int | None
        :returns: JSON document representing the manifest.
        :rtype: str
        """
        payload = asdict(self)
        payload["metric_names"] = list(self.metric_names)
        return json.dumps(payload, indent=indent, sort_keys=True)


def _canonical_json(value: Any) -> str:
    """Render ``value`` as a deterministic JSON document.

    :param value: Any JSON-serializable value.
    :type value: Any
    :returns: Canonical JSON string with sorted keys.
    :rtype: str
    """
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def hash_dataset(samples: Sequence[Mapping[str, Any]]) -> str:
    """Compute a deterministic SHA-256 over a sequence of sample payloads.

    :param samples: Sequence of dict-like sample representations.
    :type samples: collections.abc.Sequence[collections.abc.Mapping[str, Any]]
    :returns: 64-character hexadecimal SHA-256 digest.
    :rtype: str
    """
    serialized = _canonical_json([dict(s) for s in samples])
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_params(params: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 over a parameter dictionary.

    :param params: Provider parameters forwarded on every call.
    :type params: collections.abc.Mapping[str, Any]
    :returns: 64-character hexadecimal SHA-256 digest.
    :rtype: str
    """
    return hashlib.sha256(_canonical_json(dict(params)).encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    :returns: Current UTC timestamp formatted with second precision.
    :rtype: str
    """
    return datetime.now(UTC).isoformat(timespec="seconds")
