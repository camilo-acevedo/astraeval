"""Definition of the :class:`Sample` evaluation primitive."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Sample:
    """A single evaluation example fed to the model under test.

    :ivar input: Prompt or question presented to the model.
    :vartype input: str
    :ivar expected: Optional reference answer used by metrics that require
        ground truth.
    :vartype expected: str | None
    :ivar context: Tuple of context passages, used by RAG-style metrics.
    :vartype context: tuple[str, ...]
    :ivar metadata: Arbitrary auxiliary fields attached to the sample.
    :vartype metadata: collections.abc.Mapping[str, Any]
    """

    input: str
    expected: str | None = None
    context: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
