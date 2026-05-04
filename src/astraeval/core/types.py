"""Core dataclasses shared across providers, metrics, and runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Response:
    """Outcome of a single completion request to an LLM provider.

    :ivar text: Generated text returned by the model.
    :vartype text: str
    :ivar model: Model identifier passed to the upstream provider.
    :vartype model: str
    :ivar provider: Provider name that produced this response, e.g.
        ``"anthropic"`` or ``"openai"``.
    :vartype provider: str
    :ivar prompt_tokens: Tokens consumed by the prompt as reported by the
        provider, or ``None`` when not available.
    :vartype prompt_tokens: int | None
    :ivar completion_tokens: Tokens consumed by the generated text, or
        ``None`` when the provider does not report it.
    :vartype completion_tokens: int | None
    :ivar finish_reason: Provider-reported reason for the generation stopping.
    :vartype finish_reason: str | None
    :ivar latency_ms: Wall-clock latency of the call in milliseconds.
    :vartype latency_ms: float | None
    :ivar raw: Provider-specific payload retained for debugging or auditing.
    :vartype raw: dict[str, Any]
    """

    text: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
