"""Shared helpers for LLM-as-judge metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from astraeval.core.types import Response
from astraeval.exceptions import MetricError
from astraeval.providers.base import Provider

_TRUNCATED_FINISH_REASONS = frozenset({"max_tokens", "length"})


class LLMJudge:
    """Thin wrapper around a :class:`Provider` configured for repeated judge calls.

    Bundling the model and shared parameters with the provider keeps metric
    classes small: they only need to compose a prompt and call
    :meth:`ask`. The wrapper does not hold any state across calls.

    :param provider: Underlying provider that will execute judge calls.
    :type provider: Provider
    :param model: Model identifier forwarded to ``Provider.complete``.
    :type model: str
    :param params: Provider parameters (temperature, max_tokens, ...) used
        on every call. Defaults to ``{"temperature": 0.0}`` so judges are as
        deterministic as the provider allows.
    :type params: collections.abc.Mapping[str, Any] | None
    """

    def __init__(
        self,
        provider: Provider,
        *,
        model: str,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._params: Mapping[str, Any] = (
            dict(params) if params is not None else {"temperature": 0.0}
        )

    @property
    def provider(self) -> Provider:
        """Return the underlying provider.

        :returns: Provider executing judge calls.
        :rtype: Provider
        """
        return self._provider

    @property
    def model(self) -> str:
        """Return the model identifier used for judge calls.

        :returns: Model identifier.
        :rtype: str
        """
        return self._model

    def ask(self, prompt: str) -> Response:
        """Submit ``prompt`` to the configured provider and return the full response.

        Returning the full :class:`~astraeval.core.types.Response` (rather than
        just the text) lets callers and downstream tooling capture token
        counts, latency, and finish reason for the judge call as part of
        the run audit trail.

        :param prompt: Fully rendered prompt for the judge.
        :type prompt: str
        :returns: Structured response produced by the provider.
        :rtype: Response
        """
        return self._provider.complete(prompt, model=self._model, **self._params)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of LLM output, tolerating fences and prose.

    Real-world LLMs sometimes break the "JSON only" instruction. This helper
    is forgiving: it strips a surrounding triple-backtick fence, then tries
    a direct parse, and on failure falls back to extracting the first
    balanced ``{...}`` block and parsing that. Only a top-level JSON
    *object* (mapping) is accepted; arrays, strings, and scalars raise.

    :param text: Raw text returned by an LLM judge.
    :type text: str
    :returns: Parsed mapping representing the JSON object.
    :rtype: dict[str, Any]
    :raises astraeval.exceptions.MetricError: When no JSON object can be
        recovered from the text.
    """
    cleaned = _strip_code_fence(text).strip()

    direct = _try_parse_object(cleaned)
    if direct is not None:
        return direct

    extracted = _extract_first_object(cleaned)
    if extracted is not None:
        from_block = _try_parse_object(extracted)
        if from_block is not None:
            return from_block

    snippet = cleaned if len(cleaned) <= 200 else cleaned[:200] + "..."
    raise MetricError(f"Judge did not return a valid JSON object. First 200 chars: {snippet!r}")


def parse_judge_response(response: Response) -> dict[str, Any]:
    """Parse the JSON payload from a judge :class:`Response` with truncation context.

    Wraps :func:`parse_json_object` and, when the underlying provider
    truncated the output (``finish_reason`` of ``"max_tokens"`` or
    ``"length"``), rewrites the error to name the cause explicitly so the
    caller knows the fix is to raise the judge's token budget rather than
    debug a malformed prompt.

    :param response: Response returned by :meth:`LLMJudge.ask`.
    :type response: Response
    :returns: Parsed mapping representing the JSON object.
    :rtype: dict[str, Any]
    :raises astraeval.exceptions.MetricError: When the payload cannot be
        recovered. The message identifies truncation when applicable.
    """
    try:
        return parse_json_object(response.text)
    except MetricError as exc:
        if response.finish_reason in _TRUNCATED_FINISH_REASONS:
            raise MetricError(
                f"Judge response was truncated "
                f"(finish_reason={response.finish_reason!r}); "
                f"increase the judge's 'max_tokens' budget. "
                f"Underlying parse error: {exc}"
            ) from exc
        raise


def _try_parse_object(text: str) -> dict[str, Any] | None:
    """Return a parsed JSON object from ``text``, or ``None`` if not parseable.

    :param text: Candidate JSON text.
    :type text: str
    :returns: The parsed mapping, or ``None`` when the input is invalid JSON
        or parses to a non-object value.
    :rtype: dict[str, Any] | None
    """
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None
    return result


def _extract_first_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring of ``text``.

    Walks the string with brace-depth tracking that ignores braces inside
    JSON string literals (handling escape sequences). Returns ``None`` when
    no balanced object is found.

    :param text: Source text potentially containing prose around JSON.
    :type text: str
    :returns: The first balanced object substring, or ``None``.
    :rtype: str | None
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_code_fence(text: str) -> str:
    """Strip a single surrounding markdown code fence from ``text`` if present.

    :param text: Text potentially wrapped in a triple-backtick code fence,
        with or without a language tag (for example ``json``).
    :type text: str
    :returns: Text with the outermost fence removed.
    :rtype: str
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped
    body = stripped[first_newline + 1 :]
    trailing = body.rstrip()
    if trailing.endswith("```"):
        body = trailing[:-3].rstrip()
    return body
