"""Shared helpers for LLM-as-judge metrics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from llm_evals.exceptions import MetricError
from llm_evals.providers.base import Provider


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

    def ask(self, prompt: str) -> str:
        """Submit ``prompt`` to the configured provider and return raw text.

        :param prompt: Fully rendered prompt for the judge.
        :type prompt: str
        :returns: Generated text returned by the provider.
        :rtype: str
        """
        response = self._provider.complete(prompt, model=self._model, **self._params)
        return response.text


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object out of LLM output, tolerating markdown code fences.

    LLMs frequently wrap JSON inside triple-backtick fences even when asked
    not to. This helper strips a single surrounding fence (with or without a
    language tag), then defers to :func:`json.loads`.

    :param text: Raw text returned by an LLM judge.
    :type text: str
    :returns: Parsed mapping representing the JSON object.
    :rtype: dict[str, Any]
    :raises llm_evals.exceptions.MetricError: When the text does not parse
        as JSON or parses to a non-object value.
    """
    cleaned = _strip_code_fence(text).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise MetricError(f"Judge did not return valid JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise MetricError(f"Judge returned non-object JSON of type {type(result).__name__}.")
    return result


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
