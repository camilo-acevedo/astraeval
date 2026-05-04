"""Ollama provider for completions served by a local Ollama daemon."""

from __future__ import annotations

import time
from typing import Any

from llm_evals.core.types import Response
from llm_evals.exceptions import ProviderError
from llm_evals.providers.base import Provider


class OllamaProvider(Provider):
    """Provider backed by an `Ollama <https://ollama.com>`_ server.

    Ollama exposes locally hosted models behind an HTTP API. The upstream
    client is constructed lazily so the ``ollama`` package is only
    required when this provider is actually used.

    :param host: URL of the Ollama daemon. When ``None`` the SDK falls back
        to its own default (``http://localhost:11434``).
    :type host: str | None
    :param client: Pre-built upstream client used as-is. Primarily a hook
        for unit tests that supply a mock.
    :type client: Any
    :raises ImportError: When the optional ``ollama`` extra is not
        installed and ``client`` is not provided.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        client: Any = None,
    ) -> None:
        self.name = "ollama"
        if client is not None:
            self._client = client
        else:
            try:
                from ollama import Client
            except ImportError as exc:
                raise ImportError(
                    "OllamaProvider requires the 'ollama' extra. "
                    "Install with: pip install 'llm-evals[ollama]'"
                ) from exc
            self._client = Client(host=host) if host is not None else Client()

    def complete(self, prompt: str, *, model: str, **params: Any) -> Response:
        """Send ``prompt`` to the Ollama chat endpoint.

        Provider parameters are nested under the ``options`` field expected
        by the Ollama API. Pass ``temperature``, ``num_predict``, and
        similar fields directly as keyword arguments.

        :param prompt: User prompt sent as a single ``user`` message.
        :type prompt: str
        :param model: Ollama model tag such as ``"llama3.2"``.
        :type model: str
        :param params: Sampling and decoding options forwarded as the
            ``options`` payload.
        :returns: Structured response with the generated text and counts.
        :rtype: Response
        :raises llm_evals.exceptions.ProviderError: When the upstream call
            fails or the response shape is unexpected.
        """
        start = time.perf_counter()
        try:
            result = self._client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options=params or None,
            )
        except Exception as exc:
            raise ProviderError(f"Ollama call failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        message = _get(result, "message")
        text = _get(message, "content") or ""
        if not isinstance(text, str):
            raise ProviderError("Ollama response 'message.content' was not a string.")
        return Response(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=_get(result, "prompt_eval_count"),
            completion_tokens=_get(result, "eval_count"),
            finish_reason=_get(result, "done_reason"),
            latency_ms=elapsed_ms,
            raw={},
        )


def _get(obj: Any, key: str) -> Any:
    """Look up ``key`` on ``obj`` whether it is a mapping or an attribute holder.

    The ``ollama`` Python package returns a typed object on newer releases
    and a plain ``dict`` on older ones. This helper bridges the two so the
    provider works against both shapes.

    :param obj: Mapping or object to read from.
    :type obj: Any
    :param key: Attribute or mapping key.
    :type key: str
    :returns: The looked-up value, or ``None`` when missing.
    :rtype: Any
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
