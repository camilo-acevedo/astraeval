"""OpenAI provider built on the official ``openai`` SDK."""

from __future__ import annotations

import time
from typing import Any

from astraeval.core.types import Response
from astraeval.exceptions import ProviderError
from astraeval.providers.base import Provider


class OpenAIProvider(Provider):
    """Provider that issues completions through the OpenAI Chat Completions API.

    The upstream client is constructed lazily so the ``openai`` package is
    only required when this provider is actually used. Pass a custom
    ``client`` for tests or to share a configured instance.

    :param api_key: API key passed to :class:`openai.OpenAI`. Ignored when
        ``client`` is provided. When ``None`` the SDK reads the
        ``OPENAI_API_KEY`` environment variable.
    :type api_key: str | None
    :param base_url: Optional base URL override for OpenAI-compatible
        endpoints (Azure OpenAI, vLLM, llama.cpp server, etc.).
    :type base_url: str | None
    :param client: Pre-built upstream client used as-is. Primarily a hook
        for unit tests that supply a mock.
    :type client: Any
    :raises ImportError: When the optional ``openai`` extra is not installed
        and ``client`` is not provided.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
    ) -> None:
        self.name = "openai"
        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "OpenAIProvider requires the 'openai' extra. "
                    "Install with: pip install 'astraeval[openai]'"
                ) from exc
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str, *, model: str, **params: Any) -> Response:
        """Send ``prompt`` to the OpenAI Chat Completions API.

        :param prompt: User prompt sent as a single ``user`` message.
        :type prompt: str
        :param model: OpenAI model identifier such as ``"gpt-4o-mini"``.
        :type model: str
        :param params: Additional parameters forwarded to
            ``chat.completions.create`` (``temperature``, ``max_tokens``,
            ``response_format``, etc.).
        :returns: Structured response with the generated text and usage.
        :rtype: Response
        :raises astraeval.exceptions.ProviderError: When the upstream call
            fails or the response payload is empty.
        """
        start = time.perf_counter()
        try:
            result = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **params,
            )
        except Exception as exc:
            raise ProviderError(f"OpenAI call failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        choices = getattr(result, "choices", None) or []
        if not choices:
            raise ProviderError("OpenAI response contained no choices.")
        choice = choices[0]
        message = getattr(choice, "message", None)
        text = getattr(message, "content", None) or ""
        usage = getattr(result, "usage", None)
        return Response(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            finish_reason=getattr(choice, "finish_reason", None),
            latency_ms=elapsed_ms,
            raw={"id": getattr(result, "id", None)} if hasattr(result, "id") else {},
        )
