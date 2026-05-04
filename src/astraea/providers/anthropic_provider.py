"""Anthropic provider built on the official ``anthropic`` SDK."""

from __future__ import annotations

import time
from typing import Any

from astraea.core.types import Response
from astraea.exceptions import ProviderError
from astraea.providers.base import Provider


class AnthropicProvider(Provider):
    """Provider that issues completions through the Anthropic Messages API.

    The upstream client is constructed lazily so the ``anthropic`` package
    is only required when this provider is actually used. Pass a custom
    ``client`` for tests or to share a configured instance.

    :param api_key: API key passed to :class:`anthropic.Anthropic`. Ignored
        when ``client`` is provided. When ``None`` the SDK reads the
        ``ANTHROPIC_API_KEY`` environment variable.
    :type api_key: str | None
    :param client: Pre-built upstream client used as-is. Primarily a hook
        for unit tests that supply a mock.
    :type client: Any
    :param default_max_tokens: Value of ``max_tokens`` forwarded when the
        caller does not provide one. The Anthropic API requires this field.
    :type default_max_tokens: int
    :raises ImportError: When the optional ``anthropic`` extra is not
        installed and ``client`` is not provided.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any = None,
        default_max_tokens: int = 1024,
    ) -> None:
        self.name = "anthropic"
        self._default_max_tokens = default_max_tokens
        if client is not None:
            self._client = client
        else:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise ImportError(
                    "AnthropicProvider requires the 'anthropic' extra. "
                    "Install with: pip install 'astraea[anthropic]'"
                ) from exc
            self._client = Anthropic(api_key=api_key)

    def complete(self, prompt: str, *, model: str, **params: Any) -> Response:
        """Send ``prompt`` to the Anthropic Messages API.

        :param prompt: User prompt sent as a single ``user`` message.
        :type prompt: str
        :param model: Anthropic model identifier such as
            ``"claude-opus-4-7"``.
        :type model: str
        :param params: Additional parameters forwarded to ``messages.create``
            (``temperature``, ``max_tokens``, ``system``, etc.).
        :returns: Structured response with the generated text and usage.
        :rtype: Response
        :raises astraea.exceptions.ProviderError: When the upstream call
            fails or the response payload cannot be interpreted.
        """
        max_tokens = int(params.pop("max_tokens", self._default_max_tokens))
        start = time.perf_counter()
        try:
            result = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **params,
            )
        except Exception as exc:
            raise ProviderError(f"Anthropic call failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        text = _extract_text(result)
        usage = getattr(result, "usage", None)
        return Response(
            text=text,
            model=model,
            provider=self.name,
            prompt_tokens=getattr(usage, "input_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "output_tokens", None) if usage else None,
            finish_reason=getattr(result, "stop_reason", None),
            latency_ms=elapsed_ms,
            raw={"id": getattr(result, "id", None)} if hasattr(result, "id") else {},
        )


def _extract_text(result: Any) -> str:
    """Pull the assistant text out of an Anthropic ``Message`` response.

    The Anthropic API returns ``content`` as a list of typed blocks; only
    text blocks are joined here. Non-text blocks (such as tool use) are
    skipped.

    :param result: Object returned by ``client.messages.create``.
    :type result: Any
    :returns: Concatenated text from every ``text`` content block.
    :rtype: str
    """
    blocks = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)
