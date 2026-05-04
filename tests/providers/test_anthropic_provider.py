"""Tests for :mod:`astraea.providers.anthropic_provider`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from astraea.exceptions import ProviderError
from astraea.providers.anthropic_provider import AnthropicProvider


def _build_messages_response(
    *,
    text: str = "hello",
    input_tokens: int = 5,
    output_tokens: int = 3,
    stop_reason: str = "end_turn",
    response_id: str = "msg_abc",
) -> SimpleNamespace:
    """Construct a minimal stand-in for an Anthropic Message response.

    :param text: Text returned by the assistant in a single text block.
    :type text: str
    :param input_tokens: Value reported in ``usage.input_tokens``.
    :type input_tokens: int
    :param output_tokens: Value reported in ``usage.output_tokens``.
    :type output_tokens: int
    :param stop_reason: Value reported in ``stop_reason``.
    :type stop_reason: str
    :param response_id: Value reported in ``id``.
    :type response_id: str
    :returns: Object exposing the attributes the provider reads.
    :rtype: types.SimpleNamespace
    """
    return SimpleNamespace(
        id=response_id,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


def test_complete_returns_normalized_response() -> None:
    """The provider unwraps the SDK payload into a :class:`Response`."""
    client = MagicMock()
    client.messages.create.return_value = _build_messages_response(text="ok", input_tokens=7)
    provider = AnthropicProvider(client=client)

    response = provider.complete("ping", model="claude-test")

    assert response.text == "ok"
    assert response.model == "claude-test"
    assert response.provider == "anthropic"
    assert response.prompt_tokens == 7
    assert response.completion_tokens == 3
    assert response.finish_reason == "end_turn"
    assert response.raw == {"id": "msg_abc"}
    assert response.latency_ms is not None


def test_complete_forwards_params_and_messages() -> None:
    """Caller params reach ``messages.create`` and the prompt is wrapped as a user message."""
    client = MagicMock()
    client.messages.create.return_value = _build_messages_response()
    provider = AnthropicProvider(client=client, default_max_tokens=512)

    provider.complete("hi", model="m", temperature=0.2, system="be terse")

    call = client.messages.create.call_args
    assert call.kwargs["model"] == "m"
    assert call.kwargs["max_tokens"] == 512
    assert call.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert call.kwargs["temperature"] == 0.2
    assert call.kwargs["system"] == "be terse"


def test_caller_max_tokens_overrides_default() -> None:
    """Explicit ``max_tokens`` in the call wins over ``default_max_tokens``."""
    client = MagicMock()
    client.messages.create.return_value = _build_messages_response()
    provider = AnthropicProvider(client=client, default_max_tokens=512)

    provider.complete("hi", model="m", max_tokens=64)

    assert client.messages.create.call_args.kwargs["max_tokens"] == 64


def test_multiple_text_blocks_are_concatenated() -> None:
    """Anthropic returns content as a list; text blocks are joined in order."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        id="x",
        content=[
            SimpleNamespace(type="text", text="part1"),
            SimpleNamespace(type="tool_use"),
            SimpleNamespace(type="text", text="-part2"),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    provider = AnthropicProvider(client=client)

    response = provider.complete("p", model="m")

    assert response.text == "part1-part2"


def test_upstream_failure_is_wrapped_as_provider_error() -> None:
    """Any exception raised by the SDK becomes :class:`ProviderError`."""
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    provider = AnthropicProvider(client=client)

    with pytest.raises(ProviderError, match="Anthropic"):
        provider.complete("p", model="m")


def test_response_handles_missing_usage_gracefully() -> None:
    """A response without a ``usage`` block produces ``None`` token counts."""
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        id="x",
        content=[SimpleNamespace(type="text", text="ok")],
        usage=None,
        stop_reason=None,
    )
    provider = AnthropicProvider(client=client)

    response = provider.complete("p", model="m")

    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.finish_reason is None


def test_constructor_without_anthropic_extra_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``client`` and without the SDK installed, the constructor raises."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="anthropic"):
        AnthropicProvider()
