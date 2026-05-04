"""Tests for :mod:`astraea.providers.openai_provider`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from astraea.exceptions import ProviderError
from astraea.providers.openai_provider import OpenAIProvider


def _build_chat_completion(
    *,
    text: str = "hi there",
    finish_reason: str = "stop",
    prompt_tokens: int = 4,
    completion_tokens: int = 6,
    response_id: str = "chatcmpl_xyz",
) -> SimpleNamespace:
    """Construct a stand-in for an OpenAI ChatCompletion response.

    :param text: Text returned in ``choices[0].message.content``.
    :type text: str
    :param finish_reason: Value reported in ``choices[0].finish_reason``.
    :type finish_reason: str
    :param prompt_tokens: Value reported in ``usage.prompt_tokens``.
    :type prompt_tokens: int
    :param completion_tokens: Value reported in ``usage.completion_tokens``.
    :type completion_tokens: int
    :param response_id: Value reported in ``id``.
    :type response_id: str
    :returns: Object exposing the attributes the provider reads.
    :rtype: types.SimpleNamespace
    """
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(role="assistant", content=text),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def test_complete_returns_normalized_response() -> None:
    """The provider unwraps the SDK payload into a :class:`Response`."""
    client = MagicMock()
    client.chat.completions.create.return_value = _build_chat_completion(text="hi")
    provider = OpenAIProvider(client=client)

    response = provider.complete("ping", model="gpt-test")

    assert response.text == "hi"
    assert response.model == "gpt-test"
    assert response.provider == "openai"
    assert response.prompt_tokens == 4
    assert response.completion_tokens == 6
    assert response.finish_reason == "stop"
    assert response.raw == {"id": "chatcmpl_xyz"}


def test_complete_forwards_params() -> None:
    """Caller params reach ``chat.completions.create``."""
    client = MagicMock()
    client.chat.completions.create.return_value = _build_chat_completion()
    provider = OpenAIProvider(client=client)

    provider.complete("hi", model="m", temperature=0.5, max_tokens=100)

    call = client.chat.completions.create.call_args
    assert call.kwargs["model"] == "m"
    assert call.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert call.kwargs["temperature"] == 0.5
    assert call.kwargs["max_tokens"] == 100


def test_empty_choices_raises_provider_error() -> None:
    """A response with zero ``choices`` is reported as a provider error."""
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(id="x", choices=[], usage=None)
    provider = OpenAIProvider(client=client)

    with pytest.raises(ProviderError, match="no choices"):
        provider.complete("p", model="m")


def test_null_message_content_becomes_empty_string() -> None:
    """When ``content`` is ``None`` (tool-call only) the text falls back to ``""``."""
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        id="x",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(role="assistant", content=None),
            )
        ],
        usage=None,
    )
    provider = OpenAIProvider(client=client)

    response = provider.complete("p", model="m")

    assert response.text == ""
    assert response.finish_reason == "tool_calls"


def test_upstream_failure_is_wrapped_as_provider_error() -> None:
    """SDK exceptions surface as :class:`ProviderError`."""
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    provider = OpenAIProvider(client=client)

    with pytest.raises(ProviderError, match="OpenAI"):
        provider.complete("p", model="m")


def test_constructor_without_openai_extra_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``client`` and without the SDK installed, the constructor raises."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="openai"):
        OpenAIProvider()
