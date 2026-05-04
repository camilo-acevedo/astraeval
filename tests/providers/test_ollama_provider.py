"""Tests for :mod:`astraeval.providers.ollama_provider`."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from astraeval.exceptions import ProviderError
from astraeval.providers.ollama_provider import OllamaProvider


def _dict_response(
    *,
    text: str = "hi",
    prompt_eval_count: int | None = 8,
    eval_count: int | None = 12,
    done_reason: str | None = "stop",
) -> dict[str, Any]:
    """Build the dict shape historically returned by the Ollama client.

    :param text: Value of ``message.content``.
    :type text: str
    :param prompt_eval_count: Tokens evaluated for the prompt.
    :type prompt_eval_count: int | None
    :param eval_count: Tokens generated as the response.
    :type eval_count: int | None
    :param done_reason: Reason reported by the daemon for stopping.
    :type done_reason: str | None
    :returns: Dict mirroring the JSON returned by Ollama.
    :rtype: dict[str, Any]
    """
    return {
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }


def test_dict_response_is_unwrapped() -> None:
    """The provider reads dict-shaped responses (legacy ``ollama`` releases)."""
    client = MagicMock()
    client.chat.return_value = _dict_response(text="hi", prompt_eval_count=10, eval_count=20)
    provider = OllamaProvider(client=client)

    response = provider.complete("ping", model="llama3.2")

    assert response.text == "hi"
    assert response.model == "llama3.2"
    assert response.provider == "ollama"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 20
    assert response.finish_reason == "stop"


def test_object_response_is_unwrapped() -> None:
    """The provider also reads typed-object responses (newer ``ollama`` releases)."""
    from types import SimpleNamespace

    client = MagicMock()
    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content="object-style"),
        done=True,
        done_reason="stop",
        prompt_eval_count=3,
        eval_count=5,
    )
    provider = OllamaProvider(client=client)

    response = provider.complete("p", model="m")

    assert response.text == "object-style"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 5


def test_params_become_options_payload() -> None:
    """Sampling kwargs are forwarded under the ``options`` key."""
    client = MagicMock()
    client.chat.return_value = _dict_response()
    provider = OllamaProvider(client=client)

    provider.complete("hi", model="m", temperature=0.4, num_predict=64)

    call = client.chat.call_args
    assert call.kwargs["model"] == "m"
    assert call.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert call.kwargs["options"] == {"temperature": 0.4, "num_predict": 64}


def test_no_params_sends_options_none() -> None:
    """When no params are provided ``options`` is ``None``."""
    client = MagicMock()
    client.chat.return_value = _dict_response()
    provider = OllamaProvider(client=client)

    provider.complete("hi", model="m")

    assert client.chat.call_args.kwargs["options"] is None


def test_non_string_content_raises_provider_error() -> None:
    """Unexpected ``message.content`` types are reported clearly."""
    client = MagicMock()
    client.chat.return_value = {"message": {"role": "assistant", "content": 123}}
    provider = OllamaProvider(client=client)

    with pytest.raises(ProviderError, match=r"message\.content"):
        provider.complete("p", model="m")


def test_upstream_failure_is_wrapped_as_provider_error() -> None:
    """SDK exceptions surface as :class:`ProviderError`."""
    client = MagicMock()
    client.chat.side_effect = RuntimeError("boom")
    provider = OllamaProvider(client=client)

    with pytest.raises(ProviderError, match="Ollama"):
        provider.complete("p", model="m")


def test_constructor_without_ollama_extra_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``client`` and without the SDK installed, the constructor raises."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "ollama":
            raise ImportError("no ollama")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="ollama"):
        OllamaProvider()
