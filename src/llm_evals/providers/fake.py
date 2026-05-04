"""In-memory provider used by tests and examples."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from llm_evals.core.types import Response
from llm_evals.providers.base import Provider


class FakeProvider(Provider):
    """Deterministic provider that returns canned responses.

    Construct with either a static iterable of response strings (consumed in
    order on successive calls) or a callable that receives
    ``(prompt, model, params)`` and returns the response text. Useful for unit
    tests and for demonstrating the harness without an API key.

    :param responses: Static iterable of response texts, consumed in order.
    :type responses: collections.abc.Iterable[str] | None
    :param handler: Callable producing one response text per invocation.
    :type handler: collections.abc.Callable[..., str] | None
    :raises ValueError: When neither ``responses`` nor ``handler`` is provided,
        or when both are provided.
    """

    def __init__(
        self,
        responses: Iterable[str] | None = None,
        *,
        handler: Callable[[str, str, Mapping[str, Any]], str] | None = None,
    ) -> None:
        if (responses is None) == (handler is None):
            raise ValueError("FakeProvider requires exactly one of 'responses' or 'handler'.")
        self.name = "fake"
        self._responses: list[str] | None = list(responses) if responses is not None else None
        self._handler = handler
        self._index = 0

    def complete(self, prompt: str, *, model: str, **params: Any) -> Response:
        """Return the next canned response.

        :param prompt: Prompt forwarded to the configured handler, if any.
        :type prompt: str
        :param model: Model identifier echoed back on the response object.
        :type model: str
        :param params: Forwarded to the configured handler, if any.
        :returns: Canned response paired with the requested model.
        :rtype: Response
        :raises RuntimeError: When the static response list is exhausted.
        """
        start = time.perf_counter()
        if self._handler is not None:
            text = self._handler(prompt, model, params)
        else:
            assert self._responses is not None
            if self._index >= len(self._responses):
                raise RuntimeError("FakeProvider exhausted its canned responses.")
            text = self._responses[self._index]
            self._index += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return Response(
            text=text,
            model=model,
            provider=self.name,
            latency_ms=elapsed_ms,
        )
