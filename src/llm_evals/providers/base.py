"""Abstract :class:`Provider` interface and request-key utility."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from llm_evals.core.types import Response


class Provider(ABC):
    """Abstract base class for synchronous LLM completion providers.

    Concrete subclasses assign :attr:`name` in their constructor and implement
    :meth:`complete`. Providers are expected to be cheap to construct and may
    hold a long-lived HTTP client internally.

    :ivar name: Stable identifier for this provider, used in cache keys and
        run manifests.
    :vartype name: str
    """

    name: str

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        model: str,
        **params: Any,
    ) -> Response:
        """Send ``prompt`` to ``model`` and return a :class:`Response`.

        :param prompt: Text prompt sent to the model.
        :type prompt: str
        :param model: Model identifier accepted by the upstream provider.
        :type model: str
        :param params: Provider-specific keyword arguments such as
            ``temperature`` or ``max_tokens`` forwarded verbatim to the
            upstream API.
        :returns: Structured response containing the generated text and
            associated metadata.
        :rtype: Response
        :raises llm_evals.exceptions.ProviderError: When the upstream call
            fails or returns an unexpected payload shape.
        """


def request_key(
    *,
    provider: str,
    model: str,
    prompt: str,
    params: Mapping[str, Any],
) -> str:
    """Compute a stable SHA-256 key for a completion request.

    The resulting key is independent of dictionary insertion order and only
    changes when the prompt, model, provider, or any forwarded parameter
    changes. It is intended to serve as the primary key for the on-disk
    request cache.

    :param provider: Provider name as exposed via :attr:`Provider.name`.
    :type provider: str
    :param model: Model identifier passed to :meth:`Provider.complete`.
    :type model: str
    :param prompt: Prompt text submitted to the provider.
    :type prompt: str
    :param params: Provider parameters that affect the response.
    :type params: collections.abc.Mapping[str, Any]
    :returns: 64-character hexadecimal SHA-256 digest.
    :rtype: str
    """
    payload = {
        "provider": provider,
        "model": model,
        "prompt": prompt,
        "params": dict(params),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
