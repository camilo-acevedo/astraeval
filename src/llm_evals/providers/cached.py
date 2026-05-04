"""Provider decorator that adds transparent caching of completion responses."""

from __future__ import annotations

from typing import Any

from llm_evals.core.cache import Cache
from llm_evals.core.types import Response
from llm_evals.providers.base import Provider, request_key


class CachedProvider(Provider):
    """Wrap a :class:`Provider` so identical requests are served from a cache.

    The wrapper preserves the inner provider's :attr:`name` so cache keys,
    logs, and run manifests reflect the underlying upstream rather than the
    decorator. A cache miss delegates to the inner provider and stores the
    result before returning it.

    :param inner: Underlying provider that performs the actual upstream call
        on a cache miss.
    :type inner: Provider
    :param cache: Storage backend used to persist responses.
    :type cache: Cache
    """

    def __init__(self, inner: Provider, cache: Cache) -> None:
        self.name = inner.name
        self._inner = inner
        self._cache = cache

    @property
    def inner(self) -> Provider:
        """Return the wrapped provider.

        :returns: The provider used to satisfy cache misses.
        :rtype: Provider
        """
        return self._inner

    def complete(self, prompt: str, *, model: str, **params: Any) -> Response:
        """Return a cached response when available, otherwise call the inner provider.

        :param prompt: Text prompt sent to the model.
        :type prompt: str
        :param model: Model identifier.
        :type model: str
        :param params: Provider parameters forwarded to the inner provider on
            a cache miss and used when computing the cache key.
        :returns: Either the cached response or a freshly produced one.
        :rtype: Response
        """
        key = request_key(
            provider=self._inner.name,
            model=model,
            prompt=prompt,
            params=params,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        response = self._inner.complete(prompt, model=model, **params)
        self._cache.set(key, response)
        return response
