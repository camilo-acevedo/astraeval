"""LLM provider adapters for Anthropic, OpenAI, and Ollama."""

from astraea.providers.anthropic_provider import AnthropicProvider
from astraea.providers.base import Provider, request_key
from astraea.providers.cached import CachedProvider
from astraea.providers.fake import FakeProvider
from astraea.providers.ollama_provider import OllamaProvider
from astraea.providers.openai_provider import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "CachedProvider",
    "FakeProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "Provider",
    "request_key",
]
