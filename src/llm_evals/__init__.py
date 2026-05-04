"""Reproducible LLM evaluation harness with run manifests, prompt hashing, and SQLite caching."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-evals")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
