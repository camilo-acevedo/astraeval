"""SQLite-backed cache for completion responses."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from astraeval.core.types import Response
from astraeval.exceptions import CacheError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_responses_provider ON responses(provider);
CREATE INDEX IF NOT EXISTS idx_responses_created_at ON responses(created_at);
"""


class Cache:
    """Persistent SQLite cache of :class:`Response` objects keyed by request hash.

    The cache is safe to use from multiple threads in the same process; each
    operation opens, commits, and closes its own connection. The schema is
    created on first use.

    :param path: Filesystem path to the SQLite database file. The parent
        directory is created automatically when missing.
    :type path: str | pathlib.Path
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def path(self) -> Path:
        """Filesystem path to the underlying SQLite database.

        :returns: Resolved path of the database file.
        :rtype: pathlib.Path
        """
        return self._path

    def get(self, key: str) -> Response | None:
        """Look up a cached response by its request key.

        :param key: SHA-256 request hash as produced by
            :func:`astraeval.providers.base.hash_request`.
        :type key: str
        :returns: The cached :class:`Response`, or ``None`` on a cache miss.
        :rtype: Response | None
        :raises astraeval.exceptions.CacheError: When the stored payload is
            malformed and cannot be deserialized.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM responses WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
            return Response(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CacheError(f"Cache entry for key {key!r} is corrupted") from exc

    def set(self, key: str, response: Response) -> None:
        """Insert or replace a cached response.

        :param key: SHA-256 request key.
        :type key: str
        :param response: Response object to persist.
        :type response: Response
        """
        payload = json.dumps(asdict(response))
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, provider, model, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (key, response.provider, response.model, payload, time.time()),
            )

    def clear(self) -> None:
        """Remove every entry in the cache."""
        with self._connect() as conn:
            conn.execute("DELETE FROM responses")

    def __len__(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(row[0])

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM responses WHERE key = ? LIMIT 1",
                (key,),
            ).fetchone()
        return row is not None

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
