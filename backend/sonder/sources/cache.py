import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from sonder.config import get_settings

DEFAULT_TTL_SECONDS = 7 * 24 * 3600


def make_key(service: str, path: str, params: dict[str, Any] | None = None) -> str:
    query = urlencode(sorted((params or {}).items()))
    return f"{service}:{path}?{query}"


class SqliteCache:
    def __init__(
        self,
        path: Path | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path if path is not None else get_settings().cache_db
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS http_cache "
                    "(key TEXT PRIMARY KEY, value TEXT NOT NULL, fetched_at REAL NOT NULL)"
                )
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, key: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT value, fetched_at FROM http_cache WHERE key = ?", (key,)
            ).fetchone()
        finally:
            conn.close()
        if row is None or self._clock() - row[1] > self.ttl_seconds:
            return None
        value: dict[str, Any] = json.loads(row[0])
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO http_cache (key, value, fetched_at) VALUES (?, ?, ?)",
                    (key, json.dumps(value), self._clock()),
                )
        finally:
            conn.close()
