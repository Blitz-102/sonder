import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sonder.config import get_settings
from sonder.sources.cache import SqliteCache, make_key
from sonder.sources.ratelimit import LASTFM_LIMITER, RateLimiter

BASE_URL = "https://ws.audioscrobbler.com/2.0/"
USER_AGENT = "sonder/0.1"
NOT_FOUND_CODE = 6
SHORT_RETRY_CODES = frozenset({8, 11, 16})
SHORT_RETRY_SECONDS = 5.0
RATE_LIMIT_CODE = 29
RATE_LIMIT_WAIT_SECONDS = 30.0
MAX_RETRIES = 3


class LastfmError(Exception):
    def __init__(self, code: int | None, message: str) -> None:
        super().__init__(f"Last.fm error {code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LastfmTrack:
    name: str
    artist: str
    match: float
    mbid: str | None


@dataclass(frozen=True)
class LastfmArtist:
    name: str
    match: float
    mbid: str | None


@dataclass(frozen=True)
class LastfmTag:
    name: str
    count: int


def _as_list(x: Any) -> list[Any]:
    """Last.fm returns a list, a single object, or nothing for the same field."""
    if x is None or x == "":
        return []
    if isinstance(x, list):
        return x
    return [x]


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _artist_name(x: Any) -> str:
    if isinstance(x, dict):
        return str(x.get("name") or "")
    return str(x or "")


class LastfmClient:
    def __init__(
        self,
        api_key: str | None = None,
        http: httpx.Client | None = None,
        cache: SqliteCache | None = None,
        limiter: RateLimiter = LASTFM_LIMITER,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = api_key if api_key is not None else get_settings().lastfm_api_key
        if not key:
            raise LastfmError(None, "no API key; set LASTFM_API_KEY in .env")
        self._api_key = key
        self._http = (
            http
            if http is not None
            else httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT})
        )
        self._cache = cache if cache is not None else SqliteCache()
        self._limiter = limiter
        self._sleep = sleep

    def _get(self, method: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Return the parsed body, or None when Last.fm says the item was not found."""
        # The api_key is deliberately left out of the cache key.
        key = make_key("lastfm", method, params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        request_params = {
            **params,
            "method": method,
            "api_key": self._api_key,
            "format": "json",
        }
        attempt = 0
        while True:
            self._limiter.wait()
            resp = self._http.get(
                BASE_URL, params=request_params, headers={"User-Agent": USER_AGENT}
            )
            try:
                body = resp.json()
            except ValueError:
                body = None
            if isinstance(body, dict) and "error" in body:
                code = _to_int(body.get("error"))
                message = str(body.get("message", ""))
                wait = (
                    SHORT_RETRY_SECONDS
                    if code in SHORT_RETRY_CODES
                    else RATE_LIMIT_WAIT_SECONDS
                    if code == RATE_LIMIT_CODE
                    else None
                )
                if wait is not None and attempt < MAX_RETRIES:
                    attempt += 1
                    self._sleep(wait)
                    continue
                if code == NOT_FOUND_CODE:
                    return None
                raise LastfmError(code, message)
            resp.raise_for_status()
            if not isinstance(body, dict):
                raise LastfmError(None, "response was not a JSON object")
            self._cache.set(key, body)
            return body

    def similar_tracks(self, artist: str, title: str, limit: int = 100) -> list[LastfmTrack]:
        data = self._get(
            "track.getSimilar",
            {"artist": artist, "track": title, "limit": limit, "autocorrect": 1},
        )
        if data is None:
            return []
        items = _as_list((data.get("similartracks") or {}).get("track"))
        return [
            LastfmTrack(
                name=str(i.get("name", "")),
                artist=_artist_name(i.get("artist")),
                match=_to_float(i.get("match")),
                mbid=i.get("mbid") or None,
            )
            for i in items
            if isinstance(i, dict)
        ]

    def similar_artists(self, artist: str, limit: int = 50) -> list[LastfmArtist]:
        data = self._get("artist.getSimilar", {"artist": artist, "limit": limit, "autocorrect": 1})
        if data is None:
            return []
        items = _as_list((data.get("similarartists") or {}).get("artist"))
        return [
            LastfmArtist(
                name=str(i.get("name", "")),
                match=_to_float(i.get("match")),
                mbid=i.get("mbid") or None,
            )
            for i in items
            if isinstance(i, dict)
        ]

    def top_tags(self, artist: str, title: str, limit: int = 20) -> list[LastfmTag]:
        data = self._get("track.getTopTags", {"artist": artist, "track": title, "autocorrect": 1})
        if data is None:
            return []
        items = _as_list((data.get("toptags") or {}).get("tag"))
        tags = [
            LastfmTag(name=str(i.get("name", "")), count=_to_int(i.get("count")))
            for i in items
            if isinstance(i, dict)
        ]
        return tags[:limit]
