import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sonder.sources.cache import SqliteCache, make_key
from sonder.sources.ratelimit import DEEZER_LIMITER, RateLimiter

BASE_URL = "https://api.deezer.com"
QUOTA_CODE = 4
NOT_FOUND_CODE = 800
QUOTA_WAIT_SECONDS = 5.0
QUOTA_MAX_RETRIES = 3


class DeezerError(Exception):
    def __init__(self, type: str, message: str, code: int | None = None) -> None:
        super().__init__(f"Deezer error {code} ({type}): {message}")
        self.type = type
        self.message = message
        self.code = code


@dataclass(frozen=True)
class DeezerArtist:
    id: int
    name: str


@dataclass(frozen=True)
class DeezerTrack:
    id: int
    title: str
    title_short: str
    artist: str
    album_title: str
    cover_url: str | None
    duration: int
    isrc: str | None
    bpm: float | None
    rank: int


def _strip_preview(data: Any) -> Any:
    """Recursively drop every `preview` key (ground rule 2: never store preview URLs)."""
    if isinstance(data, dict):
        return {k: _strip_preview(v) for k, v in data.items() if k != "preview"}
    if isinstance(data, list):
        return [_strip_preview(v) for v in data]
    return data


def _parse_track(d: dict[str, Any]) -> DeezerTrack:
    album = d.get("album") or {}
    bpm = d.get("bpm")
    return DeezerTrack(
        id=int(d["id"]),
        title=d.get("title", ""),
        title_short=d.get("title_short") or d.get("title", ""),
        artist=(d.get("artist") or {}).get("name", ""),
        album_title=album.get("title", ""),
        cover_url=album.get("cover_big") or album.get("cover_medium") or album.get("cover"),
        duration=int(d.get("duration") or 0),
        isrc=d.get("isrc") or None,
        bpm=float(bpm) if bpm else None,
        rank=int(d.get("rank") or 0),
    )


def _escape(s: str) -> str:
    return s.replace('"', '\\"')


class DeezerClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        cache: SqliteCache | None = None,
        limiter: RateLimiter = DEEZER_LIMITER,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http if http is not None else httpx.Client(timeout=15.0)
        self._cache = cache if cache is not None else SqliteCache()
        self._limiter = limiter
        self._sleep = sleep

    def _get(
        self, path: str, params: dict[str, Any] | None = None, use_cache: bool = True
    ) -> dict[str, Any]:
        key = make_key("deezer", path, params)
        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        attempt = 0
        while True:
            self._limiter.wait()
            resp = self._http.get(f"{BASE_URL}{path}", params=params)
            resp.raise_for_status()
            data = resp.json()
            err = data.get("error") if isinstance(data, dict) else None
            if not err:
                break
            error = DeezerError(
                str(err.get("type", "")), str(err.get("message", "")), err.get("code")
            )
            is_quota = error.code == QUOTA_CODE or "quota" in error.type.lower()
            if is_quota and attempt < QUOTA_MAX_RETRIES:
                attempt += 1
                self._sleep(QUOTA_WAIT_SECONDS)
                continue
            raise error
        if use_cache:
            self._cache.set(key, _strip_preview(data))
        return data

    def search(self, artist: str, title: str, limit: int = 10) -> list[DeezerTrack]:
        # The advanced `artist:"..."` filter currently returns nothing, so query plainly first.
        for q in (f"{artist} {title}", f'track:"{_escape(title)}"'):
            data = self._get("/search", {"q": q, "limit": limit})
            items = data.get("data") or []
            if items:
                return [_parse_track(i) for i in items]
        return []

    def get_track(self, track_id: int) -> DeezerTrack | None:
        try:
            data = self._get(f"/track/{track_id}")
        except DeezerError as e:
            if e.code == NOT_FOUND_CODE:
                return None
            raise
        return _parse_track(data)

    def artist_top(self, artist_id: int, limit: int = 5) -> list[DeezerTrack]:
        data = self._get(f"/artist/{artist_id}/top", {"limit": limit})
        return [_parse_track(i) for i in data.get("data") or []]

    def artist_related(self, artist_id: int, limit: int = 20) -> list[DeezerArtist]:
        data = self._get(f"/artist/{artist_id}/related", {"limit": limit})
        return [
            DeezerArtist(id=int(i["id"]), name=i.get("name", "")) for i in data.get("data") or []
        ]

    def fresh_preview_url(self, track_id: int) -> str | None:
        """Fetch a fresh preview URL. Never cached: the URL carries an expiring token."""
        try:
            data = self._get(f"/track/{track_id}", use_cache=False)
        except DeezerError as e:
            if e.code == NOT_FOUND_CODE:
                return None
            raise
        return data.get("preview") or None
