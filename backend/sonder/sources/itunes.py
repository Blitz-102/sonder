import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sonder.sources.cache import SqliteCache, make_key
from sonder.sources.ratelimit import ITUNES_LIMITER, RateLimiter

BASE_URL = "https://itunes.apple.com"
BACKOFF_SECONDS = 60.0


@dataclass(frozen=True)
class ItunesTrack:
    track_id: int
    title: str
    artist: str
    album: str
    preview_url: str | None
    duration_ms: int
    artwork_url: str | None


def _parse(d: dict[str, Any]) -> ItunesTrack:
    return ItunesTrack(
        track_id=int(d["trackId"]),
        title=d.get("trackName", ""),
        artist=d.get("artistName", ""),
        album=d.get("collectionName", ""),
        preview_url=d.get("previewUrl") or None,
        duration_ms=int(d.get("trackTimeMillis") or 0),
        artwork_url=d.get("artworkUrl100") or None,
    )


class ItunesClient:
    def __init__(
        self,
        http: httpx.Client | None = None,
        cache: SqliteCache | None = None,
        limiter: RateLimiter = ITUNES_LIMITER,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http if http is not None else httpx.Client(timeout=15.0)
        self._cache = cache if cache is not None else SqliteCache()
        self._limiter = limiter
        self._sleep = sleep

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        key = make_key("itunes", path, params)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        retried = False
        while True:
            self._limiter.wait()
            resp = self._http.get(f"{BASE_URL}{path}", params=params)
            if resp.status_code in (403, 429) and not retried:
                retried = True
                self._sleep(BACKOFF_SECONDS)
                continue
            resp.raise_for_status()
            # iTunes often serves JSON as text/javascript, so parse the body directly.
            data: dict[str, Any] = json.loads(resp.text)
            self._cache.set(key, data)
            return data

    def search_song(
        self, artist: str, title: str, limit: int = 10, country: str = "US"
    ) -> list[ItunesTrack]:
        data = self._get(
            "/search",
            {
                "term": f"{artist} {title}",
                "entity": "song",
                "media": "music",
                "limit": limit,
                "country": country,
            },
        )
        return [_parse(r) for r in data.get("results") or [] if r.get("trackId")]
