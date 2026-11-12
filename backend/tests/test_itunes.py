import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from sonder.sources.cache import SqliteCache
from sonder.sources.itunes import ItunesClient
from sonder.sources.ratelimit import RateLimiter

URL = "https://itunes.apple.com/search"

RESULT: dict[str, Any] = {
    "trackId": 42,
    "trackName": "Creep",
    "artistName": "Radiohead",
    "collectionName": "Pablo Honey",
    "previewUrl": "https://audio-ssl.itunes.apple.com/p.m4a",
    "trackTimeMillis": 238000,
    "artworkUrl100": "https://is1/100x100bb.jpg",
}


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def client(tmp_path: Path, sleeps: list[float]) -> ItunesClient:
    limiter = RateLimiter(3.0, clock=lambda: 0.0, sleep=lambda s: None)
    return ItunesClient(
        http=httpx.Client(),
        cache=SqliteCache(tmp_path / "c.sqlite"),
        limiter=limiter,
        sleep=sleeps.append,
    )


def test_parses_results(client: ItunesClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(URL).respond(json={"resultCount": 1, "results": [RESULT]})
    tracks = client.search_song("Radiohead", "Creep")
    params = route.calls[0].request.url.params
    assert params["term"] == "Radiohead Creep"
    assert params["entity"] == "song"
    assert params["media"] == "music"
    assert len(tracks) == 1
    t = tracks[0]
    assert (t.track_id, t.title, t.artist, t.album) == (42, "Creep", "Radiohead", "Pablo Honey")
    assert t.preview_url == RESULT["previewUrl"]
    assert t.duration_ms == 238000
    assert t.artwork_url == RESULT["artworkUrl100"]


def test_no_results(client: ItunesClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(URL).respond(json={"resultCount": 0, "results": []})
    assert client.search_song("x", "y") == []


def test_result_without_preview(client: ItunesClient, respx_mock: respx.MockRouter) -> None:
    item = {k: v for k, v in RESULT.items() if k != "previewUrl"}
    respx_mock.get(URL).respond(json={"resultCount": 1, "results": [item]})
    assert client.search_song("a", "b")[0].preview_url is None


def test_text_javascript_content_type_parses(
    client: ItunesClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(URL).respond(
        content=json.dumps({"resultCount": 1, "results": [RESULT]}),
        headers={"content-type": "text/javascript; charset=utf-8"},
    )
    assert client.search_song("Radiohead", "Creep")[0].track_id == 42


def test_429_backs_off_and_retries_once(
    client: ItunesClient, respx_mock: respx.MockRouter, sleeps: list[float]
) -> None:
    route = respx_mock.get(URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json={"resultCount": 1, "results": [RESULT]}),
        ]
    )
    assert len(client.search_song("Radiohead", "Creep")) == 1
    assert route.call_count == 2
    assert sleeps == [60.0]


def test_repeated_429_raises(
    client: ItunesClient, respx_mock: respx.MockRouter, sleeps: list[float]
) -> None:
    route = respx_mock.get(URL).respond(status_code=429)
    with pytest.raises(httpx.HTTPStatusError):
        client.search_song("a", "b")
    assert route.call_count == 2
    assert sleeps == [60.0]


def test_second_search_is_cached(client: ItunesClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(URL).respond(json={"resultCount": 1, "results": [RESULT]})
    client.search_song("Radiohead", "Creep")
    client.search_song("Radiohead", "Creep")
    assert route.call_count == 1
