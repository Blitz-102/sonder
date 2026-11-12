import sqlite3
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from sonder.sources.cache import SqliteCache
from sonder.sources.lastfm import LastfmClient, LastfmError
from sonder.sources.ratelimit import RateLimiter

URL = "https://ws.audioscrobbler.com/2.0/"
KEY = "SECRET-KEY-123"


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache.sqlite"


@pytest.fixture
def client(cache_path: Path, sleeps: list[float]) -> LastfmClient:
    limiter = RateLimiter(1.0, clock=lambda: 0.0, sleep=lambda s: None)
    return LastfmClient(
        api_key=KEY,
        http=httpx.Client(),
        cache=SqliteCache(cache_path),
        limiter=limiter,
        sleep=sleeps.append,
    )


def similar_tracks_json(*tracks: dict[str, Any]) -> dict[str, Any]:
    return {"similartracks": {"track": list(tracks)}}


TRACK: dict[str, Any] = {
    "name": "Karma Police",
    "match": 0.87,
    "mbid": "abc",
    "artist": {"name": "Radiohead", "mbid": "r"},
}


def test_similar_tracks_parses(client: LastfmClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(URL).respond(json=similar_tracks_json(TRACK))
    tracks = client.similar_tracks("Radiohead", "Creep", limit=10)
    params = route.calls[0].request.url.params
    assert params["method"] == "track.getSimilar"
    assert params["autocorrect"] == "1"
    assert params["limit"] == "10"
    assert params["format"] == "json"
    assert len(tracks) == 1
    assert (tracks[0].name, tracks[0].artist, tracks[0].match, tracks[0].mbid) == (
        "Karma Police",
        "Radiohead",
        0.87,
        "abc",
    )


def test_similar_tracks_single_object_and_empty(
    client: LastfmClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(URL, params={"artist": "A"}).respond(json={"similartracks": {"track": TRACK}})
    respx_mock.get(URL, params={"artist": "B"}).respond(json={"similartracks": {"track": []}})
    respx_mock.get(URL, params={"artist": "C"}).respond(json={"similartracks": {}})
    respx_mock.get(URL, params={"artist": "D"}).respond(json={})
    assert len(client.similar_tracks("A", "x")) == 1
    assert client.similar_tracks("B", "x") == []
    assert client.similar_tracks("C", "x") == []
    assert client.similar_tracks("D", "x") == []


def test_similar_artists_casts_string_match(
    client: LastfmClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(URL).respond(
        json={
            "similarartists": {
                "artist": [{"name": "Blur", "match": "0.75", "mbid": ""}, {"name": "Suede"}]
            }
        }
    )
    artists = client.similar_artists("Oasis")
    assert artists[0].name == "Blur"
    assert artists[0].match == 0.75
    assert isinstance(artists[0].match, float)
    assert artists[0].mbid is None
    assert artists[1].match == 0.0


def test_similar_artists_single_object(client: LastfmClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(URL).respond(json={"similarartists": {"artist": {"name": "Blur", "match": "1"}}})
    assert [a.name for a in client.similar_artists("Oasis")] == ["Blur"]


def test_top_tags_sliced_and_counted(client: LastfmClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(URL).respond(
        json={"toptags": {"tag": [{"name": f"t{i}", "count": 100 - i} for i in range(30)]}}
    )
    tags = client.top_tags("Radiohead", "Creep", limit=5)
    assert "limit" not in route.calls[0].request.url.params
    assert [t.name for t in tags] == ["t0", "t1", "t2", "t3", "t4"]
    assert tags[0].count == 100


def test_error_29_retries_then_succeeds(
    client: LastfmClient, respx_mock: respx.MockRouter, sleeps: list[float]
) -> None:
    route = respx_mock.get(URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": 29, "message": "Rate limit exceeded"}),
            httpx.Response(200, json=similar_tracks_json(TRACK)),
        ]
    )
    assert len(client.similar_tracks("Radiohead", "Creep")) == 1
    assert route.call_count == 2
    assert sleeps == [30.0]


def test_error_16_retries_with_short_wait(
    client: LastfmClient, respx_mock: respx.MockRouter, sleeps: list[float]
) -> None:
    respx_mock.get(URL).mock(
        side_effect=[
            httpx.Response(500, json={"error": 16, "message": "Temporary"}),
            httpx.Response(200, json=similar_tracks_json(TRACK)),
        ]
    )
    client.similar_tracks("Radiohead", "Creep")
    assert sleeps == [5.0]


def test_retries_are_bounded(
    client: LastfmClient, respx_mock: respx.MockRouter, sleeps: list[float]
) -> None:
    route = respx_mock.get(URL).respond(429, json={"error": 29, "message": "Rate limit"})
    with pytest.raises(LastfmError) as exc:
        client.similar_tracks("Radiohead", "Creep")
    assert exc.value.code == 29
    assert route.call_count == 4
    assert sleeps == [30.0, 30.0, 30.0]


def test_error_6_returns_empty_and_is_not_cached(
    client: LastfmClient, respx_mock: respx.MockRouter, cache_path: Path
) -> None:
    route = respx_mock.get(URL).respond(404, json={"error": 6, "message": "Track not found"})
    assert client.similar_tracks("Nobody", "Nothing") == []
    assert client.similar_artists("Nobody") == []
    assert client.top_tags("Nobody", "Nothing") == []
    assert route.call_count == 3
    conn = sqlite3.connect(cache_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM http_cache").fetchone()[0] == 0
    finally:
        conn.close()


def test_error_10_with_http_403_raises(client: LastfmClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(URL).respond(403, json={"error": 10, "message": "Invalid API key"})
    with pytest.raises(LastfmError) as exc:
        client.similar_tracks("Radiohead", "Creep")
    assert exc.value.code == 10


def test_http_error_without_json_raises(client: LastfmClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(URL).respond(500, text="boom")
    with pytest.raises(httpx.HTTPStatusError):
        client.similar_tracks("Radiohead", "Creep")


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class NoKey:
        lastfm_api_key = None

    monkeypatch.setattr("sonder.sources.lastfm.get_settings", lambda: NoKey())
    with pytest.raises(LastfmError, match="LASTFM_API_KEY"):
        LastfmClient(cache=SqliteCache(tmp_path / "c.sqlite"))


def test_api_key_never_cached(
    client: LastfmClient, respx_mock: respx.MockRouter, cache_path: Path
) -> None:
    route = respx_mock.get(URL).respond(json=similar_tracks_json(TRACK))
    client.similar_tracks("Radiohead", "Creep")
    assert route.calls[0].request.url.params["api_key"] == KEY
    conn = sqlite3.connect(cache_path)
    try:
        rows = conn.execute("SELECT key, value FROM http_cache").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert all(KEY not in key and KEY not in value for key, value in rows)
    assert all("api_key" not in key for key, _ in rows)


def test_second_call_served_from_cache(client: LastfmClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(URL).respond(json=similar_tracks_json(TRACK))
    first = client.similar_tracks("Radiohead", "Creep")
    second = client.similar_tracks("Radiohead", "Creep")
    assert first == second
    assert route.call_count == 1


def test_user_agent_sent(client: LastfmClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(URL).respond(json=similar_tracks_json(TRACK))
    client.similar_tracks("Radiohead", "Creep")
    assert route.calls[0].request.headers["user-agent"] == "sonder/0.1"
