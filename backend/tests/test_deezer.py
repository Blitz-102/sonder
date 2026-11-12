import sqlite3
from pathlib import Path

import httpx
import pytest
import respx

from sonder.sources.cache import SqliteCache
from sonder.sources.deezer import DeezerClient, DeezerError
from sonder.sources.ratelimit import RateLimiter
from tests.fixtures_deezer import (
    NOT_FOUND_ERROR,
    PREVIEW,
    QUOTA_ERROR,
    search_json,
    track_json,
)

API = "https://api.deezer.com"


class Sleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


@pytest.fixture
def sleeper() -> Sleeper:
    return Sleeper()


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache.sqlite"


@pytest.fixture
def client(cache_path: Path, sleeper: Sleeper) -> DeezerClient:
    limiter = RateLimiter(0.2, clock=lambda: 0.0, sleep=lambda s: None)
    return DeezerClient(
        http=httpx.Client(), cache=SqliteCache(cache_path), limiter=limiter, sleep=sleeper
    )


def test_search_parses_tracks(client: DeezerClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{API}/search").respond(json=search_json(track_json()))
    tracks = client.search("Radiohead", "Creep")
    assert route.call_count == 1
    assert route.calls[0].request.url.params["q"] == "Radiohead Creep"
    assert len(tracks) == 1
    t = tracks[0]
    assert (t.id, t.title, t.artist, t.album_title) == (
        3135556,
        "Creep",
        "Radiohead",
        "Pablo Honey",
    )
    assert t.cover_url == "https://cdn/cover_big.jpg"
    assert t.isrc == "GBAYE9200070"
    assert not hasattr(t, "preview")


def test_empty_plain_search_falls_back_to_track_query(
    client: DeezerClient, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{API}/search").mock(
        side_effect=[
            httpx.Response(200, json=search_json()),
            httpx.Response(200, json=search_json(track_json(title='Say "Hi"'))),
        ]
    )
    tracks = client.search("Radiohead", 'Say "Hi"')
    assert [c.request.url.params["q"] for c in route.calls] == [
        'Radiohead Say "Hi"',
        'track:"Say \\"Hi\\""',
    ]
    assert len(tracks) == 1


def test_search_returns_empty_when_both_queries_empty(
    client: DeezerClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(f"{API}/search").respond(json=search_json())
    assert client.search("Nobody", "Nothing") == []


def test_error_inside_200_raises(client: DeezerClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{API}/search").respond(
        json={"error": {"type": "OAuthException", "message": "bad", "code": 300}}
    )
    with pytest.raises(DeezerError) as exc:
        client.search("a", "b")
    assert exc.value.code == 300
    assert exc.value.type == "OAuthException"


def test_http_error_raises(client: DeezerClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{API}/track/1").respond(status_code=500)
    with pytest.raises(httpx.HTTPStatusError):
        client.get_track(1)


def test_get_track_not_found_returns_none(
    client: DeezerClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(f"{API}/track/1").respond(json=NOT_FOUND_ERROR)
    assert client.get_track(1) is None


def test_get_track_reads_bpm_and_isrc(client: DeezerClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{API}/track/3135556").respond(json=track_json(bpm=91.9))
    t = client.get_track(3135556)
    assert t is not None
    assert t.bpm == 91.9
    assert t.isrc == "GBAYE9200070"


def test_zero_bpm_becomes_none(client: DeezerClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{API}/track/5").respond(json=track_json(track_id=5, bpm=0))
    t = client.get_track(5)
    assert t is not None
    assert t.bpm is None


def test_quota_error_retries_after_sleep(
    client: DeezerClient, respx_mock: respx.MockRouter, sleeper: Sleeper
) -> None:
    route = respx_mock.get(f"{API}/search").mock(
        side_effect=[
            httpx.Response(200, json=QUOTA_ERROR),
            httpx.Response(200, json=QUOTA_ERROR),
            httpx.Response(200, json=search_json(track_json())),
        ]
    )
    tracks = client.search("Radiohead", "Creep")
    assert len(tracks) == 1
    assert route.call_count == 3
    assert sleeper.calls == [5.0, 5.0]


def test_quota_error_gives_up_after_three_retries(
    client: DeezerClient, respx_mock: respx.MockRouter, sleeper: Sleeper
) -> None:
    route = respx_mock.get(f"{API}/search").respond(json=QUOTA_ERROR)
    with pytest.raises(DeezerError) as exc:
        client.search("a", "b")
    assert exc.value.code == 4
    assert route.call_count == 4
    assert sleeper.calls == [5.0, 5.0, 5.0]


def test_cache_db_never_contains_preview(
    client: DeezerClient, respx_mock: respx.MockRouter, cache_path: Path
) -> None:
    respx_mock.get(f"{API}/search").respond(json=search_json(track_json()))
    respx_mock.get(f"{API}/track/3135556").respond(json=track_json())
    respx_mock.get(f"{API}/artist/399/top").respond(json=search_json(track_json()))
    client.search("Radiohead", "Creep")
    client.get_track(3135556)
    client.artist_top(399)
    client.fresh_preview_url(3135556)

    conn = sqlite3.connect(cache_path)
    try:
        values = [row[0] for row in conn.execute("SELECT value FROM http_cache")]
    finally:
        conn.close()
    assert len(values) == 3
    for v in values:
        assert "preview" not in v
        assert "hdnea" not in v
        assert "dzcdn" not in v


def test_fresh_preview_url_always_hits_network(
    client: DeezerClient, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{API}/track/3135556").respond(json=track_json())
    assert client.get_track(3135556) is not None
    assert route.call_count == 1
    assert client.fresh_preview_url(3135556) == PREVIEW
    assert client.fresh_preview_url(3135556) == PREVIEW
    assert route.call_count == 3


def test_fresh_preview_url_missing_or_empty(
    client: DeezerClient, respx_mock: respx.MockRouter
) -> None:
    respx_mock.get(f"{API}/track/1").respond(json=track_json(track_id=1, preview=""))
    respx_mock.get(f"{API}/track/2").respond(json=NOT_FOUND_ERROR)
    assert client.fresh_preview_url(1) is None
    assert client.fresh_preview_url(2) is None


def test_second_identical_search_served_from_cache(
    client: DeezerClient, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.get(f"{API}/search").respond(json=search_json(track_json()))
    first = client.search("Radiohead", "Creep")
    second = client.search("Radiohead", "Creep")
    assert first == second
    assert route.call_count == 1


def test_errors_are_not_cached(client: DeezerClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{API}/track/1").mock(
        side_effect=[
            httpx.Response(200, json=NOT_FOUND_ERROR),
            httpx.Response(200, json=track_json(track_id=1)),
        ]
    )
    assert client.get_track(1) is None
    assert client.get_track(1) is not None
    assert route.call_count == 2


def test_artist_top(client: DeezerClient, respx_mock: respx.MockRouter) -> None:
    route = respx_mock.get(f"{API}/artist/399/top").respond(
        json=search_json(track_json(), track_json(track_id=2, title="Karma Police"))
    )
    tracks = client.artist_top(399, limit=2)
    assert route.calls[0].request.url.params["limit"] == "2"
    assert [t.title for t in tracks] == ["Creep", "Karma Police"]


def test_artist_related(client: DeezerClient, respx_mock: respx.MockRouter) -> None:
    respx_mock.get(f"{API}/artist/399/related").respond(
        json={"data": [{"id": 1, "name": "Muse"}, {"id": 2, "name": "Placebo"}], "total": 2}
    )
    artists = client.artist_related(399)
    assert [(a.id, a.name) for a in artists] == [(1, "Muse"), (2, "Placebo")]
