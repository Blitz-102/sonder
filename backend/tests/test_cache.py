from pathlib import Path

import pytest

from sonder.config import get_settings
from sonder.sources.cache import SqliteCache, make_key


def test_round_trip(tmp_path: Path) -> None:
    cache = SqliteCache(tmp_path / "c.sqlite", ttl_seconds=60)
    assert cache.get("k") is None
    cache.set("k", {"a": [1, 2]})
    assert cache.get("k") == {"a": [1, 2]}


def test_expired_row_is_ignored(tmp_path: Path) -> None:
    now = [1000.0]
    cache = SqliteCache(tmp_path / "c.sqlite", ttl_seconds=60, clock=lambda: now[0])
    cache.set("k", {"a": 1})
    now[0] += 59
    assert cache.get("k") == {"a": 1}
    now[0] += 2
    assert cache.get("k") is None


def test_set_overwrites(tmp_path: Path) -> None:
    cache = SqliteCache(tmp_path / "c.sqlite")
    cache.set("k", {"v": 1})
    cache.set("k", {"v": 2})
    assert cache.get("k") == {"v": 2}


def test_key_is_stable_regardless_of_param_order() -> None:
    a = make_key("deezer", "/search", {"q": "x", "limit": 5})
    b = make_key("deezer", "/search", {"limit": 5, "q": "x"})
    assert a == b == "deezer:/search?limit=5&q=x"


def test_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "c.sqlite"
    SqliteCache(path)
    assert path.exists()


def test_defaults_to_settings_cache_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SONDER_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    try:
        cache = SqliteCache()
        assert cache.path == tmp_path / "data" / "cache.sqlite"
    finally:
        get_settings.cache_clear()
