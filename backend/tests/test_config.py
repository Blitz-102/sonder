from pathlib import Path

import pytest

from sonder.config import REPO_ROOT, Settings, get_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    monkeypatch.delenv("SONDER_DATA_DIR", raising=False)
    monkeypatch.delenv("GUITAR_PRESENCE_MIN", raising=False)
    get_settings.cache_clear()


def test_defaults_resolve_under_repo_root() -> None:
    settings = Settings(_env_file=None)
    assert settings.lastfm_api_key is None
    assert settings.data_dir == REPO_ROOT / "data"
    assert settings.cache_db == REPO_ROOT / "data" / "cache.sqlite"
    assert settings.index_db == REPO_ROOT / "data" / "sonder.sqlite"
    assert settings.tmp_dir == REPO_ROOT / "data" / "tmp"


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LASTFM_API_KEY", "abc123")
    monkeypatch.setenv("SONDER_DATA_DIR", str(tmp_path))
    settings = Settings(_env_file=None)
    assert settings.lastfm_api_key == "abc123"
    assert settings.data_dir == tmp_path
    assert settings.cache_db == tmp_path / "cache.sqlite"


def test_reads_values_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(f"LASTFM_API_KEY=fromfile\nSONDER_DATA_DIR={tmp_path / 'd'}\n")
    settings = Settings(_env_file=env_file)
    assert settings.lastfm_api_key == "fromfile"
    assert settings.data_dir == tmp_path / "d"


def test_ensure_dirs_creates_data_and_tmp(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    assert not settings.data_dir.exists()
    settings.ensure_dirs()
    assert settings.data_dir.is_dir()
    assert settings.tmp_dir.is_dir()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_guitar_presence_min_default() -> None:
    assert Settings(_env_file=None).guitar_presence_min == 0.15


def test_guitar_presence_min_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUITAR_PRESENCE_MIN", "0.3")
    assert Settings(_env_file=None).guitar_presence_min == 0.3
