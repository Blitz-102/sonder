import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sonder.audio.clap import MODEL_VERSION as VIBE_VERSION
from sonder.audio.guitar import GUITAR_MODEL_VERSION
from sonder.index import db
from sonder.index.db import Analysis, TrackRecord


def rec(
    deezer_id: int | None = 1,
    title: str = "Creep",
    artist: str = "Radiohead",
    local_path: str | None = None,
    source: str = "deezer",
    album: str | None = "Pablo Honey",
) -> TrackRecord:
    return TrackRecord(
        deezer_id=deezer_id,
        isrc="GBAYE9200070",
        title=title,
        artist=artist,
        album=album,
        cover_url=None,
        duration=238.0,
        source=source,
        local_path=local_path,
    )


def analysis(seed: int = 0, guitar: bool = True, dim: int = 512, **kw: object) -> Analysis:
    rng = np.random.default_rng(seed)
    return Analysis(
        tempo=120.0,
        key="E",
        mode="minor",
        loudness_db=-14.0,
        guitar_presence=0.4,
        vibe=rng.standard_normal(dim).astype(np.float32),
        guitar=rng.standard_normal(dim).astype(np.float32) if guitar else None,
        vibe_model_version=VIBE_VERSION,
        guitar_model_version=GUITAR_MODEL_VERSION if guitar else None,
        features_model_version=db.FEATURES_VERSION,
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = db.connect(tmp_path / "sub" / "index.sqlite")
    yield c  # type: ignore[misc]
    c.close()


def test_schema_created_and_reopen_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "i.sqlite"
    c = db.connect(path)
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert c.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    tables = {r["name"] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tracks", "features", "vectors", "failures"} <= tables
    cols = {r["name"] for r in c.execute("PRAGMA table_info(tracks)")}
    assert not any("preview" in col for col in cols)
    track_id = db.upsert_track(c, rec())
    c.close()

    again = db.connect(path)
    assert db.get_track(again, track_id) is not None
    assert db.count_tracks(again) == 1
    again.close()


def test_newer_schema_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "i.sqlite"
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    raw.commit()
    raw.close()
    with pytest.raises(db.SchemaVersionError, match="newer"):
        db.connect(path)


def test_upsert_idempotent_on_deezer_id(conn: sqlite3.Connection) -> None:
    a = db.upsert_track(conn, rec(title="Creep"))
    b = db.upsert_track(conn, rec(title="Creep (Remastered)", album="Other"))
    assert a == b
    assert db.count_tracks(conn) == 1
    row = db.get_track(conn, a)
    assert row is not None
    assert (row["title"], row["album"]) == ("Creep (Remastered)", "Other")
    assert db.upsert_track(conn, rec(deezer_id=2)) != a


def test_upsert_idempotent_on_local_path(conn: sqlite3.Connection) -> None:
    local = rec(deezer_id=None, source="local", local_path="C:/m/a.mp3", title="A")
    a = db.upsert_track(conn, local)
    b = db.upsert_track(
        conn, rec(deezer_id=None, source="local", local_path="C:/m/a.mp3", title="B")
    )
    assert a == b
    row = db.get_track(conn, a)
    assert row is not None and row["title"] == "B"
    assert db.get_track(conn, 999) is None


def test_upsert_needs_a_natural_key(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="deezer_id or a local_path"):
        db.upsert_track(conn, rec(deezer_id=None))


def test_source_check_constraint(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.upsert_track(conn, rec(source="spotify"))


def test_vector_round_trip_exact_and_versions(conn: sqlite3.Connection) -> None:
    a1, a2 = analysis(1), analysis(2)
    t1 = db.upsert_track(conn, rec(deezer_id=1))
    t2 = db.upsert_track(conn, rec(deezer_id=2))
    db.save_analysis(conn, t1, a1)
    db.save_analysis(conn, t2, a2)
    ids, vibe = db.load_vectors(conn, "vibe")
    assert ids.dtype == np.int64 and vibe.dtype == np.float32
    assert ids.tolist() == [t1, t2]
    assert vibe.shape == (2, 512)
    assert np.array_equal(vibe[0], a1.vibe) and np.array_equal(vibe[1], a2.vibe)
    gids, guitar = db.load_vectors(conn, "guitar")
    assert gids.tolist() == [t1, t2]
    assert np.array_equal(guitar[0], a1.guitar)

    versions = {
        (r["track_id"], r["kind"]): (r["model_version"], r["dim"])
        for r in conn.execute("SELECT * FROM vectors")
    }
    assert versions[(t1, "vibe")] == (VIBE_VERSION, 512)
    assert versions[(t1, "guitar")] == (GUITAR_MODEL_VERSION, 512)
    feat = conn.execute("SELECT * FROM features WHERE track_id = ?", (t1,)).fetchone()
    assert feat["model_version"] == db.FEATURES_VERSION == "librosa-feature-tempo-kk-v1"
    assert (feat["tempo"], feat["key"], feat["mode"], feat["guitar_presence"]) == (
        120.0,
        "E",
        "minor",
        0.4,
    )


def test_save_accepts_float64_and_non_contiguous(conn: sqlite3.Connection) -> None:
    t = db.upsert_track(conn, rec())
    base = np.random.default_rng(0).standard_normal((512, 2))
    a = analysis(0, guitar=False)
    a = Analysis(**{**a.__dict__, "vibe": base[:, 0]})
    db.save_analysis(conn, t, a)
    _, m = db.load_vectors(conn, "vibe")
    assert np.array_equal(m[0], base[:, 0].astype(np.float32))


@pytest.mark.parametrize(
    "bad",
    [np.zeros((2, 4), dtype=np.float32), np.zeros(0, dtype=np.float32), np.arange(4)],
)
def test_save_rejects_bad_vectors(conn: sqlite3.Connection, bad: np.ndarray) -> None:
    t = db.upsert_track(conn, rec())
    a = analysis(0)
    with pytest.raises(ValueError):
        db.save_analysis(conn, t, Analysis(**{**a.__dict__, "vibe": bad}))
    nan = np.full(512, np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        db.save_analysis(conn, t, Analysis(**{**a.__dict__, "guitar": nan}))
    # A rejected save leaves nothing behind.
    assert conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0


def test_save_is_atomic(conn: sqlite3.Connection) -> None:
    t = db.upsert_track(conn, rec())
    db.save_analysis(conn, t, analysis(1))
    broken = Analysis(
        **{**analysis(2).__dict__, "vibe_model_version": None, "tempo": 99.0}  # NOT NULL
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.save_analysis(conn, t, broken)
    # features were replaced inside the failed transaction: it must have rolled back.
    feat = conn.execute("SELECT * FROM features WHERE track_id = ?", (t,)).fetchone()
    _, vibe = db.load_vectors(conn, "vibe")
    assert np.array_equal(vibe[0], analysis(1).vibe)
    assert feat["tempo"] == 120.0


def test_is_finished_tracks_model_version(conn: sqlite3.Connection) -> None:
    assert not db.is_finished(conn, deezer_id=1, vibe_model_version=VIBE_VERSION)
    t = db.upsert_track(conn, rec())
    assert not db.is_finished(conn, deezer_id=1, vibe_model_version=VIBE_VERSION)  # no analysis
    db.save_analysis(conn, t, analysis())
    assert db.is_finished(conn, deezer_id=1, vibe_model_version=VIBE_VERSION)
    assert not db.is_finished(conn, deezer_id=1, vibe_model_version="clap-v2")
    # Re-embedding with a new model makes it finished again for that version.
    a = analysis()
    db.save_analysis(conn, t, Analysis(**{**a.__dict__, "vibe_model_version": "clap-v2"}))
    assert db.is_finished(conn, deezer_id=1, vibe_model_version="clap-v2")
    assert not db.is_finished(conn, deezer_id=1, vibe_model_version=VIBE_VERSION)
    db.save_analysis(conn, t, a)
    assert db.is_finished(conn, deezer_id=1, vibe_model_version=VIBE_VERSION)
    with pytest.raises(ValueError):
        db.is_finished(conn, vibe_model_version=VIBE_VERSION)


def test_is_finished_by_local_path(conn: sqlite3.Connection) -> None:
    t = db.upsert_track(conn, rec(deezer_id=None, source="local", local_path="/x/a.wav"))
    db.save_analysis(conn, t, analysis())
    assert db.is_finished(conn, local_path="/x/a.wav", vibe_model_version=VIBE_VERSION)
    assert not db.is_finished(conn, local_path="/x/b.wav", vibe_model_version=VIBE_VERSION)


def test_stale_guitar_vector_deleted_on_reingest(conn: sqlite3.Connection) -> None:
    t = db.upsert_track(conn, rec())
    db.save_analysis(conn, t, analysis(1, guitar=True))
    assert db.load_vectors(conn, "guitar")[0].tolist() == [t]
    db.save_analysis(conn, t, analysis(1, guitar=False))
    ids, matrix = db.load_vectors(conn, "guitar")
    assert ids.size == 0 and matrix.shape[0] == 0
    assert db.load_vectors(conn, "vibe")[0].tolist() == [t]


def test_deleting_a_track_cascades(conn: sqlite3.Connection) -> None:
    t = db.upsert_track(conn, rec())
    db.save_analysis(conn, t, analysis())
    with conn:
        conn.execute("DELETE FROM tracks WHERE id = ?", (t,))
    assert conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM features").fetchone()[0] == 0


def test_load_vectors_empty_and_unknown_kind(conn: sqlite3.Connection) -> None:
    ids, matrix = db.load_vectors(conn, "vibe")
    assert ids.dtype == np.int64 and ids.size == 0 and matrix.dtype == np.float32
    with pytest.raises(ValueError):
        db.load_vectors(conn, "bass")


def test_failures_count_scrub_and_clear(conn: sqlite3.Connection) -> None:
    url = "https://cdnt-preview.dzcdn.net/x.mp3?hdnea=exp=1~hmac=abc"
    db.record_failure(conn, "123", f"HTTPStatusError: 403 for url '{url}'")
    db.record_failure(conn, "123", "no preview")
    db.record_failure(conn, "/m/a.mp3", f"bad {url} and http://other.example/y")
    assert db.failed_keys(conn) == {"123", "/m/a.mp3"}
    row = conn.execute("SELECT * FROM failures WHERE key = '123'").fetchone()
    assert (row["attempts"], row["reason"]) == (2, "no preview")
    row = conn.execute("SELECT * FROM failures WHERE key = '/m/a.mp3'").fetchone()
    assert row["reason"] == "bad <url> and <url>"
    db.record_failure(conn, "u", f"HTTPStatusError: 403 for url '{url}'")
    reason = conn.execute("SELECT reason FROM failures WHERE key = 'u'").fetchone()["reason"]
    assert "dzcdn" not in reason and "hdnea" not in reason
    db.clear_failure(conn, "123")
    db.clear_failure(conn, "nope")
    assert db.failed_keys(conn) == {"/m/a.mp3", "u"}


def test_failure_reason_is_truncated(conn: sqlite3.Connection) -> None:
    db.record_failure(conn, "k", "x" * 5000)
    reason = conn.execute("SELECT reason FROM failures").fetchone()["reason"]
    assert len(reason) == db.MAX_REASON_CHARS


def test_default_path_comes_from_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONDER_DATA_DIR", str(tmp_path / "data"))
    db.get_settings.cache_clear()
    try:
        c = db.connect()
        c.close()
    finally:
        db.get_settings.cache_clear()
    assert (tmp_path / "data" / "sonder.sqlite").exists()


def test_importing_index_does_not_load_ml_libs() -> None:
    code = (
        "import sonder.index.db, sonder.index.search, sys; "
        "bad = [m for m in ('torch', 'transformers', 'demucs_infer') if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
