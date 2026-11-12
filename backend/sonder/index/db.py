"""SQLite track index: tracks, features, vectors and ingest failures.

Ground rule 2: there is deliberately no preview-URL column, and failure reasons are scrubbed of
URLs before they are stored.
"""

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sonder.audio.clap import MODEL_VERSION as VIBE_MODEL_VERSION
from sonder.audio.guitar import GUITAR_MODEL_VERSION
from sonder.config import get_settings

SCHEMA_VERSION = 1
FEATURES_VERSION = "librosa-feature-tempo-kk-v1"
VECTOR_KINDS = ("vibe", "guitar")
MAX_REASON_CHARS = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY,
    deezer_id INTEGER UNIQUE,
    isrc TEXT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    cover_url TEXT,
    duration REAL,
    source TEXT NOT NULL CHECK(source IN ('deezer','itunes','local')),
    local_path TEXT UNIQUE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS features (
    track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    tempo REAL,
    key TEXT,
    mode TEXT,
    loudness_db REAL,
    guitar_presence REAL,
    model_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vectors (
    track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    kind TEXT CHECK(kind IN ('vibe','guitar')),
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    model_version TEXT NOT NULL,
    PRIMARY KEY(track_id, kind)
);
CREATE TABLE IF NOT EXISTS failures (
    key TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    last_at REAL NOT NULL
);
"""


class SchemaVersionError(RuntimeError):
    """The database was written by a newer version of sonder."""


@dataclass(frozen=True)
class TrackRecord:
    deezer_id: int | None
    isrc: str | None
    title: str
    artist: str
    album: str | None
    cover_url: str | None
    duration: float | None
    source: str
    local_path: str | None


@dataclass(frozen=True)
class Analysis:
    tempo: float | None
    key: str | None
    mode: str | None
    loudness_db: float | None
    guitar_presence: float | None
    vibe: np.ndarray
    guitar: np.ndarray | None
    vibe_model_version: str = VIBE_MODEL_VERSION
    guitar_model_version: str | None = GUITAR_MODEL_VERSION
    features_model_version: str = FEATURES_VERSION


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating and migrating if needed) the index database."""
    db_path = path if path is not None else get_settings().index_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        _migrate(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"index database has schema version {version}, newer than the supported "
            f"{SCHEMA_VERSION}; upgrade sonder"
        )
    if version == 0:
        conn.executescript(_SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()


def scrub_urls(text: str) -> str:
    """Replace URLs (preview URLs carry expiring tokens) with a placeholder."""
    return re.sub(r"https?://\S+", "<url>", text)


def _find_id(conn: sqlite3.Connection, deezer_id: int | None, local_path: str | None) -> int | None:
    if deezer_id is not None:
        row = conn.execute("SELECT id FROM tracks WHERE deezer_id = ?", (deezer_id,)).fetchone()
        if row is not None:
            return int(row["id"])
    if local_path is not None:
        row = conn.execute("SELECT id FROM tracks WHERE local_path = ?", (local_path,)).fetchone()
        if row is not None:
            return int(row["id"])
    return None


def upsert_track(conn: sqlite3.Connection, rec: TrackRecord) -> int:
    """Insert or update a track, keyed by deezer_id or local_path. Returns the track id."""
    if rec.deezer_id is None and rec.local_path is None:
        raise ValueError("a track needs a deezer_id or a local_path")
    values = (
        rec.deezer_id,
        rec.isrc,
        rec.title,
        rec.artist,
        rec.album,
        rec.cover_url,
        rec.duration,
        rec.source,
        rec.local_path,
    )
    with conn:
        existing = _find_id(conn, rec.deezer_id, rec.local_path)
        if existing is not None:
            conn.execute(
                "UPDATE tracks SET deezer_id=?, isrc=?, title=?, artist=?, album=?, cover_url=?,"
                " duration=?, source=?, local_path=? WHERE id=?",
                (*values, existing),
            )
            return existing
        cur = conn.execute(
            "INSERT INTO tracks(deezer_id, isrc, title, artist, album, cover_url, duration,"
            " source, local_path, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (*values, time.time()),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid


def _vector_blob(vector: np.ndarray, name: str) -> tuple[int, bytes]:
    arr = np.asarray(vector)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} vector must be 1-D and non-empty, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{name} vector must be floating point, got {arr.dtype}")
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} vector contains non-finite values")
    return int(arr.size), arr.tobytes()


def save_analysis(conn: sqlite3.Connection, track_id: int, analysis: Analysis) -> None:
    """Replace the track's features and vectors in a single transaction."""
    vibe_dim, vibe_blob = _vector_blob(analysis.vibe, "vibe")
    guitar: tuple[int, bytes] | None = None
    if analysis.guitar is not None:
        guitar = _vector_blob(analysis.guitar, "guitar")
        if analysis.guitar_model_version is None:
            raise ValueError("a guitar vector needs guitar_model_version")
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO features(track_id, tempo, key, mode, loudness_db,"
            " guitar_presence, model_version) VALUES (?,?,?,?,?,?,?)",
            (
                track_id,
                analysis.tempo,
                analysis.key,
                analysis.mode,
                analysis.loudness_db,
                analysis.guitar_presence,
                analysis.features_model_version,
            ),
        )
        conn.execute(
            "INSERT OR REPLACE INTO vectors(track_id, kind, dim, vector, model_version)"
            " VALUES (?, 'vibe', ?, ?, ?)",
            (track_id, vibe_dim, vibe_blob, analysis.vibe_model_version),
        )
        if guitar is not None:
            conn.execute(
                "INSERT OR REPLACE INTO vectors(track_id, kind, dim, vector, model_version)"
                " VALUES (?, 'guitar', ?, ?, ?)",
                (track_id, guitar[0], guitar[1], analysis.guitar_model_version),
            )
        else:
            conn.execute("DELETE FROM vectors WHERE track_id = ? AND kind = 'guitar'", (track_id,))


def is_finished(
    conn: sqlite3.Connection,
    *,
    deezer_id: int | None = None,
    local_path: str | None = None,
    vibe_model_version: str,
) -> bool:
    """True if the track has features and a vibe vector of the given model version."""
    if deezer_id is None and local_path is None:
        raise ValueError("pass deezer_id or local_path")
    track_id = _find_id(conn, deezer_id, local_path)
    if track_id is None:
        return False
    row = conn.execute(
        "SELECT 1 FROM features f JOIN vectors v ON v.track_id = f.track_id AND v.kind = 'vibe'"
        " WHERE f.track_id = ? AND v.model_version = ?",
        (track_id, vibe_model_version),
    ).fetchone()
    return row is not None


def record_failure(conn: sqlite3.Connection, key: str, reason: str) -> None:
    reason = scrub_urls(reason)[:MAX_REASON_CHARS]
    with conn:
        conn.execute(
            "INSERT INTO failures(key, reason, attempts, last_at) VALUES (?,?,1,?)"
            " ON CONFLICT(key) DO UPDATE SET reason = excluded.reason,"
            " attempts = attempts + 1, last_at = excluded.last_at",
            (key, reason, time.time()),
        )


def failed_keys(conn: sqlite3.Connection) -> set[str]:
    return {str(r["key"]) for r in conn.execute("SELECT key FROM failures")}


def clear_failure(conn: sqlite3.Connection, key: str) -> None:
    with conn:
        conn.execute("DELETE FROM failures WHERE key = ?", (key,))


def load_vectors(conn: sqlite3.Connection, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """All vectors of a kind: (int64 ids, float32 matrix of shape (n, dim))."""
    if kind not in VECTOR_KINDS:
        raise ValueError(f"unknown vector kind {kind!r}")
    rows = conn.execute(
        "SELECT track_id, dim, vector FROM vectors WHERE kind = ? ORDER BY track_id", (kind,)
    ).fetchall()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    dims = {int(r["dim"]) for r in rows}
    if len(dims) != 1:
        raise ValueError(f"{kind} vectors have mixed dimensions: {sorted(dims)}")
    ids = np.array([r["track_id"] for r in rows], dtype=np.int64)
    matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    return ids, np.ascontiguousarray(matrix)


def get_track(conn: sqlite3.Connection, track_id: int) -> sqlite3.Row | None:
    row: sqlite3.Row | None = conn.execute(
        "SELECT * FROM tracks WHERE id = ?", (track_id,)
    ).fetchone()
    return row


def count_tracks(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
