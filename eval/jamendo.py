"""MTG-Jamendo evaluation set: dataset parsing and the eval embedding store.

No ML imports here. Paths come from settings, so SONDER_DATA_DIR moves everything.
"""

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sonder.config import get_settings

CATEGORIES = ("genre", "mood/theme", "instrument")
GUITAR_TAGS = frozenset(
    {
        "instrument---guitar",
        "instrument---electricguitar",
        "instrument---acousticguitar",
        "instrument---classicalguitar",
    }
)
VECTOR_KINDS = ("full_vibe", "clip_vibe", "clip_guitar")
ARCHIVE_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/{name}"
# autotagging.tsv is a symlink to this file; on Windows it checks out as a text stub.
TAGS_TSV = "raw_30s_cleantags_50artists.tsv"
SHA_LIST = "raw_30s_audio-low_sha256_tars.txt"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    track_id INTEGER PRIMARY KEY,
    artist_id INTEGER NOT NULL,
    album_id INTEGER NOT NULL,
    duration REAL NOT NULL,
    tags_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vectors (
    track_id INTEGER NOT NULL REFERENCES tracks(track_id),
    kind TEXT NOT NULL CHECK (kind IN ('full_vibe', 'clip_vibe', 'clip_guitar')),
    model_version TEXT NOT NULL,
    dim INTEGER NOT NULL,
    blob BLOB NOT NULL,
    PRIMARY KEY (track_id, kind)
);
CREATE TABLE IF NOT EXISTS presence (
    track_id INTEGER PRIMARY KEY REFERENCES tracks(track_id),
    guitar_presence REAL NOT NULL,
    model_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS failures (
    track_id INTEGER PRIMARY KEY,
    reason TEXT NOT NULL
);
"""


# ---- paths ------------------------------------------------------------------------------


def jamendo_dir() -> Path:
    return get_settings().data_dir / "jamendo"


def repo_dir() -> Path:
    return get_settings().data_dir / "mtg-jamendo-dataset"


def audio_dir() -> Path:
    return jamendo_dir() / "audio-low"


def eval_db_path() -> Path:
    return jamendo_dir() / "eval.sqlite"


def tags_tsv() -> Path:
    return repo_dir() / "data" / TAGS_TSV


def sha_list() -> Path:
    return repo_dir() / "data" / "download" / SHA_LIST


# ---- dataset ----------------------------------------------------------------------------


@dataclass(frozen=True)
class JamendoTrack:
    track_id: int
    artist_id: int
    album_id: int
    path: str  # as in the TSV, e.g. "14/214.mp3"
    duration: float
    tags: dict[str, list[str]] = field(default_factory=dict)  # category -> tag names

    @property
    def all_tags(self) -> set[str]:
        return {f"{cat}---{t}" for cat, names in self.tags.items() for t in names}


def parse_sha_list(path: Path) -> list[tuple[str, str]]:
    """[(archive_name, sha256)] in file order. Lines are '<sha256> <name>.tar' (CRLF)."""
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, name = line.partition(" ")
        name = name.strip()
        if not sha or not name:
            raise ValueError(f"bad sha list line: {line!r}")
        out.append((name, sha.lower()))
    return out


def _numeric_id(value: str) -> int:
    return int(value.rsplit("_", 1)[-1])


def load_tracks(tsv: Path) -> list[JamendoTrack]:
    """Parse a Jamendo tags TSV. Tags are trailing tab-separated 'category---tag' fields."""
    lines = tsv.read_text(encoding="utf-8").splitlines()
    tracks: list[JamendoTrack] = []
    for line in lines[1:]:  # header
        if not line.strip():
            continue
        cols = line.rstrip("\r\n").split("\t")
        if len(cols) < 5:
            raise ValueError(f"bad TSV row: {line!r}")
        tags: dict[str, list[str]] = {}
        for raw in cols[5:]:
            cat, sep, name = raw.partition("---")
            if not sep or not name:
                continue
            tags.setdefault(cat, []).append(name)
        tracks.append(
            JamendoTrack(
                track_id=_numeric_id(cols[0]),
                artist_id=_numeric_id(cols[1]),
                album_id=_numeric_id(cols[2]),
                path=cols[3],
                duration=float(cols[4]),
                tags=tags,
            )
        )
    return tracks


def has_guitar(track: JamendoTrack) -> bool:
    return bool(track.all_tags & GUITAR_TAGS)


def member_name(track_path: str) -> str:
    """Name inside the audio-low archives: '14/214.mp3' -> '14/214.low.mp3'."""
    stem, dot, ext = track_path.rpartition(".")
    if not dot:
        raise ValueError(f"path has no extension: {track_path!r}")
    return f"{stem}.low.{ext}"


def audio_path(track: JamendoTrack, root: Path | None = None) -> Path:
    return (root or audio_dir()) / member_name(track.path)


def available_tracks(
    tracks: Iterable[JamendoTrack], root: Path | None = None
) -> list[JamendoTrack]:
    return [t for t in tracks if audio_path(t, root).is_file()]


# ---- eval store -------------------------------------------------------------------------


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or eval_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _blob(vector: np.ndarray) -> tuple[int, bytes]:
    v = np.asarray(vector, dtype=np.float32)
    if v.ndim != 1 or not np.all(np.isfinite(v)):
        raise ValueError("vector must be 1-D and finite")
    return v.shape[0], v.tobytes()


def save_track_result(
    conn: sqlite3.Connection,
    track: JamendoTrack,
    vectors: dict[str, np.ndarray],
    versions: dict[str, str],
    guitar_presence: float,
) -> None:
    """Store one track atomically. `vectors` may omit clip_guitar (silent clip)."""
    unknown = set(vectors) - set(VECTOR_KINDS)
    if unknown:
        raise ValueError(f"unknown vector kinds: {sorted(unknown)}")
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO tracks VALUES (?, ?, ?, ?, ?)",
            (
                track.track_id,
                track.artist_id,
                track.album_id,
                track.duration,
                json.dumps(track.tags, sort_keys=True),
            ),
        )
        conn.execute("DELETE FROM vectors WHERE track_id = ?", (track.track_id,))
        for kind, vec in vectors.items():
            dim, blob = _blob(vec)
            conn.execute(
                "INSERT INTO vectors VALUES (?, ?, ?, ?, ?)",
                (track.track_id, kind, versions[kind], dim, blob),
            )
        conn.execute(
            "INSERT OR REPLACE INTO presence VALUES (?, ?, ?)",
            (track.track_id, guitar_presence, versions["clip_guitar"]),
        )
        conn.execute("DELETE FROM failures WHERE track_id = ?", (track.track_id,))


def is_done(conn: sqlite3.Connection, track_id: int, versions: dict[str, str]) -> bool:
    """True when the track has full_vibe and clip_vibe at the current versions and a presence
    row at the current guitar version."""
    rows = conn.execute(
        "SELECT kind, model_version FROM vectors WHERE track_id = ?", (track_id,)
    ).fetchall()
    have = {r["kind"]: r["model_version"] for r in rows}
    for kind in ("full_vibe", "clip_vibe"):
        if have.get(kind) != versions[kind]:
            return False
    pres = conn.execute(
        "SELECT model_version FROM presence WHERE track_id = ?", (track_id,)
    ).fetchone()
    if pres is None or pres["model_version"] != versions["clip_guitar"]:
        return False
    return "clip_guitar" not in have or have["clip_guitar"] == versions["clip_guitar"]


def record_failure(conn: sqlite3.Connection, track_id: int, reason: str) -> None:
    with conn:
        conn.execute("INSERT OR REPLACE INTO failures VALUES (?, ?)", (track_id, reason[:500]))


def failed_ids(conn: sqlite3.Connection) -> set[int]:
    return {r["track_id"] for r in conn.execute("SELECT track_id FROM failures")}


def load_matrix(conn: sqlite3.Connection, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """(track_ids int64 (N,), vectors float32 (N, D)) for one kind, ordered by track id."""
    if kind not in VECTOR_KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    rows = conn.execute(
        "SELECT track_id, dim, blob FROM vectors WHERE kind = ? ORDER BY track_id", (kind,)
    ).fetchall()
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    ids = np.array([r["track_id"] for r in rows], dtype=np.int64)
    matrix = np.stack([np.frombuffer(r["blob"], dtype=np.float32, count=r["dim"]) for r in rows])
    return ids, matrix
