import argparse
import importlib.util
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from sonder.audio.clap import MAX_TRACK_WINDOWS
from sonder.audio.clap import MODEL_VERSION as VIBE_VERSION
from sonder.config import REPO_ROOT
from sonder.index import db
from sonder.index.db import Analysis
from sonder.index.search import VectorIndex
from sonder.sources.deezer import DeezerError, DeezerTrack
from sonder.sources.itunes import ItunesTrack


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ingest_script", REPO_ROOT / "scripts" / "ingest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ingest_script"] = module
    spec.loader.exec_module(module)
    return module


ing: Any = _load()

# Deliberately looks like a real Deezer preview URL (ground rule 2 must keep it out of the DB).
DEEZER_URL = "https://cdnt-preview.dzcdn.net/stream/c-abc123-1.mp3?hdnea=exp=1700000000~hmac=ff"
ITUNES_URL = "https://audio-ssl.itunes.apple.com/clip/preview-1.m4a"


def dz(id: int, artist: str = "Radiohead", title: str = "Creep", bpm: float | None = None) -> Any:
    return DeezerTrack(
        id=id,
        title=title,
        title_short=title,
        artist=artist,
        album_title="Pablo Honey",
        cover_url="https://img.example/c.jpg",
        duration=238,
        isrc="GB1",
        bpm=bpm,
        rank=0,
    )


class FakeDeezer:
    def __init__(self, tracks: list[Any], previews: dict[int, str | None]) -> None:
        self.tracks = {t.id: t for t in tracks}
        self.previews = previews
        self.get_calls = 0
        self.preview_calls = 0

    def get_track(self, track_id: int) -> Any:
        self.get_calls += 1
        if track_id == 666:
            raise DeezerError("Exception", "quota boom", 4)
        return self.tracks.get(track_id)

    def fresh_preview_url(self, track_id: int) -> str | None:
        self.preview_calls += 1
        return self.previews.get(track_id)


class FakeItunes:
    def __init__(self, tracks: list[ItunesTrack] | None = None) -> None:
        self.tracks = tracks or []
        self.calls = 0

    def search_song(
        self, artist: str, title: str, limit: int = 10, country: str = "US"
    ) -> list[ItunesTrack]:
        self.calls += 1
        return self.tracks


def itunes_track(artist: str = "Radiohead", title: str = "Creep") -> ItunesTrack:
    return ItunesTrack(1, title, artist, "", ITUNES_URL, 1000, None)


class Env:
    """A pipeline wired to fakes, with call counters."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_dir = tmp_path / "tmp"
        self.db_path = tmp_path / "index.sqlite"
        self.conn = db.connect(self.db_path)
        self.deezer = FakeDeezer([dz(1), dz(2, title="Other")], {1: DEEZER_URL, 2: DEEZER_URL})
        self.itunes = FakeItunes()
        self.downloads: list[str] = []
        self.status = 200
        self.analyze_calls: list[tuple[Any, Any, Any, Any]] = []
        self.loaded_suffixes: list[str] = []
        self.fail_analyze = False
        self.fail_load = False
        self.guitar = True
        self.http = httpx.Client(transport=httpx.MockTransport(self._handler))
        self.pipeline = self.make()

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.downloads.append(str(request.url.host))
        return httpx.Response(self.status, content=b"fake-audio-bytes")

    def _analyze(self, stereo: Any, mono: Any, max_windows: Any, bpm: Any) -> Analysis:
        self.analyze_calls.append((stereo, mono, max_windows, bpm))
        if self.fail_analyze:
            raise RuntimeError(f"analysis blew up on {DEEZER_URL}")
        rng = np.random.default_rng(len(self.analyze_calls))
        return Analysis(
            tempo=bpm or 100.0,
            key="A",
            mode="minor",
            loudness_db=-12.0,
            guitar_presence=0.5,
            vibe=rng.standard_normal(512).astype(np.float32),
            guitar=rng.standard_normal(512).astype(np.float32) if self.guitar else None,
            guitar_model_version="g1" if self.guitar else None,
        )

    def _load(self, path: Path, sr: int, channels: int) -> np.ndarray:
        self.loaded_suffixes.append(path.suffix)
        assert path.exists() and path.parent == self.tmp_dir
        if self.fail_load:
            raise RuntimeError("decode failed")
        n = 2 * sr
        return np.zeros((2, n), dtype=np.float32) if channels == 2 else np.zeros(n, np.float32)

    def make(self, retry_failed: bool = False) -> Any:
        return ing.Pipeline(
            conn=self.conn,
            deezer=self.deezer,
            itunes=self.itunes,
            http=self.http,
            analyze=self._analyze,
            tmp_dir=self.tmp_dir,
            retry_failed=retry_failed,
            load=self._load,
        )

    def tmp_files(self) -> list[Path]:
        return list(self.tmp_dir.iterdir()) if self.tmp_dir.exists() else []

    def failures(self) -> dict[str, tuple[str, int]]:
        rows = self.conn.execute("SELECT * FROM failures")
        return {r["key"]: (r["reason"], r["attempts"]) for r in rows}


@pytest.fixture
def env(tmp_path: Path) -> Env:
    e = Env(tmp_path)
    yield e  # type: ignore[misc]
    e.conn.close()


# ---- classify / read_items -------------------------------------------------------------


@pytest.mark.parametrize(
    ("item", "kind"),
    [
        ("3135556", "deezer"),
        (" 42 ", "deezer"),
        ("C:\\music\\a.mp3", "local"),
        ("C:\\music\\2023\\123", "local"),
        ("/home/me/123.mp3", "local"),
        ("123.mp3", "local"),
        ("12a", "local"),
        ("track 1", "local"),
        ("\u0663\u0664", "local"),  # Arabic-Indic digits are not a Deezer id
    ],
)
def test_classify(item: str, kind: str) -> None:
    assert ing.classify(item) == kind


def test_read_items_dedupes_and_skips_comments(tmp_path: Path) -> None:
    f = tmp_path / "ids.txt"
    f.write_text("# header\n\n  1  \n2\n# 3\n1\nC:\\m\\a.mp3  \n  \n", encoding="utf-8")
    args = argparse.Namespace(items=["2", "9", " 9 "], from_file=f)
    assert ing.read_items(args) == ["2", "9", "1", "C:\\m\\a.mp3"]
    assert ing.read_items(argparse.Namespace(items=[], from_file=None)) == []


def test_parse_filename() -> None:
    assert ing.parse_filename(Path("Artist - Title.mp3")) == ("Artist", "Title")
    assert ing.parse_filename(Path("A - B - C.wav")) == ("A", "B - C")
    assert ing.parse_filename(Path("just a song.flac")) == ("Unknown", "just a song")
    assert ing.parse_filename(Path(" - x.mp3")) == ("Unknown", " - x")


def test_main_usage_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as e:
        ing.main([])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e2:
        ing.main(["--from-file", str(tmp_path / "missing.txt")])
    assert e2.value.code == 2


# ---- Deezer ingest ---------------------------------------------------------------------


def test_new_deezer_track_is_stored_and_second_run_skips(env: Env) -> None:
    s1 = env.pipeline.run(["1"])
    assert (s1.ingested, s1.skipped, s1.failed) == (1, 0, 0)
    assert len(env.downloads) == 1 and len(env.analyze_calls) == 1
    assert env.loaded_suffixes == [".mp3", ".mp3"]

    row = env.conn.execute("SELECT * FROM tracks WHERE deezer_id = 1").fetchone()
    assert (row["title"], row["artist"], row["album"], row["source"]) == (
        "Creep",
        "Radiohead",
        "Pablo Honey",
        "deezer",
    )
    assert row["duration"] == 238.0 and row["isrc"] == "GB1" and row["local_path"] is None
    feat = env.conn.execute("SELECT * FROM features").fetchone()
    assert feat["model_version"] == db.FEATURES_VERSION
    kinds = {
        r["kind"]: r["model_version"]
        for r in env.conn.execute("SELECT kind, model_version FROM vectors")
    }
    assert kinds == {"vibe": VIBE_VERSION, "guitar": "g1"}

    s2 = env.pipeline.run(["1"])
    assert (s2.ingested, s2.skipped, s2.failed) == (0, 1, 0)
    assert len(env.downloads) == 1 and len(env.analyze_calls) == 1
    assert env.deezer.get_calls == 1 and env.deezer.preview_calls == 1


def test_analysis_inputs_for_previews(env: Env) -> None:
    env.deezer = FakeDeezer([dz(1, bpm=93.0)], {1: DEEZER_URL})
    p = env.make()
    p.run(["1"])
    stereo, mono, max_windows, bpm = env.analyze_calls[0]
    assert stereo.shape[0] == 2 and mono.ndim == 1
    assert max_windows is None and bpm == 93.0
    feat = env.conn.execute("SELECT tempo FROM features").fetchone()
    assert feat["tempo"] == 93.0


def test_temp_dir_empty_after_success(env: Env) -> None:
    env.pipeline.run(["1", "2"])
    assert env.tmp_files() == []


@pytest.mark.parametrize("failure", ["analyze", "load", "http"])
def test_temp_dir_empty_after_failure(env: Env, failure: str) -> None:
    if failure == "analyze":
        env.fail_analyze = True
    elif failure == "load":
        env.fail_load = True
    else:
        env.status = 403
    summary = env.pipeline.run(["1"])
    assert summary.failed == 1 and summary.ingested == 0
    assert env.tmp_files() == []
    assert db.count_tracks(env.conn) == 0  # nothing half-stored
    reason, attempts = env.failures()["1"]
    assert attempts == 1
    assert "dzcdn" not in reason and "hdnea" not in reason and "http" not in reason


def test_db_never_contains_preview_url(env: Env) -> None:
    assert "hdnea=exp=" in DEEZER_URL and "dzcdn" in DEEZER_URL
    env.pipeline.run(["1", "2"])
    env.fail_analyze = True
    env.deezer.tracks[3] = dz(3, title="Third")
    env.deezer.previews[3] = DEEZER_URL
    env.pipeline.run(["3"])  # failure text mentions the URL; must be scrubbed
    env.conn.close()
    blob = b"".join(p.read_bytes() for p in env.db_path.parent.glob(env.db_path.name + "*"))
    assert len(blob) > 1000
    for needle in (b"preview", b"hdnea", b"dzcdn", b"hmac"):
        assert needle not in blob
    env.conn = db.connect(env.db_path)  # fixture closes it


@pytest.fixture
def restore_transport_loggers() -> Any:
    names = ("httpx", "httpcore")
    saved = {n: logging.getLogger(n).level for n in names}
    yield
    for n, level in saved.items():
        logging.getLogger(n).setLevel(level)


def test_logs_never_contain_the_preview_url(
    env: Env, caplog: pytest.LogCaptureFixture, restore_transport_loggers: None
) -> None:
    caplog.set_level(logging.INFO)
    # Precondition: with httpx at INFO the signed URL really is logged for every request,
    # so the assertion below is not vacuous.
    logging.getLogger("httpx").setLevel(logging.INFO)
    env.pipeline.run(["1"])
    assert "hdnea=exp=" in caplog.text

    caplog.clear()
    ing.configure_logging()
    env.fail_analyze = True  # failure text mentions the URL too; the log line must be scrubbed
    env.pipeline.run(["2"])
    assert "failed 2" in caplog.text
    for needle in ("hdnea", "dzcdn", "hmac", "preview"):
        assert needle not in caplog.text


def test_unresolvable_path_fails_that_item_only(env: Env) -> None:
    summary = env.pipeline.run(["bad\0path.mp3", "1"])
    assert (summary.ingested, summary.failed) == (1, 1)
    assert len(env.failures()) == 1


def test_itunes_fallback_sets_source(env: Env) -> None:
    env.deezer.previews[1] = None
    env.itunes.tracks = [itunes_track()]
    env.pipeline.run(["1"])
    row = env.conn.execute("SELECT * FROM tracks").fetchone()
    assert (row["source"], row["deezer_id"]) == ("itunes", 1)
    assert env.downloads == ["audio-ssl.itunes.apple.com"]
    assert env.loaded_suffixes[0] == ".m4a"
    assert env.itunes.calls == 1


def test_itunes_weak_match_is_rejected(env: Env) -> None:
    env.deezer.previews[1] = None
    env.itunes.tracks = [itunes_track("Somebody Else", "Completely Different")]
    summary = env.pipeline.run(["1"])
    assert summary.failed == 1
    assert env.failures()["1"][0] == "no preview"


def test_no_preview_failure_and_batch_continues(env: Env) -> None:
    env.deezer.previews[1] = None
    summary = env.pipeline.run(["1", "999", "666", "2"])
    assert (summary.ingested, summary.failed) == (1, 3)
    fails = env.failures()
    assert fails["1"][0] == "no preview"
    assert fails["999"][0] == "track not found"
    assert fails["666"][0].startswith("DeezerError: Deezer error 4")
    assert db.count_tracks(env.conn) == 1
    assert "failed 3" in summary.format()


def test_failed_items_skipped_unless_retry(env: Env) -> None:
    env.deezer.previews[1] = None
    env.pipeline.run(["1"])
    env.deezer.previews[1] = DEEZER_URL  # it works now

    s = env.pipeline.run(["1"])
    assert (s.already_failed, s.ingested, s.failed) == (1, 0, 0)
    assert env.deezer.get_calls == 1

    s = env.make(retry_failed=True).run(["1"])
    assert (s.ingested, s.already_failed) == (1, 0)
    assert env.failures() == {}  # cleared on success


def test_repeated_failure_increments_attempts(env: Env) -> None:
    env.deezer.previews[1] = None
    env.pipeline.run(["1"])
    env.make(retry_failed=True).run(["1"])
    assert env.failures()["1"] == ("no preview", 2)


def test_reingest_after_model_version_change(env: Env) -> None:
    env.pipeline.run(["1"])
    row = env.conn.execute("SELECT track_id FROM vectors WHERE kind='vibe'").fetchone()
    with env.conn:
        env.conn.execute("UPDATE vectors SET model_version = 'old' WHERE kind = 'vibe'")
    s = env.pipeline.run(["1"])
    assert s.ingested == 1
    assert env.conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1
    assert env.conn.execute("SELECT track_id FROM vectors WHERE kind='vibe'").fetchone() == row


def test_no_guitar_vector_when_analysis_has_none(env: Env) -> None:
    env.guitar = False
    env.pipeline.run(["1"])
    assert env.conn.execute("SELECT COUNT(*) FROM vectors WHERE kind='guitar'").fetchone()[0] == 0
    assert env.conn.execute("SELECT COUNT(*) FROM vectors WHERE kind='vibe'").fetchone()[0] == 1


# ---- local files -----------------------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def write_wav(path: Path, seconds: float = 2.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = int(seconds * 44_100)
    t = np.arange(n) / 44_100
    left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(n)
    right = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.02 * rng.standard_normal(n)
    sf.write(path, np.stack([left, right], axis=1).astype(np.float32), 44_100)


@needs_ffmpeg
def test_local_file_ingest_with_real_decoder(env: Env, tmp_path: Path) -> None:
    music = tmp_path / "music"
    music.mkdir()
    wav = music / "Some Band - Some Song.wav"
    write_wav(wav)
    p = ing.Pipeline(
        conn=env.conn,
        deezer=env.deezer,
        itunes=env.itunes,
        http=env.http,
        analyze=env._analyze,
        tmp_dir=env.tmp_dir,
    )
    s = p.run([str(wav), str(wav)])  # duplicates within a run are skipped by is_finished
    assert (s.ingested, s.skipped, s.failed) == (1, 1, 0)
    stereo, mono, max_windows, bpm = env.analyze_calls[0]
    assert stereo.shape[0] == 2 and abs(stereo.shape[1] - 2 * 44_100) < 100
    assert abs(mono.size - 2 * 22_050) < 100
    assert max_windows == MAX_TRACK_WINDOWS and bpm is None
    row = env.conn.execute("SELECT * FROM tracks").fetchone()
    assert (row["artist"], row["title"], row["source"]) == ("Some Band", "Some Song", "local")
    assert row["local_path"] == str(wav.resolve()) and row["deezer_id"] is None
    assert row["duration"] == pytest.approx(2.0, abs=0.05)
    assert env.downloads == [] and env.deezer.get_calls == 0
    assert db.is_finished(env.conn, local_path=str(wav.resolve()), vibe_model_version=VIBE_VERSION)


@needs_ffmpeg
def test_local_file_without_dash_and_failures(env: Env, tmp_path: Path) -> None:
    wav = tmp_path / "plain.wav"
    write_wav(wav, 1.0)
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"not audio at all")
    missing = tmp_path / "gone.mp3"
    p = ing.Pipeline(
        conn=env.conn,
        deezer=env.deezer,
        itunes=env.itunes,
        http=env.http,
        analyze=env._analyze,
        tmp_dir=env.tmp_dir,
    )
    s = p.run([str(bad), str(missing), str(wav)])
    assert (s.ingested, s.failed) == (1, 2)
    row = env.conn.execute("SELECT * FROM tracks").fetchone()
    assert (row["artist"], row["title"]) == ("Unknown", "plain")
    fails = env.failures()
    assert fails[str(missing.resolve())][0] == "file not found"
    assert fails[str(bad.resolve())][0].startswith("AudioDecodeError")
    # failed local files are skipped on the next run
    assert p.run([str(bad)]).already_failed == 1


# ---- import hygiene --------------------------------------------------------------------


def test_script_import_does_not_load_ml_libs() -> None:
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('s', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "sys.modules['s'] = m\n"
        "spec.loader.exec_module(m)\n"
        "bad = [x for x in ('torch', 'transformers', 'demucs_infer') if x in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(REPO_ROOT / "scripts" / "ingest.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# ---- slow: real models -----------------------------------------------------------------

WEIGHTS_DIR = Path.home() / ".cache" / "demucs-infer"


def _clap_cached() -> bool:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        hub = Path(HF_HUB_CACHE)
    except ImportError:
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return (hub / "models--laion--larger_clap_music").is_dir()


@pytest.mark.slow
def test_real_local_ingest_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("demucs_infer")
    pytest.importorskip("transformers")
    if not WEIGHTS_DIR.is_dir() or not any(WEIGHTS_DIR.rglob("*")):
        pytest.skip(f"htdemucs_6s weights not found in {WEIGHTS_DIR}")
    if not _clap_cached():
        pytest.skip("CLAP model laion/larger_clap_music is not in the Hugging Face cache")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not on PATH")

    first = tmp_path / "Test Artist - Test Song.wav"
    second = tmp_path / "Other Artist - Other Song.wav"
    write_wav(first, 12.0, seed=1)
    write_wav(second, 6.0, seed=2)

    conn = db.connect(tmp_path / "index.sqlite")
    try:

        def pipeline(threshold: float) -> Any:
            return ing.Pipeline(
                conn=conn,
                deezer=FakeDeezer([], {}),
                itunes=FakeItunes(),
                http=httpx.Client(),
                analyze=ing.make_analyzer(guitar_threshold=threshold),
                tmp_dir=tmp_path / "tmp",
            )

        s = pipeline(10.0).run([str(first)])  # threshold far above any presence
        assert (s.ingested, s.failed) == (1, 0)
        s = pipeline(0.0).run([str(second)])
        assert (s.ingested, s.failed) == (1, 0)

        first_id = conn.execute(
            "SELECT id FROM tracks WHERE local_path = ?", (str(first.resolve()),)
        ).fetchone()["id"]
        second_id = conn.execute(
            "SELECT id FROM tracks WHERE local_path = ?", (str(second.resolve()),)
        ).fetchone()["id"]

        ids, vibe = db.load_vectors(conn, "vibe")
        assert vibe.shape == (2, 512)
        query = vibe[list(ids).index(first_id)]
        index = VectorIndex.load(conn)
        top = index.top_k(query, "vibe", k=2)
        assert top[0][0] == first_id
        assert top[0][1] == pytest.approx(1.0, abs=1e-4)

        feat = conn.execute("SELECT * FROM features WHERE track_id = ?", (first_id,)).fetchone()
        assert feat["model_version"] == db.FEATURES_VERSION
        assert 0.0 <= feat["guitar_presence"] <= 2.0

        gids, guitar = db.load_vectors(conn, "guitar")
        assert first_id not in gids.tolist()  # threshold forced high: no guitar vector
        assert gids.tolist() == [second_id] and guitar.shape == (1, 512)
        assert index.size("guitar") == 1
    finally:
        conn.close()


def test_sqlite_is_stdlib() -> None:
    assert sqlite3.sqlite_version_info >= (3, 24)  # ON CONFLICT ... DO UPDATE (upsert)
