import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from sonder.config import REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import jamendo as jm  # noqa: E402


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_build", EVAL_DIR / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_build"] = module
    spec.loader.exec_module(module)
    return module


eb: Any = _load()

SR = eb.SAMPLE_RATE


def _track(track_id: int) -> jm.JamendoTrack:
    return jm.JamendoTrack(
        track_id=track_id,
        artist_id=1,
        album_id=1,
        path=f"{track_id % 100:02d}/{track_id}.mp3",
        duration=100.0,
        tags={"instrument": ["guitar"]},
    )


def _vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(8).astype(np.float32)


class FakeAnalyzer:
    def __init__(self, guitar: bool = True, fail_on: set[int] | None = None) -> None:
        self.guitar = guitar
        self.fail_on = fail_on or set()
        self.calls: list[tuple[int, int]] = []  # (full samples, clip samples)

    def __call__(self, full: np.ndarray, clip: np.ndarray) -> Any:
        self.calls.append((full.shape[1], clip.shape[1]))
        if int(round(float(full[0, 0]))) in self.fail_on:
            raise RuntimeError("boom")
        return eb.TrackAnalysis(_vec(1), _vec(2), _vec(3) if self.guitar else None, 0.37)


def _loader(seconds: float = 100.0) -> Any:
    def load(path: Path, sr: int, channels: int) -> np.ndarray:
        assert (sr, channels) == (SR, 2)
        track_id = int(path.name.split(".")[0])
        audio = np.zeros((2, int(seconds * sr)), dtype=np.float32)
        audio[:, 0] = track_id  # lets the fake analyzer identify the track
        return audio

    return load


# ---- middle_clip ------------------------------------------------------------------------


def test_middle_clip_is_centered_30s() -> None:
    n = 100 * SR
    audio = np.arange(n, dtype=np.float32)[None].repeat(2, axis=0)
    clip = eb.middle_clip(audio)
    assert clip.shape == (2, 30 * SR)
    start = (n - 30 * SR) // 2
    assert clip[0, 0] == start
    assert clip[0, -1] == start + 30 * SR - 1


def test_middle_clip_short_track_returned_whole() -> None:
    audio = np.zeros((2, 10 * SR), dtype=np.float32)
    assert eb.middle_clip(audio).shape == (2, 10 * SR)
    exact = np.zeros((2, 30 * SR), dtype=np.float32)
    assert eb.middle_clip(exact).shape == (2, 30 * SR)


# ---- build ------------------------------------------------------------------------------


def test_build_stores_three_vectors_and_presence(tmp_path: Path) -> None:
    conn = jm.connect(tmp_path / "e.sqlite")
    analyzer = FakeAnalyzer()
    summary = eb.build(conn, [_track(214)], analyzer, load=_loader())
    assert summary.built == 1
    assert analyzer.calls == [(100 * SR, 30 * SR)]
    for kind in jm.VECTOR_KINDS:
        ids, matrix = jm.load_matrix(conn, kind)
        assert ids.tolist() == [214] and matrix.shape == (1, 8)
    assert conn.execute("SELECT guitar_presence FROM presence").fetchone()[0] == pytest.approx(0.37)
    assert conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1


def test_build_silent_clip_stores_presence_without_guitar_vector(tmp_path: Path) -> None:
    conn = jm.connect(tmp_path / "e.sqlite")
    eb.build(conn, [_track(1)], FakeAnalyzer(guitar=False), load=_loader())
    assert jm.load_matrix(conn, "clip_guitar")[0].size == 0
    assert jm.load_matrix(conn, "full_vibe")[0].tolist() == [1]


def test_build_resume_skips_finished_tracks(tmp_path: Path) -> None:
    conn = jm.connect(tmp_path / "e.sqlite")
    tracks = [_track(1), _track(2)]
    eb.build(conn, tracks[:1], FakeAnalyzer(), load=_loader())
    analyzer = FakeAnalyzer()
    summary = eb.build(conn, tracks, analyzer, load=_loader())
    assert (summary.built, summary.skipped) == (1, 1)
    assert len(analyzer.calls) == 1


def test_build_records_failure_and_continues(tmp_path: Path) -> None:
    conn = jm.connect(tmp_path / "e.sqlite")
    summary = eb.build(
        conn, [_track(1), _track(2), _track(3)], FakeAnalyzer(fail_on={2}), load=_loader()
    )
    assert (summary.built, summary.failed) == (2, 1)
    assert jm.failed_ids(conn) == {2}
    assert jm.load_matrix(conn, "full_vibe")[0].tolist() == [1, 3]


def test_build_skips_failed_unless_retry(tmp_path: Path) -> None:
    conn = jm.connect(tmp_path / "e.sqlite")
    eb.build(conn, [_track(2)], FakeAnalyzer(fail_on={2}), load=_loader())
    again = eb.build(conn, [_track(2)], FakeAnalyzer(), load=_loader())
    assert (again.built, again.already_failed) == (0, 1)
    retried = eb.build(conn, [_track(2)], FakeAnalyzer(), load=_loader(), retry_failed=True)
    assert retried.built == 1
    assert jm.failed_ids(conn) == set()


def test_build_loader_error_is_a_failure(tmp_path: Path) -> None:
    conn = jm.connect(tmp_path / "e.sqlite")

    def bad_load(path: Path, sr: int, channels: int) -> np.ndarray:
        raise OSError("cannot decode")

    summary = eb.build(conn, [_track(7)], FakeAnalyzer(), load=bad_load)
    assert summary.failed == 1
    row = conn.execute("SELECT reason FROM failures").fetchone()
    assert "cannot decode" in row["reason"]


def test_version_change_triggers_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = jm.connect(tmp_path / "e.sqlite")
    eb.build(conn, [_track(1)], FakeAnalyzer(), load=_loader())
    monkeypatch.setattr(eb, "VERSIONS", {**eb.VERSIONS, "full_vibe": "clap-v2"})
    summary = eb.build(conn, [_track(1)], FakeAnalyzer(), load=_loader())
    assert summary.built == 1


# ---- import hygiene ---------------------------------------------------------------------


@pytest.mark.parametrize("script", ["build.py", "fetch_jamendo.py", "jamendo.py"])
def test_eval_modules_do_not_import_ml_libs(script: str) -> None:
    code = (
        "import importlib.util, sys\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "spec = importlib.util.spec_from_file_location('s', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "bad = [n for n in ('torch', 'transformers', 'demucs_infer') if n in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(EVAL_DIR / script), str(EVAL_DIR)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
