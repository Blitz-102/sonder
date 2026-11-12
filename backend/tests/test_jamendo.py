import hashlib
import importlib.util
import io
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import numpy as np
import pytest
import respx

from sonder.config import REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import jamendo as jm  # noqa: E402


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"eval_{name}", EVAL_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"eval_{name}"] = module
    spec.loader.exec_module(module)
    return module


fj: Any = _load_script("fetch_jamendo")

TSV = (
    "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\r\n"
    "track_0000214\tartist_000014\talbum_000031\t14/214.mp3\t124.6\tgenre---punkrock\r\n"
    "track_0002966\tartist_000441\talbum_000460\t66/2966.mp3\t198.8\tgenre---chanson\t"
    "genre---popfunk\tinstrument---electricguitar\tmood/theme---happy\r\n"
    "track_0000005\tartist_000001\talbum_000002\t05/5.mp3\t60.0\r\n"
)


def _tsv(tmp_path: Path) -> list[jm.JamendoTrack]:
    p = tmp_path / "tags.tsv"
    p.write_bytes(TSV.encode())
    return jm.load_tracks(p)


# ---- parsing ----------------------------------------------------------------------------


def test_parse_sha_list_handles_crlf_and_order(tmp_path: Path) -> None:
    p = tmp_path / "sha.txt"
    p.write_bytes(b"AAA1 raw_30s_audio-low-00.tar\r\nbbb2 raw_30s_audio-low-01.tar\r\n\r\n")
    assert jm.parse_sha_list(p) == [
        ("raw_30s_audio-low-00.tar", "aaa1"),
        ("raw_30s_audio-low-01.tar", "bbb2"),
    ]


def test_parse_sha_list_rejects_malformed_line(tmp_path: Path) -> None:
    p = tmp_path / "sha.txt"
    p.write_text("justonetoken\n")
    with pytest.raises(ValueError):
        jm.parse_sha_list(p)


def test_load_tracks_groups_tags_and_ids(tmp_path: Path) -> None:
    tracks = _tsv(tmp_path)
    assert [t.track_id for t in tracks] == [214, 2966, 5]
    t = tracks[1]
    assert (t.artist_id, t.album_id, t.duration) == (441, 460, 198.8)
    assert t.tags == {
        "genre": ["chanson", "popfunk"],
        "instrument": ["electricguitar"],
        "mood/theme": ["happy"],
    }
    assert tracks[2].tags == {}


def test_has_guitar(tmp_path: Path) -> None:
    tracks = _tsv(tmp_path)
    assert [jm.has_guitar(t) for t in tracks] == [False, True, False]


def test_guitar_tags_exclude_bass_guitar() -> None:
    assert "instrument---acousticbassguitar" not in jm.GUITAR_TAGS


def test_member_name_and_audio_path(tmp_path: Path) -> None:
    assert jm.member_name("14/214.mp3") == "14/214.low.mp3"
    track = _tsv(tmp_path)[0]
    assert jm.audio_path(track, tmp_path) == tmp_path / "14" / "214.low.mp3"


def test_available_tracks_only_those_on_disk(tmp_path: Path) -> None:
    tracks = _tsv(tmp_path)
    f = tmp_path / "66" / "2966.low.mp3"
    f.parent.mkdir()
    f.write_bytes(b"x")
    assert jm.available_tracks(tracks, tmp_path) == [tracks[1]]


# ---- store ------------------------------------------------------------------------------

V = {"full_vibe": "clap", "clip_vibe": "clap", "clip_guitar": "demucs+clap"}


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def test_store_roundtrip_and_is_done(tmp_path: Path) -> None:
    track = _tsv(tmp_path)[1]
    conn = jm.connect(tmp_path / "eval.sqlite")
    assert not jm.is_done(conn, track.track_id, V)
    vecs = {"full_vibe": _vec(1), "clip_vibe": _vec(2), "clip_guitar": _vec(3)}
    jm.save_track_result(conn, track, vecs, V, 0.42)
    assert jm.is_done(conn, track.track_id, V)
    assert not jm.is_done(conn, track.track_id, {**V, "full_vibe": "clap-v2"})
    ids, matrix = jm.load_matrix(conn, "clip_guitar")
    assert ids.tolist() == [2966]
    np.testing.assert_array_equal(matrix[0], vecs["clip_guitar"])
    row = conn.execute("SELECT guitar_presence FROM presence").fetchone()
    assert row["guitar_presence"] == pytest.approx(0.42)


def test_store_allows_missing_guitar_vector(tmp_path: Path) -> None:
    track = _tsv(tmp_path)[0]
    conn = jm.connect(tmp_path / "eval.sqlite")
    jm.save_track_result(conn, track, {"full_vibe": _vec(1), "clip_vibe": _vec(2)}, V, 0.0)
    assert jm.is_done(conn, track.track_id, V)
    assert jm.load_matrix(conn, "clip_guitar")[0].size == 0


def test_store_rejects_bad_vectors(tmp_path: Path) -> None:
    track = _tsv(tmp_path)[0]
    conn = jm.connect(tmp_path / "eval.sqlite")
    bad = np.array([1.0, np.nan], dtype=np.float32)
    with pytest.raises(ValueError):
        jm.save_track_result(conn, track, {"full_vibe": bad, "clip_vibe": _vec(1)}, V, 0.1)
    with pytest.raises(ValueError):
        jm.save_track_result(conn, track, {"nope": _vec(1)}, V, 0.1)
    assert conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 0


def test_failure_recorded_then_cleared_on_success(tmp_path: Path) -> None:
    track = _tsv(tmp_path)[0]
    conn = jm.connect(tmp_path / "eval.sqlite")
    jm.record_failure(conn, track.track_id, "boom")
    assert jm.failed_ids(conn) == {214}
    jm.save_track_result(conn, track, {"full_vibe": _vec(1), "clip_vibe": _vec(2)}, V, 0.1)
    assert jm.failed_ids(conn) == set()


def test_load_matrix_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        jm.load_matrix(jm.connect(tmp_path / "e.sqlite"), "bogus")


# ---- fetch ------------------------------------------------------------------------------


def _tar_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


BASE = "https://cdn.example.test/{name}"


def _fetch(client: httpx.Client, name: str, sha: str, tmp_path: Path, **kw: Any) -> str:
    return fj.fetch_archive(
        client,
        name,
        sha,
        tar_dir=tmp_path / "tars",
        dest=tmp_path / "audio",
        base_url=BASE,
        **kw,
    )


@respx.mock
def test_fetch_verifies_extracts_and_deletes_tar(tmp_path: Path) -> None:
    data = _tar_bytes({"14/214.low.mp3": b"audio"})
    respx.get("https://cdn.example.test/a.tar").respond(200, content=data)
    with httpx.Client() as client:
        result = _fetch(client, "a.tar", hashlib.sha256(data).hexdigest(), tmp_path)
    assert result == "fetched"
    assert (tmp_path / "audio" / "14" / "214.low.mp3").read_bytes() == b"audio"
    assert (tmp_path / "tars" / "a.tar.done").exists()
    assert not (tmp_path / "tars" / "a.tar").exists()
    assert not (tmp_path / "tars" / "a.tar.part").exists()


@respx.mock
def test_fetch_keep_tar(tmp_path: Path) -> None:
    data = _tar_bytes({"1/1.low.mp3": b"x"})
    respx.get("https://cdn.example.test/a.tar").respond(200, content=data)
    with httpx.Client() as client:
        _fetch(client, "a.tar", hashlib.sha256(data).hexdigest(), tmp_path, keep_tar=True)
    assert (tmp_path / "tars" / "a.tar").exists()


@respx.mock
def test_fetch_sha_mismatch_rejected_and_part_removed(tmp_path: Path) -> None:
    data = _tar_bytes({"1/1.low.mp3": b"x"})
    respx.get("https://cdn.example.test/a.tar").respond(200, content=data)
    with httpx.Client() as client, pytest.raises(fj.ArchiveError, match="mismatch"):
        _fetch(client, "a.tar", "0" * 64, tmp_path)
    assert not (tmp_path / "tars" / "a.tar.part").exists()
    assert not (tmp_path / "audio").exists()
    assert not (tmp_path / "tars" / "a.tar.done").exists()


@respx.mock
def test_fetch_skips_done_archive(tmp_path: Path) -> None:
    (tmp_path / "tars").mkdir()
    (tmp_path / "tars" / "a.tar.done").write_text("x")
    route = respx.get("https://cdn.example.test/a.tar").respond(200, content=b"")
    with httpx.Client() as client:
        assert _fetch(client, "a.tar", "0" * 64, tmp_path) == "skipped"
    assert not route.called


@respx.mock
def test_fetch_blocks_path_traversal(tmp_path: Path) -> None:
    data = _tar_bytes({"../evil.txt": b"pwn"})
    respx.get("https://cdn.example.test/a.tar").respond(200, content=data)
    with httpx.Client() as client, pytest.raises(fj.ArchiveError):
        _fetch(client, "a.tar", hashlib.sha256(data).hexdigest(), tmp_path)
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "tars" / "a.tar.done").exists()


@respx.mock
def test_fetch_resumes_partial_download(tmp_path: Path) -> None:
    data = _tar_bytes({"1/1.low.mp3": b"x" * 2000})
    half = len(data) // 2
    (tmp_path / "tars").mkdir()
    (tmp_path / "tars" / "a.tar.part").write_bytes(data[:half])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["range"] == f"bytes={half}-"
        return httpx.Response(206, content=data[half:])

    respx.get("https://cdn.example.test/a.tar").mock(side_effect=handler)
    with httpx.Client() as client:
        assert _fetch(client, "a.tar", hashlib.sha256(data).hexdigest(), tmp_path) == "fetched"
    assert (tmp_path / "audio" / "1" / "1.low.mp3").read_bytes() == b"x" * 2000


@respx.mock
def test_fetch_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    data = _tar_bytes({"1/1.low.mp3": b"y" * 2000})
    (tmp_path / "tars").mkdir()
    (tmp_path / "tars" / "a.tar.part").write_bytes(b"garbage")
    respx.get("https://cdn.example.test/a.tar").respond(200, content=data)
    with httpx.Client() as client:
        assert _fetch(client, "a.tar", hashlib.sha256(data).hexdigest(), tmp_path) == "fetched"
    assert (tmp_path / "audio" / "1" / "1.low.mp3").read_bytes() == b"y" * 2000
