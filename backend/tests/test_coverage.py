import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf

from sonder.config import REPO_ROOT
from sonder.sources.deezer import DeezerTrack
from sonder.sources.itunes import ItunesTrack


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "coverage_script", REPO_ROOT / "scripts" / "preview_coverage.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["coverage_script"] = module
    spec.loader.exec_module(module)
    return module


cov: Any = _load()


def dz(id: int, artist: str, title: str) -> DeezerTrack:
    return DeezerTrack(
        id=id,
        title=title,
        title_short=title,
        artist=artist,
        album_title="",
        cover_url=None,
        duration=200,
        isrc=None,
        bpm=None,
        rank=0,
    )


def it(artist: str, title: str, preview: str | None) -> ItunesTrack:
    return ItunesTrack(
        track_id=1,
        title=title,
        artist=artist,
        album="",
        preview_url=preview,
        duration_ms=1000,
        artwork_url=None,
    )


class FakeDeezer:
    def __init__(self, tracks: list[DeezerTrack], previews: dict[int, str | None]) -> None:
        self.tracks = tracks
        self.previews = previews

    def search(self, artist: str, title: str, limit: int = 10) -> list[DeezerTrack]:
        return self.tracks

    def fresh_preview_url(self, track_id: int) -> str | None:
        return self.previews.get(track_id)


class FakeItunes:
    def __init__(self, tracks: list[ItunesTrack]) -> None:
        self.tracks = tracks
        self.calls = 0

    def search_song(
        self, artist: str, title: str, limit: int = 10, country: str = "US"
    ) -> list[ItunesTrack]:
        self.calls += 1
        return self.tracks


SEED = cov.Seed("Radiohead", "Creep")


def fake_measure(url: str) -> float:
    return 30.0


def test_deezer_hit_skips_itunes() -> None:
    deezer = FakeDeezer([dz(1, "Radiohead", "Creep")], {1: "https://dz/p.mp3"})
    itunes = FakeItunes([])
    r = cov.check_seed(SEED, deezer, itunes, fake_measure)
    assert (r.reason, r.deezer_id, r.deezer_preview, r.itunes_rescued) == ("ok", 1, True, False)
    assert r.duration == 30.0
    assert itunes.calls == 0


def test_itunes_rescues_deezer_without_preview() -> None:
    deezer = FakeDeezer([dz(1, "Radiohead", "Creep")], {1: None})
    itunes = FakeItunes([it("Radiohead", "Creep", "https://it/p.m4a")])
    r = cov.check_seed(SEED, deezer, itunes, fake_measure)
    assert (r.reason, r.deezer_id, r.deezer_preview, r.itunes_rescued) == ("ok", 1, False, True)


def test_itunes_rescues_deezer_miss() -> None:
    deezer = FakeDeezer([], {})
    itunes = FakeItunes([it("Radiohead", "Creep", "https://it/p.m4a")])
    r = cov.check_seed(SEED, deezer, itunes, None)
    assert (r.reason, r.deezer_id, r.itunes_rescued, r.duration) == ("ok", None, True, None)


def test_total_miss_reasons() -> None:
    miss = cov.check_seed(SEED, FakeDeezer([], {}), FakeItunes([]), fake_measure)
    assert (miss.reason, miss.deezer_id) == ("no_deezer_match", None)
    no_preview = cov.check_seed(
        SEED,
        FakeDeezer([dz(1, "Radiohead", "Creep")], {1: None}),
        FakeItunes([it("Radiohead", "Creep", None)]),
        fake_measure,
    )
    assert (no_preview.reason, no_preview.deezer_id) == ("no_preview", 1)


def test_measure_failure_does_not_fail_seed() -> None:
    def boom(url: str) -> float:
        raise RuntimeError(f"bad audio at {url}?hdnea=exp=1")

    deezer = FakeDeezer([dz(1, "Radiohead", "Creep")], {1: "https://dz/p.mp3"})
    r = cov.check_seed(SEED, deezer, FakeItunes([]), boom)
    assert (r.reason, r.duration) == ("ok", None)
    # The failure is recorded (class and message) but the preview URL is scrubbed.
    assert r.measure_error == "RuntimeError: bad audio at <url>"
    summary = cov.summarize([r])
    assert summary.pct_preview == 100.0
    assert summary.measure_errors == ["Radiohead - Creep: RuntimeError: bad audio at <url>"]
    assert "Duration measurement failed for 1 clip(s)" in cov.format_report(summary)


def _result(artist: str, title: str, reason: str, **kw: Any) -> Any:
    return cov.SeedResult(
        cov.Seed(artist, title),
        kw.get("deezer_id"),
        kw.get("deezer_preview", False),
        kw.get("itunes_rescued", False),
        kw.get("duration"),
        reason,
    )


def test_summary_and_gate_verdict() -> None:
    results = [
        _result("A", "1", "ok", deezer_id=1, deezer_preview=True, duration=30.0),
        _result("A", "2", "ok", deezer_id=2, deezer_preview=True, duration=28.0),
        _result("B", "3", "ok", deezer_id=3, itunes_rescued=True, duration=29.5),
        _result("C", "4", "ok", itunes_rescued=True),
        _result("D", "5", "no_deezer_match"),
    ]
    s = cov.summarize(results)
    assert s.n == 5
    assert s.pct_deezer_match == 60.0
    assert s.pct_deezer_preview == 40.0
    assert s.pct_itunes_rescued == 40.0
    assert s.pct_preview == 80.0
    assert s.durations == [30.0, 28.0, 29.5]
    assert s.short_clips == 1
    assert s.failures_by_artist == {"D": ["5"]}
    report = cov.format_report(s)
    assert "GATE A: PASS" in report
    assert "D: 5" in report

    failing = cov.summarize(results + [_result("E", "6", "no_preview", deezer_id=6)])
    assert "GATE A: FAIL" in cov.format_report(failing)


def test_summary_empty() -> None:
    s = cov.summarize([])
    assert s.n == 0
    assert "GATE A: FAIL" in cov.format_report(s)


def test_read_seeds_skips_blank_rows(tmp_path: Path) -> None:
    p = tmp_path / "seeds.csv"
    p.write_text("artist,title\nRadiohead,Creep\n\n , \nOasis,  Wonderwall \n", encoding="utf-8")
    assert cov.read_seeds(p) == [cov.Seed("Radiohead", "Creep"), cov.Seed("Oasis", "Wonderwall")]


def test_results_csv_has_no_urls(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    cov.write_results_csv(
        out, [_result("A", "1", "ok", deezer_id=1, deezer_preview=True, duration=30.0)]
    )
    text = out.read_text(encoding="utf-8")
    assert "http" not in text
    assert "A,1,1,True,False,30.00,ok" in text


def test_measure_duration_deletes_temp_file(tmp_path: Path) -> None:
    sr = 22050
    t = np.arange(sr * 2) / sr
    wav = tmp_path / "src.wav"
    sf.write(wav, 0.1 * np.sin(2 * np.pi * 440 * t), sr)
    payload = wav.read_bytes()
    tmp_dir = tmp_path / "tmp"
    http = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=payload))
    )
    duration = cov.measure_duration("https://example.test/clip", tmp_dir=tmp_dir, http=http)
    assert duration == pytest.approx(2.0, abs=0.05)
    assert list(tmp_dir.iterdir()) == []


def test_measure_duration_deletes_temp_file_on_failure(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "tmp"
    http = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b"not audio"))
    )
    with pytest.raises(Exception):  # noqa: B017
        cov.measure_duration("https://example.test/clip", tmp_dir=tmp_dir, http=http)
    assert list(tmp_dir.iterdir()) == []


@pytest.mark.parametrize(
    ("url", "content_type", "expected"),
    [
        # Deezer: .mp3 in the path, token in the query string
        ("https://cdn.example/p/abc.mp3?hdnea=exp=1~acl=/p/*", "audio/mpeg", ".mp3"),
        ("https://audio.example/Music/clip.M4A", "audio/mp4", ".m4a"),
        ("https://cdn.example/preview", "audio/mpeg; charset=binary", ".mp3"),
        ("https://cdn.example/preview", "audio/x-m4a", ".m4a"),
        ("https://cdn.example/preview", "", ".mp3"),
    ],
)
def test_audio_suffix(url: str, content_type: str, expected: str) -> None:
    assert cov._audio_suffix(url, {"content-type": content_type}) == expected


def test_measure_duration_uses_a_real_suffix_for_the_temp_file(tmp_path: Path) -> None:
    """libsndfile only recognises Deezer's MP3s (empty ID3 tag) when the file ends in .mp3."""
    seen: list[str] = []

    def fake_duration(*, path: str) -> float:
        seen.append(Path(path).suffix)
        return 30.0

    import librosa

    original = librosa.get_duration
    librosa.get_duration = fake_duration  # type: ignore[assignment]
    try:
        http = httpx.Client(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, content=b"x", headers={"content-type": "audio/mpeg"}
                )
            )
        )
        cov.measure_duration(
            "https://cdn.example/p/abc.mp3?hdnea=exp=1", tmp_dir=tmp_path, http=http
        )
    finally:
        librosa.get_duration = original
    assert seen == [".mp3"]
    assert list(tmp_path.iterdir()) == []
