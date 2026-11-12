import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from sonder.audio.load import AudioDecodeError, FfmpegNotFoundError, load_audio

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def sine(sr: int, seconds: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@needs_ffmpeg
def test_load_resamples_to_mono_float32(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    sf.write(path, sine(22050), 22050)
    audio = load_audio(path, 16000)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 16000) <= 200
    assert 0.1 <= float(np.abs(audio).max()) <= 1.0


@needs_ffmpeg
def test_load_accepts_str_path(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    sf.write(path, sine(22050), 22050)
    assert load_audio(str(path), 16000).ndim == 1


@needs_ffmpeg
def test_stereo_becomes_mono(tmp_path: Path) -> None:
    mono = sine(22050)
    stereo = np.stack([mono, mono], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, 22050)
    audio = load_audio(path, 22050)
    assert audio.ndim == 1
    assert abs(len(audio) - 22050) <= 200


@needs_ffmpeg
def test_non_audio_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("this is not audio")
    with pytest.raises(AudioDecodeError):
        load_audio(path, 16000)


@needs_ffmpeg
def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(AudioDecodeError):
        load_audio(tmp_path / "nope.wav", 16000)


def test_missing_ffmpeg_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FfmpegNotFoundError, match="winget install Gyan.FFmpeg"):
        load_audio("x.wav", 16000)


def two_tone_stereo(sr: int) -> np.ndarray:
    return np.stack([sine(sr, freq=440.0), sine(sr, freq=1000.0)], axis=1)


def dominant_freq(x: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(x))
    return float(np.fft.rfftfreq(x.size, 1 / sr)[int(np.argmax(spectrum))])


@needs_ffmpeg
def test_stereo_decode_shape_and_channels(tmp_path: Path) -> None:
    path = tmp_path / "lr.wav"
    sf.write(path, two_tone_stereo(44100), 44100)
    audio = load_audio(path, 44100, channels=2)
    assert audio.shape[0] == 2
    assert abs(audio.shape[1] - 44100) <= 200
    assert audio.dtype == np.float32
    assert audio.flags["C_CONTIGUOUS"]
    assert dominant_freq(audio[0], 44100) == pytest.approx(440.0, abs=2.0)
    assert dominant_freq(audio[1], 44100) == pytest.approx(1000.0, abs=2.0)


@needs_ffmpeg
def test_mono_file_with_two_channels_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "mono.wav"
    sf.write(path, sine(44100), 44100)
    audio = load_audio(path, 44100, channels=2)
    assert audio.shape[0] == 2
    np.testing.assert_array_equal(audio[0], audio[1])


@needs_ffmpeg
def test_channels_one_still_returns_1d(tmp_path: Path) -> None:
    path = tmp_path / "lr.wav"
    sf.write(path, two_tone_stereo(44100), 44100)
    assert load_audio(path, 44100, channels=1).ndim == 1


def test_trims_partial_frame_for_stereo(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = np.arange(10, dtype="<f4").tobytes()  # 40 bytes = 5 stereo frames
    fake = SimpleNamespace(returncode=0, stdout=frames + b"\x00" * 4, stderr=b"")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    audio = load_audio("x.wav", 44100, channels=2)
    assert audio.shape == (2, 5)
    np.testing.assert_array_equal(audio[0], [0, 2, 4, 6, 8])
    np.testing.assert_array_equal(audio[1], [1, 3, 5, 7, 9])


def test_no_whole_frame_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(returncode=0, stdout=b"\x00" * 4, stderr=b"")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: fake)
    with pytest.raises(AudioDecodeError):
        load_audio("x.wav", 44100, channels=2)


def test_zero_channels_raises() -> None:
    with pytest.raises(ValueError):
        load_audio("x.wav", 44100, channels=0)
