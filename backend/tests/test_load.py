import shutil
import subprocess
from pathlib import Path
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
