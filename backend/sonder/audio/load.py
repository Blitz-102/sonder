import subprocess
from pathlib import Path

import numpy as np


class FfmpegNotFoundError(RuntimeError):
    """ffmpeg is not installed or not on PATH."""


class AudioDecodeError(RuntimeError):
    """ffmpeg could not decode the input."""


def load_audio(path: Path | str, sr: int) -> np.ndarray:
    """Decode any audio file to mono float32 at `sr` Hz via ffmpeg."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-vn",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise FfmpegNotFoundError(
            "ffmpeg not found on PATH. Install it with: winget install Gyan.FFmpeg"
        ) from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg failed on {path} (exit {proc.returncode}): {stderr}")
    if not proc.stdout:
        raise AudioDecodeError(f"ffmpeg produced no audio for {path}")
    usable = len(proc.stdout) - len(proc.stdout) % 4
    return np.frombuffer(proc.stdout[:usable], dtype="<f4").astype(np.float32, copy=True)
