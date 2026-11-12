import subprocess
from pathlib import Path

import numpy as np


class FfmpegNotFoundError(RuntimeError):
    """ffmpeg is not installed or not on PATH."""


class AudioDecodeError(RuntimeError):
    """ffmpeg could not decode the input."""


def load_audio(path: Path | str, sr: int, channels: int = 1) -> np.ndarray:
    """Decode any audio file to float32 at `sr` Hz via ffmpeg.

    `channels == 1` returns a 1-D array of shape (n,). `channels > 1` returns a C-contiguous
    array of shape (channels, n); mono sources are duplicated across channels.
    """
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
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
        str(channels),
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
    frame_bytes = 4 * channels
    usable = len(proc.stdout) - len(proc.stdout) % frame_bytes
    if usable == 0:
        raise AudioDecodeError(f"ffmpeg produced no complete audio frame for {path}")
    data = np.frombuffer(proc.stdout[:usable], dtype="<f4").astype(np.float32, copy=True)
    if channels == 1:
        return data
    return np.ascontiguousarray(data.reshape(-1, channels).T)
