from dataclasses import dataclass

import librosa
import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass(frozen=True)
class AudioFeatures:
    tempo: float | None
    key: str | None
    mode: str | None
    loudness_db: float | None


def estimate_tempo(audio: np.ndarray, sr: int) -> float | None:
    # Not `librosa.beat.beat_track`: its returned tempo is often off by an octave (a 120 bpm
    # click track gives 60, accented clicks give 30). `feature.tempo` uses the same onset
    # envelope and tempogram and stays within ~3% on the same signals. (Call it as
    # `librosa.feature.tempo`: `librosa.feature.rhythm` is lazy-loaded and is missing from
    # a fresh process unless something else already imported it.)
    tempo = librosa.feature.tempo(y=audio, sr=sr)
    value = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(value) or value <= 0:
        return None
    return value


def estimate_key(audio: np.ndarray, sr: int) -> tuple[str, str] | None:
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    profile = np.asarray(chroma.mean(axis=1), dtype=np.float64)
    if not np.all(np.isfinite(profile)) or np.std(profile) < 1e-8:
        return None
    best_score = -np.inf
    best: tuple[str, str] | None = None
    for mode, template in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for tonic in range(12):
            score = float(np.corrcoef(profile, np.roll(template, tonic))[0, 1])
            if score > best_score:
                best_score = score
                best = (PITCH_CLASSES[tonic], mode)
    return best


def loudness_db(audio: np.ndarray) -> float | None:
    """RMS level in dBFS, or None for silence."""
    if audio.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    if rms == 0.0 or not np.isfinite(rms):
        return None
    return 20.0 * float(np.log10(rms))


def extract_features(audio: np.ndarray, sr: int, deezer_bpm: float | None = None) -> AudioFeatures:
    tempo = float(deezer_bpm) if deezer_bpm and deezer_bpm > 0 else estimate_tempo(audio, sr)
    key_mode = estimate_key(audio, sr)
    return AudioFeatures(
        tempo=tempo,
        key=key_mode[0] if key_mode else None,
        mode=key_mode[1] if key_mode else None,
        loudness_db=loudness_db(audio),
    )
