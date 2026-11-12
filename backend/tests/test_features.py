import numpy as np
import pytest

from sonder.audio.features import (
    estimate_key,
    estimate_tempo,
    extract_features,
    loudness_db,
)

SR = 22050


def tone(freqs: list[float], seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    out = np.zeros_like(t)
    for f in freqs:
        for harmonic, amp in ((1, 1.0), (2, 0.5), (3, 0.25)):
            out += amp * np.sin(2 * np.pi * f * harmonic * t)
    return (0.3 * out / np.abs(out).max()).astype(np.float32)


def click_track(bpm: float, seconds: float = 12.0) -> np.ndarray:
    audio = np.zeros(int(SR * seconds), dtype=np.float32)
    click = np.hanning(256).astype(np.float32) * np.sin(2 * np.pi * 1000 * np.arange(256) / SR)
    step = int(SR * 60 / bpm)
    for start in range(0, len(audio) - 256, step):
        audio[start : start + 256] += click
    return audio


@pytest.mark.parametrize("bpm", [80, 100, 120, 140, 160])
def test_tempo_click_track(bpm: int) -> None:
    tempo = estimate_tempo(click_track(bpm), SR)
    assert tempo is not None
    assert abs(tempo - bpm) <= 0.04 * bpm


def test_tempo_accented_click_track_is_not_an_octave_off() -> None:
    audio = click_track(120)
    # Every fourth click at full level, the rest at half (a 4/4 bar), like a real downbeat.
    step = int(SR * 60 / 120)
    for i, start in enumerate(range(0, len(audio) - 256, step)):
        if i % 4:
            audio[start : start + 256] *= 0.5
    tempo = estimate_tempo(audio, SR)
    assert tempo is not None
    assert abs(tempo - 120) <= 0.04 * 120


def test_key_c_major() -> None:
    assert estimate_key(tone([261.63, 329.63, 392.00]), SR) == ("C", "major")


def test_key_a_minor() -> None:
    assert estimate_key(tone([220.00, 261.63, 329.63]), SR) == ("A", "minor")


def test_silence() -> None:
    silence = np.zeros(SR * 2, dtype=np.float32)
    assert estimate_key(silence, SR) is None
    assert loudness_db(silence) is None


def test_loudness_sine() -> None:
    t = np.arange(SR) / SR
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    value = loudness_db(audio)
    assert value is not None
    assert abs(value - 20 * np.log10(0.5 / np.sqrt(2))) <= 0.5


def test_extract_features_prefers_deezer_bpm() -> None:
    features = extract_features(click_track(120, 8.0), SR, deezer_bpm=97.0)
    assert features.tempo == 97.0
    assert features.loudness_db is not None


def test_extract_features_falls_back_to_detection() -> None:
    audio = click_track(120, 12.0)
    for bpm in (0, None):
        features = extract_features(audio, SR, deezer_bpm=bpm)
        assert features.tempo is not None
        assert abs(features.tempo - 120) <= 6


def test_extract_features_key_fields() -> None:
    features = extract_features(tone([261.63, 329.63, 392.00]), SR, deezer_bpm=100.0)
    assert (features.key, features.mode) == ("C", "major")
