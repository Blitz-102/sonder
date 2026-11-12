import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sonder.audio import guitar
from sonder.audio.clap import EMBED_DIM, SAMPLE_RATE
from sonder.audio.guitar import (
    DEMUCS_SAMPLE_RATE,
    GUITAR_MODEL_VERSION,
    GuitarResult,
    GuitarSeparator,
    analyze_guitar,
    guitar_presence,
    resample_to_clap,
    rms,
)

SOURCES = ["drums", "bass", "other", "vocals", "guitar", "piano"]


class FakeModel:
    sources = SOURCES


class FakeApply:
    """Fake apply_model: puts `gain * input` in the guitar slot and zeros elsewhere."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = gain
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []

    def __call__(self, model: Any, mix: np.ndarray, **kwargs: Any) -> np.ndarray:
        self.calls.append((mix, kwargs))
        out = np.zeros((1, len(model.sources), *mix.shape[1:]), dtype=np.float32)
        out[0, model.sources.index("guitar")] = self.gain * mix[0]
        return out


def fake_separator(gain: float = 1.0) -> tuple[GuitarSeparator, FakeApply]:
    apply_fn = FakeApply(gain)
    return GuitarSeparator(model=FakeModel(), device="cpu", apply_fn=apply_fn), apply_fn


def stereo_signal(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * DEMUCS_SAMPLE_RATE)
    t = np.arange(n) / DEMUCS_SAMPLE_RATE
    left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(n)
    right = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.05 * rng.standard_normal(n)
    return (np.stack([left, right]) + 0.02).astype(np.float32)


def _boom(audio: np.ndarray, max_windows: int | None) -> np.ndarray:
    raise AssertionError("embed must not be called")


def test_rms_and_presence() -> None:
    mix = stereo_signal()
    assert rms(np.ones(10, dtype=np.float32)) == pytest.approx(1.0)
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0
    assert guitar_presence(0.5 * mix, mix) == 0.5
    assert guitar_presence(mix, np.zeros_like(mix)) == 0.0


def test_model_version() -> None:
    assert GUITAR_MODEL_VERSION == "htdemucs_6s+laion/larger_clap_music"


def test_resample_length_and_dtype() -> None:
    out = resample_to_clap(np.zeros((2, 30 * DEMUCS_SAMPLE_RATE), dtype=np.float32))
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(out.size - 30 * SAMPLE_RATE) <= 1


def test_resample_preserves_frequency_and_downmixes() -> None:
    t = np.arange(DEMUCS_SAMPLE_RATE) / DEMUCS_SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out = resample_to_clap(np.stack([tone, tone]))
    peak = np.fft.rfftfreq(out.size, 1 / SAMPLE_RATE)[np.argmax(np.abs(np.fft.rfft(out)))]
    assert peak == pytest.approx(440.0, abs=1.5)
    assert rms(out) == pytest.approx(rms(tone), rel=0.02)
    # Opposite-phase channels cancel in the mono downmix.
    assert rms(resample_to_clap(np.stack([tone, -tone]))) < 1e-4


def test_normalisation_round_trip_and_apply_args() -> None:
    mix = stereo_signal()
    separator, apply_fn = fake_separator()
    stem = separator.separate(mix)
    assert stem.shape == mix.shape
    assert stem.dtype == np.float32
    np.testing.assert_allclose(stem, mix, atol=1e-4)
    normed, kwargs = apply_fn.calls[0]
    assert normed.shape == (1, 2, mix.shape[1])
    ref = normed[0].mean(axis=0)
    assert abs(float(ref.mean())) < 1e-3
    assert float(ref.std(ddof=1)) == pytest.approx(1.0, abs=1e-3)
    assert kwargs == {"shifts": 0, "split": True, "overlap": 0.25, "device": "cpu"}


def test_mono_input_rejected() -> None:
    separator, _ = fake_separator()
    with pytest.raises(ValueError, match="channels=2"):
        separator.separate(np.zeros(1000, dtype=np.float32))
    with pytest.raises(ValueError, match="channels=2"):
        analyze_guitar(np.zeros(1000, dtype=np.float32), separator=separator)


def test_analyze_below_threshold_keeps_presence_drops_embedding() -> None:
    separator, _ = fake_separator(0.05)
    result = analyze_guitar(stereo_signal(), separator=separator, embed=_boom, threshold=0.15)
    assert isinstance(result, GuitarResult)
    assert 0.0 < result.presence < 0.15
    assert result.embedding is None


def test_analyze_above_threshold_embeds_resampled_stem() -> None:
    mix = stereo_signal()
    seen: dict[str, Any] = {}
    vec = np.ones(EMBED_DIM, dtype=np.float32)

    def embed(audio: np.ndarray, max_windows: int | None) -> np.ndarray:
        seen["audio"], seen["max_windows"] = audio, max_windows
        return vec

    separator, _ = fake_separator(0.5)
    result = analyze_guitar(mix, separator=separator, embed=embed, threshold=0.15, max_windows=3)
    assert result.presence == pytest.approx(0.5, abs=0.01)
    assert result.embedding is vec
    assert seen["max_windows"] == 3
    audio = seen["audio"]
    assert audio.ndim == 1 and audio.dtype == np.float32
    assert abs(audio.size - mix.shape[1] * 160 / 147) <= 1


def test_analyze_default_threshold_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUITAR_PRESENCE_MIN", "0.9")
    guitar.get_settings.cache_clear()
    separator, _ = fake_separator(0.5)
    try:
        result = analyze_guitar(stereo_signal(), separator=separator, embed=_boom)
    finally:
        guitar.get_settings.cache_clear()
    assert result.embedding is None
    assert result.presence == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("shape", [(2, 44100), (2, 0)])
def test_silent_or_empty_input(shape: tuple[int, int]) -> None:
    separator, apply_fn = fake_separator()
    result = analyze_guitar(np.zeros(shape, dtype=np.float32), separator=separator)
    assert result.presence == 0.0
    assert result.embedding is None
    assert apply_fn.calls == []


def test_injected_separator_does_not_import_torch() -> None:
    code = (
        "import numpy as np, sys\n"
        "from sonder.audio.guitar import GuitarSeparator\n"
        "class M: sources = ['drums','bass','other','vocals','guitar','piano']\n"
        "def fn(model, mix, **kw):\n"
        "    out = np.zeros((1, 6, *mix.shape[1:]), dtype=np.float32)\n"
        "    out[0, 4] = mix[0]\n"
        "    return out\n"
        "GuitarSeparator(model=M(), device='cpu', apply_fn=fn).separate(\n"
        "    np.random.default_rng(0).standard_normal((2, 1000)).astype(np.float32))\n"
        "assert 'torch' not in sys.modules and 'demucs_infer' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_import_does_not_load_ml_libs() -> None:
    code = (
        "import sonder.audio.guitar, sys; "
        "bad = [m for m in ('torch', 'transformers', 'demucs_infer') if m in sys.modules]; "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ---- slow tests: real htdemucs_6s and CLAP ----

WEIGHTS_DIR = Path.home() / ".cache" / "demucs-infer"


@pytest.fixture(scope="module")
def real_separator() -> GuitarSeparator:
    pytest.importorskip("torch")
    pytest.importorskip("demucs_infer")
    if not WEIGHTS_DIR.is_dir() or not any(WEIGHTS_DIR.rglob("*")):
        pytest.skip(f"htdemucs_6s weights not found in {WEIGHTS_DIR}")
    return GuitarSeparator()


@pytest.mark.slow
def test_real_separator_shape_dtype_and_determinism(real_separator: GuitarSeparator) -> None:
    mix = stereo_signal(10.0, seed=1)
    stem1 = real_separator.separate(mix)
    stem2 = real_separator.separate(mix)
    assert "guitar" in real_separator._model.sources
    assert stem1.shape == mix.shape
    assert stem1.dtype == np.float32
    assert np.isfinite(stem1).all()
    assert np.array_equal(stem1, stem2) or np.allclose(stem1, stem2, atol=1e-5, rtol=0)


@pytest.mark.slow
def test_real_separator_frees_gpu_memory(real_separator: GuitarSeparator) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("cuda not available")
    real_separator.separate(stereo_signal(10.0, seed=2))
    assert torch.cuda.memory_allocated() < 500 * 2**20


@pytest.mark.slow
def test_real_analyze_guitar_end_to_end(real_separator: GuitarSeparator) -> None:
    pytest.importorskip("transformers")
    mix = stereo_signal(10.0, seed=3)
    with_emb = analyze_guitar(mix, separator=real_separator, threshold=0.0, max_windows=2)
    assert with_emb.embedding is not None
    assert with_emb.embedding.shape == (EMBED_DIM,)
    assert with_emb.embedding.dtype == np.float32
    assert np.linalg.norm(with_emb.embedding) == pytest.approx(1.0, abs=1e-4)
    without = analyze_guitar(mix, separator=real_separator, threshold=10.0)
    assert without.embedding is None
    assert without.presence == with_emb.presence
