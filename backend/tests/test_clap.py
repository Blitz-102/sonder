import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from sonder.audio import clap
from sonder.audio.clap import (
    EMBED_DIM,
    MAX_TRACK_WINDOWS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    ClapEmbedder,
    split_windows,
)


def test_short_audio_one_padded_window() -> None:
    audio = np.ones(1000, dtype=np.float32)
    windows = split_windows(audio)
    assert len(windows) == 1
    assert windows[0].shape == (WINDOW_SAMPLES,)
    assert windows[0].dtype == np.float32
    assert np.all(windows[0][:1000] == 1)
    assert np.all(windows[0][1000:] == 0)


def test_last_window_zero_padded() -> None:
    audio = np.ones(WINDOW_SAMPLES + 100, dtype=np.float32)
    windows = split_windows(audio)
    assert len(windows) == 2
    assert all(w.shape == (WINDOW_SAMPLES,) for w in windows)
    assert np.all(windows[1][:100] == 1)
    assert np.all(windows[1][100:] == 0)


def test_30s_clip_three_windows() -> None:
    assert len(split_windows(np.full(30 * SAMPLE_RATE, 0.1, dtype=np.float32))) == 3


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        split_windows(np.zeros(0, dtype=np.float32))


def test_max_windows_spreads_evenly() -> None:
    # Each sample holds its own index, so a window's first value identifies its start.
    audio = np.arange(180 * SAMPLE_RATE, dtype=np.float32)
    windows = split_windows(audio, max_windows=MAX_TRACK_WINDOWS)
    assert len(windows) == 18
    starts = [float(w[0]) for w in windows]
    assert starts[0] == 0
    assert starts[-1] == 17 * WINDOW_SAMPLES
    assert starts == sorted(starts)
    assert len(set(starts)) == 18


def test_max_windows_not_applied_when_fewer() -> None:
    audio = np.ones(25 * SAMPLE_RATE, dtype=np.float32)
    assert len(split_windows(audio, max_windows=18)) == 3


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, **kwargs: Any) -> dict[str, np.ndarray]:
        audio = kwargs["audio"]
        assert kwargs["sampling_rate"] == SAMPLE_RATE
        self.calls.append(len(audio))
        # Pass the first sample of each window through so the model can derive embeddings.
        return {"x": np.array([w[0] for w in audio], dtype=np.float32)}


class FakeModel:
    def __init__(self, wrap: bool, dim: int = EMBED_DIM) -> None:
        self.wrap = wrap
        self.dim = dim
        self.calls = 0

    def get_audio_features(self, x: np.ndarray) -> Any:
        self.calls += 1
        out = np.zeros((len(x), self.dim), dtype=np.float32)
        out[:, 0] = x
        out[:, 1] = 1.0
        return SimpleNamespace(pooler_output=out) if self.wrap else out


def windows_with_values(values: list[float]) -> list[np.ndarray]:
    return [np.full(WINDOW_SAMPLES, v, dtype=np.float32) for v in values]


def make(wrap: bool = True, batch_size: int = 6, dim: int = EMBED_DIM) -> ClapEmbedder:
    return ClapEmbedder(
        batch_size=batch_size, model=FakeModel(wrap, dim), processor=FakeProcessor()
    )


@pytest.mark.parametrize("wrap", [True, False])
def test_embed_windows_shape(wrap: bool) -> None:
    out = make(wrap).embed_windows(windows_with_values([1.0, 2.0]))
    assert out.shape == (2, EMBED_DIM)
    assert out.dtype == np.float32
    assert out[1, 0] == 2.0


@pytest.mark.parametrize("wrap", [True, False])
def test_embed_audio_mean_and_normalise(wrap: bool) -> None:
    audio = np.concatenate(
        [np.full(WINDOW_SAMPLES, 1.0, np.float32), np.full(WINDOW_SAMPLES, 3.0, np.float32)]
    )
    emb = make(wrap).embed_audio(audio)
    assert emb.shape == (EMBED_DIM,)
    assert emb.dtype == np.float32
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-6)
    # mean of [1,1,0..] and [3,1,0..] is [2,1,0..]
    expected = np.array([2.0, 1.0]) / np.sqrt(5.0)
    np.testing.assert_allclose(emb[:2], expected, rtol=1e-5)


def test_batching() -> None:
    embedder = make(batch_size=2)
    embedder.embed_windows(windows_with_values([1.0] * 5))
    assert embedder._model.calls == 3
    assert embedder._processor.calls == [2, 2, 1]


def test_wrong_dim_raises() -> None:
    audio = np.ones(WINDOW_SAMPLES, dtype=np.float32)
    with pytest.raises((AssertionError, ValueError)):
        make(dim=256).embed_audio(audio)


def test_zero_vector_raises() -> None:
    embedder = ClapEmbedder(
        model=SimpleNamespace(
            get_audio_features=lambda x: np.zeros((len(x), EMBED_DIM), dtype=np.float32)
        ),
        processor=FakeProcessor(),
    )
    with pytest.raises(ValueError, match="zero"):
        embedder.embed_audio(np.ones(WINDOW_SAMPLES, dtype=np.float32))


def test_import_does_not_load_ml_libs() -> None:
    code = (
        "import sonder.audio.clap, sys; "
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_module_wrapper_uses_cached_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = make()
    monkeypatch.setattr(clap, "get_embedder", lambda: fake)
    emb = clap.embed_audio(np.ones(WINDOW_SAMPLES, dtype=np.float32))
    assert emb.shape == (EMBED_DIM,)


def _signal(seconds: float = 12.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    sines = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 330 * t)
    return (sines + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


@pytest.mark.slow
def test_real_model_deterministic_and_discriminative() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    x = _signal()
    e1 = clap.embed_audio(x)
    e2 = clap.embed_audio(x)
    assert e1.shape == (EMBED_DIM,)
    assert e1.dtype == np.float32
    assert np.linalg.norm(e1) == pytest.approx(1.0, abs=1e-4)
    assert np.array_equal(e1, e2)

    t = np.arange(10 * SAMPLE_RATE) / SAMPLE_RATE
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    noise = (0.3 * np.random.default_rng(1).standard_normal(t.size)).astype(np.float32)
    sim = float(np.dot(clap.embed_audio(sine), clap.embed_audio(noise)))
    assert sim < 0.999
