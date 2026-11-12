from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

MODEL_ID = "laion/larger_clap_music"
MODEL_VERSION = "laion/larger_clap_music"
SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 480_000
EMBED_DIM = 512
MAX_TRACK_WINDOWS = 18
DEFAULT_BATCH_SIZE = 6


def split_windows(audio: np.ndarray, max_windows: int | None = None) -> list[np.ndarray]:
    """Cut audio into consecutive 10 s windows, zero-padding the last one.

    If `max_windows` is set and there are more windows, pick that many evenly spread.
    """
    if audio.size == 0:
        raise ValueError("cannot window empty audio")
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    audio = np.asarray(audio, dtype=np.float32)
    n_windows = -(-audio.size // WINDOW_SAMPLES)
    indices = np.arange(n_windows)
    if max_windows is not None and n_windows > max_windows:
        indices = np.unique(np.linspace(0, n_windows - 1, max_windows).round().astype(int))
    windows: list[np.ndarray] = []
    for i in indices:
        chunk = audio[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]
        window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        window[: chunk.size] = chunk
        windows.append(window)
    return windows


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return np.asarray(x.detach().float().cpu().numpy(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


class ClapEmbedder:
    """CLAP audio embedder. Loads the model lazily; torch is imported only when needed."""

    def __init__(
        self,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        self._device = device
        self.batch_size = batch_size
        self._model = model
        self._processor = processor
        self._injected = model is not None and processor is not None
        self._dtype: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        import torch
        from transformers import AutoProcessor, ClapModel

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._dtype = torch.float16 if device == "cuda" else torch.float32
        model = ClapModel.from_pretrained(MODEL_ID)
        model = model.to(device=device, dtype=self._dtype)
        model.eval()
        self._model = model
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)

    def _embed_batch(self, batch: list[np.ndarray]) -> np.ndarray:
        inputs = self._processor(audio=batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        if self._injected:
            out = self._model.get_audio_features(**inputs)
        else:
            import torch

            moved: dict[str, Any] = {}
            for key, value in inputs.items():
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    moved[key] = value.to(device=self._device, dtype=self._dtype)
                else:
                    moved[key] = value.to(self._device)
            with torch.inference_mode():
                out = self._model.get_audio_features(**moved)
        if hasattr(out, "pooler_output"):
            out = out.pooler_output
        return _to_numpy(out)

    def embed_windows(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        """Embed windows in batches. Returns (N, EMBED_DIM) float32."""
        self._ensure_loaded()
        parts = [
            self._embed_batch(list(windows[i : i + self.batch_size]))
            for i in range(0, len(windows), self.batch_size)
        ]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    def embed_audio(self, audio: np.ndarray, max_windows: int | None = None) -> np.ndarray:
        """Mean-pooled, L2-normalised embedding of 48 kHz mono audio, shape (EMBED_DIM,)."""
        windows = split_windows(audio, max_windows)
        embeddings = self.embed_windows(windows)
        pooled = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("embedding has zero or non-finite norm; cannot normalise")
        result = (pooled / norm).astype(np.float32)
        assert result.shape == (EMBED_DIM,), f"unexpected embedding shape {result.shape}"
        return result


@lru_cache(maxsize=1)
def get_embedder() -> ClapEmbedder:
    return ClapEmbedder()


def embed_audio(audio: np.ndarray, max_windows: int | None = None) -> np.ndarray:
    return get_embedder().embed_audio(audio, max_windows)
