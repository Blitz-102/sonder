import gc
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from sonder.audio.clap import MODEL_VERSION as CLAP_MODEL_VERSION
from sonder.audio.clap import SAMPLE_RATE as CLAP_SAMPLE_RATE
from sonder.config import get_settings

DEMUCS_MODEL = "htdemucs_6s"
DEMUCS_SAMPLE_RATE = 44_100
GUITAR_SOURCE = "guitar"
GUITAR_MODEL_VERSION = f"{DEMUCS_MODEL}+{CLAP_MODEL_VERSION}"

# 48000 / 44100 == 160 / 147
_UP, _DOWN = 160, 147
assert CLAP_SAMPLE_RATE * _DOWN == DEMUCS_SAMPLE_RATE * _UP

type ApplyFn = Callable[..., Any]
type EmbedFn = Callable[[np.ndarray, int | None], np.ndarray]


@dataclass(frozen=True)
class GuitarResult:
    presence: float
    embedding: np.ndarray | None


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def guitar_presence(stem: np.ndarray, mix: np.ndarray) -> float:
    """Stem RMS divided by mix RMS, rounded to 3 places. 0.0 when the mix is silent."""
    mix_rms = rms(mix)
    if mix_rms == 0.0:
        return 0.0
    return round(rms(stem) / mix_rms, 3)


def resample_to_clap(stem_stereo_44k: np.ndarray) -> np.ndarray:
    """Downmix a (channels, n) 44.1 kHz stem to mono and resample to 48 kHz float32."""
    mono = np.asarray(stem_stereo_44k, dtype=np.float32).mean(axis=0)
    return resample_poly(mono, _UP, _DOWN).astype(np.float32)


def _check_stereo(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim != 2:
        raise ValueError(
            f"expected stereo audio of shape (2, n); got {audio.ndim}-D input. "
            "Load it with load_audio(path, 44100, channels=2)."
        )
    if audio.shape[0] != 2:
        raise ValueError(f"expected 2 channels, got shape {audio.shape}")
    return audio.astype(np.float32, copy=False)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return np.asarray(x.detach().float().cpu().numpy(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


class GuitarSeparator:
    """htdemucs_6s guitar stem extractor. torch and demucs_infer load lazily on first use.

    Pass `model` (and `apply_fn`) to inject fakes; no torch import happens in that mode.
    """

    def __init__(
        self,
        model: Any = None,
        device: str | None = None,
        apply_fn: ApplyFn | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._apply_fn = apply_fn
        self._injected = model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._apply_fn is not None:
            return
        if self._injected:
            raise ValueError("an injected model needs an injected apply_fn too")
        import torch
        from demucs_infer.apply import apply_model
        from demucs_infer.pretrained import get_model

        model = get_model(DEMUCS_MODEL)
        model.eval()
        if model.samplerate != DEMUCS_SAMPLE_RATE:
            raise RuntimeError(f"{DEMUCS_MODEL} samplerate is {model.samplerate}")
        if GUITAR_SOURCE not in model.sources:
            raise RuntimeError(f"{DEMUCS_MODEL} has no {GUITAR_SOURCE} source: {model.sources}")
        self._model = model
        self._apply_fn = apply_model
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")

    def separate(self, stereo: np.ndarray) -> np.ndarray:
        """Return the guitar stem, float32 (2, n), for float32 (2, n) audio at 44.1 kHz."""
        mix = _check_stereo(stereo)
        self._ensure_loaded()
        assert self._model is not None and self._apply_fn is not None
        index = list(self._model.sources).index(GUITAR_SOURCE)

        ref = mix.mean(axis=0)
        mean = float(ref.mean())
        std = float(ref.std(ddof=1)) if ref.size > 1 else 0.0
        scale = std + 1e-8
        normed = ((mix - mean) / scale).astype(np.float32)

        if self._injected:
            out = self._apply_fn(
                self._model, normed[None], shifts=0, split=True, overlap=0.25, device=self._device
            )
            stem = _to_numpy(out)[0, index]
            return (stem * scale + mean).astype(np.float32)

        import torch

        out = None
        tensor = torch.from_numpy(normed)[None]
        try:
            with torch.inference_mode():
                out = self._apply_fn(
                    self._model, tensor, shifts=0, split=True, overlap=0.25, device=self._device
                )
                stem = _to_numpy(out[0, index])
        finally:
            del out, tensor
            gc.collect()
            if self._device == "cuda":
                torch.cuda.empty_cache()
        return (stem * scale + mean).astype(np.float32)


@lru_cache(maxsize=1)
def get_separator() -> GuitarSeparator:
    return GuitarSeparator()


def analyze_guitar(
    stereo_44k: np.ndarray,
    *,
    separator: GuitarSeparator | None = None,
    embed: EmbedFn | None = None,
    threshold: float | None = None,
    max_windows: int | None = None,
) -> GuitarResult:
    """Separate the guitar stem, report its presence and embed it when present enough.

    Presence is always the raw stem/mix RMS ratio; the embedding is None below `threshold`
    (default: settings.guitar_presence_min).
    """
    mix = _check_stereo(stereo_44k)
    if threshold is None:
        threshold = get_settings().guitar_presence_min
    if rms(mix) == 0.0:
        return GuitarResult(presence=0.0, embedding=None)
    stem = (separator or get_separator()).separate(mix)
    presence = guitar_presence(stem, mix)
    if presence < threshold:
        return GuitarResult(presence=presence, embedding=None)
    if embed is None:
        from sonder.audio.clap import embed_audio as embed
    return GuitarResult(presence=presence, embedding=embed(resample_to_clap(stem), max_windows))
