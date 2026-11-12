"""Embed the Jamendo evaluation tracks: full-track vibe, 30 s middle-clip vibe, clip guitar stem.

Run from backend/ (needs the `ml` extra):
    uv run python ..\\eval\\build.py [--limit N] [--retry-failed]

Resumable: tracks already stored at the current model versions are skipped, and failures are
recorded and skipped on later runs unless --retry-failed is given.
"""

import argparse
import logging
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jamendo  # noqa: E402
from jamendo import JamendoTrack  # noqa: E402

from sonder.audio.clap import MAX_TRACK_WINDOWS  # noqa: E402
from sonder.audio.clap import MODEL_VERSION as VIBE_VERSION  # noqa: E402
from sonder.audio.guitar import GUITAR_MODEL_VERSION  # noqa: E402
from sonder.audio.load import load_audio  # noqa: E402

log = logging.getLogger("eval.build")

SAMPLE_RATE = 44_100  # htdemucs_6s rate; CLAP input is resampled from it
CLIP_SECONDS = 30
LOG_EVERY = 25

VERSIONS = {
    "full_vibe": VIBE_VERSION,
    "clip_vibe": VIBE_VERSION,
    "clip_guitar": GUITAR_MODEL_VERSION,
}


@dataclass(frozen=True)
class TrackAnalysis:
    full_vibe: np.ndarray
    clip_vibe: np.ndarray
    clip_guitar: np.ndarray | None  # None when the clip is silent
    guitar_presence: float


type Analyze = Callable[[np.ndarray, np.ndarray], TrackAnalysis]  # (full stereo, clip stereo)
type Loader = Callable[[Path, int, int], np.ndarray]


def middle_clip(
    stereo: np.ndarray, sr: int = SAMPLE_RATE, seconds: int = CLIP_SECONDS
) -> np.ndarray:
    """The central `seconds` of a (channels, n) array; the whole array if it is shorter."""
    n = stereo.shape[1]
    length = seconds * sr
    if n <= length:
        return stereo
    start = (n - length) // 2
    return stereo[:, start : start + length]


def make_analyzer() -> Analyze:
    """The real analysis. Heavy modules load here, models load once (cached singletons)."""
    from sonder.audio.clap import embed_audio
    from sonder.audio.guitar import analyze_guitar, get_separator, resample_to_clap

    def analyze(full: np.ndarray, clip: np.ndarray) -> TrackAnalysis:
        full_vibe = embed_audio(resample_to_clap(full), MAX_TRACK_WINDOWS)
        clip_vibe = embed_audio(resample_to_clap(clip), None)
        # threshold=0.0: always embed the stem; step 3.2 needs presence and vectors for all tracks.
        guitar = analyze_guitar(clip, separator=get_separator(), threshold=0.0, max_windows=None)
        return TrackAnalysis(full_vibe, clip_vibe, guitar.embedding, guitar.presence)

    return analyze


@dataclass
class Summary:
    built: int = 0
    skipped: int = 0
    failed: int = 0
    already_failed: int = 0

    def format(self) -> str:
        return (
            f"built {self.built}, skipped {self.skipped} (already done), failed {self.failed}, "
            f"already-failed-skipped {self.already_failed}"
        )


def build(
    conn: sqlite3.Connection,
    tracks: Iterable[JamendoTrack],
    analyze: Analyze,
    *,
    audio_root: Path | None = None,
    load: Loader = load_audio,
    retry_failed: bool = False,
) -> Summary:
    summary = Summary()
    failed = jamendo.failed_ids(conn)
    tracks = list(tracks)
    start = time.monotonic()
    for i, track in enumerate(tracks, 1):
        if jamendo.is_done(conn, track.track_id, VERSIONS):
            summary.skipped += 1
            continue
        if track.track_id in failed and not retry_failed:
            summary.already_failed += 1
            continue
        try:
            stereo = load(jamendo.audio_path(track, audio_root), SAMPLE_RATE, 2)
            result = analyze(stereo, middle_clip(stereo))
            vectors = {"full_vibe": result.full_vibe, "clip_vibe": result.clip_vibe}
            if result.clip_guitar is not None:
                vectors["clip_guitar"] = result.clip_guitar
            jamendo.save_track_result(conn, track, vectors, VERSIONS, result.guitar_presence)
        except Exception as e:  # never stop the batch on one bad track
            jamendo.record_failure(conn, track.track_id, f"{type(e).__name__}: {e}")
            summary.failed += 1
            log.warning("failed track %d: %s: %s", track.track_id, type(e).__name__, e)
            continue
        summary.built += 1
        if summary.built % LOG_EVERY == 0:
            rate = (time.monotonic() - start) / max(summary.built, 1)
            remaining = (len(tracks) - i) * rate
            log.info(
                "built %d (%d/%d seen), ~%.0f min left",
                summary.built,
                i,
                len(tracks),
                remaining / 60,
            )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process at most N tracks")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tracks = jamendo.available_tracks(jamendo.load_tracks(jamendo.tags_tsv()))
    if not tracks:
        parser.error("no Jamendo audio on disk; run eval\\fetch_jamendo.py first")
    if args.limit is not None:
        tracks = tracks[: args.limit]
    conn = jamendo.connect()
    try:
        summary = build(conn, tracks, make_analyzer(), retry_failed=args.retry_failed)
    finally:
        conn.close()
    print(summary.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
