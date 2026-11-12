"""Ingest tracks into the index: Deezer ids (30 s previews) or local audio files.

Run from backend/ (use the interpreter that has the `ml` extra installed):
    python ..\\scripts\\ingest.py 3135556 C:\\music\\a.mp3 --from-file ids.txt [--retry-failed]
        [--limit N] [--guitar-threshold X]

Previews are downloaded to a temp file, analysed and deleted again (ground rule 1). Preview URLs
are never stored or logged (ground rule 2). Item failures are recorded and skipped on later runs
unless --retry-failed is given; the exit code is non-zero only for usage errors.
"""

import argparse
import logging
import re
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlparse

import httpx
import numpy as np

from sonder.audio.clap import MAX_TRACK_WINDOWS
from sonder.audio.clap import MODEL_VERSION as VIBE_MODEL_VERSION
from sonder.audio.load import load_audio
from sonder.config import get_settings
from sonder.index import db
from sonder.index.db import Analysis, TrackRecord
from sonder.sources.deezer import DeezerTrack
from sonder.sources.itunes import ItunesTrack
from sonder.sources.match import best_match

log = logging.getLogger("sonder.ingest")

STEREO_SR = 44_100
MONO_SR = 22_050

type Analyze = Callable[[np.ndarray, np.ndarray, int | None, float | None], Analysis]
type Loader = Callable[[Path, int, int], np.ndarray]


class DeezerLike(Protocol):
    def get_track(self, track_id: int) -> DeezerTrack | None: ...

    def fresh_preview_url(self, track_id: int) -> str | None: ...


class ItunesLike(Protocol):
    def search_song(
        self, artist: str, title: str, limit: int = 10, country: str = "US"
    ) -> list[ItunesTrack]: ...


class IngestFailure(Exception):  # noqa: N818
    """An expected, already-readable failure (reason is stored as is)."""


@dataclass
class Summary:
    ingested: int = 0
    skipped: int = 0
    failed: int = 0
    already_failed: int = 0

    def format(self) -> str:
        return (
            f"ingested {self.ingested}, skipped {self.skipped} (already done), "
            f"failed {self.failed}, already-failed-skipped {self.already_failed}"
        )


Outcome = Literal["ingested", "skipped"]

_CONTENT_TYPE_SUFFIX = {
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".m4a",
}


def _audio_suffix(url: str, headers: Mapping[str, str]) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix:
        return suffix
    content_type = headers.get("content-type", "").split(";")[0].strip().lower()
    return _CONTENT_TYPE_SUFFIX.get(content_type, ".mp3")


def classify(item: str) -> Literal["deezer", "local"]:
    """All ASCII digits -> Deezer track id, anything else -> local file path."""
    return "deezer" if re.fullmatch(r"[0-9]+", item.strip()) else "local"


def read_items(args: argparse.Namespace) -> list[str]:
    """Positional items plus --from-file lines; blanks and # comments dropped, de-duplicated."""
    raw: list[str] = list(args.items or [])
    if args.from_file is not None:
        for line in Path(args.from_file).read_text(encoding="utf-8-sig").splitlines():
            raw.append(line)
    items: list[str] = []
    for r in raw:
        item = r.strip()
        if item and not item.startswith("#"):
            items.append(item)
    return list(dict.fromkeys(items))


def parse_filename(path: Path) -> tuple[str, str]:
    """('Artist', 'Title') from 'Artist - Title.ext'; otherwise ('Unknown', stem)."""
    stem = path.stem
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        if artist.strip() and title.strip():
            return artist.strip(), title.strip()
    return "Unknown", stem


def describe_error(e: BaseException) -> str:
    return db.scrub_urls(f"{type(e).__name__}: {e}")


@dataclass
class Pipeline:
    conn: sqlite3.Connection
    deezer: DeezerLike
    itunes: ItunesLike
    http: httpx.Client
    analyze: Analyze
    tmp_dir: Path
    retry_failed: bool = False
    load: Loader = load_audio

    # -- Deezer -------------------------------------------------------------------------

    def ingest_deezer(self, deezer_id: int) -> Outcome:
        if db.is_finished(self.conn, deezer_id=deezer_id, vibe_model_version=VIBE_MODEL_VERSION):
            return "skipped"
        track = self.deezer.get_track(deezer_id)
        if track is None:
            raise IngestFailure("track not found")
        source = "deezer"
        url = self.deezer.fresh_preview_url(deezer_id)
        if url is None:
            found = best_match(
                track.artist, track.title, self.itunes.search_song(track.artist, track.title)
            )
            if found is not None and found.preview_url:
                url = found.preview_url
                source = "itunes"
        if url is None:
            raise IngestFailure("no preview")

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        resp = self.http.get(url, follow_redirects=True)
        resp.raise_for_status()
        fd, name = tempfile.mkstemp(suffix=_audio_suffix(url, resp.headers), dir=self.tmp_dir)
        path = Path(name)
        try:
            with open(fd, "wb") as f:
                f.write(resp.content)
            stereo = self.load(path, STEREO_SR, 2)
            mono = self.load(path, MONO_SR, 1)
        finally:
            path.unlink(missing_ok=True)
        analysis = self.analyze(stereo, mono, None, track.bpm)
        track_id = db.upsert_track(
            self.conn,
            TrackRecord(
                deezer_id=track.id,
                isrc=track.isrc,
                title=track.title,
                artist=track.artist,
                album=track.album_title or None,
                cover_url=track.cover_url,
                duration=float(track.duration) if track.duration > 0 else None,
                source=source,
                local_path=None,
            ),
        )
        db.save_analysis(self.conn, track_id, analysis)
        return "ingested"

    # -- Local files --------------------------------------------------------------------

    def ingest_local(self, item: str) -> Outcome:
        path = Path(item).expanduser().resolve()
        key = str(path)
        if db.is_finished(self.conn, local_path=key, vibe_model_version=VIBE_MODEL_VERSION):
            return "skipped"
        if not path.is_file():
            raise IngestFailure("file not found")
        stereo = self.load(path, STEREO_SR, 2)
        mono = self.load(path, MONO_SR, 1)
        analysis = self.analyze(stereo, mono, MAX_TRACK_WINDOWS, None)
        artist, title = parse_filename(path)
        track_id = db.upsert_track(
            self.conn,
            TrackRecord(
                deezer_id=None,
                isrc=None,
                title=title,
                artist=artist,
                album=None,
                cover_url=None,
                duration=stereo.shape[1] / STEREO_SR,
                source="local",
                local_path=key,
            ),
        )
        db.save_analysis(self.conn, track_id, analysis)
        return "ingested"

    # -- Batch --------------------------------------------------------------------------

    def key_for(self, item: str) -> str:
        if classify(item) == "deezer":
            return str(int(item))
        return str(Path(item).expanduser().resolve())

    def process(self, item: str) -> Outcome:
        if classify(item) == "deezer":
            return self.ingest_deezer(int(item))
        return self.ingest_local(item)

    def run(self, items: Iterable[str]) -> Summary:
        summary = Summary()
        failed = db.failed_keys(self.conn)
        for item in items:
            try:
                key = self.key_for(item)
            except Exception as e:  # e.g. a path that cannot be resolved; never stop the batch
                key = item.strip()
                db.record_failure(self.conn, key, describe_error(e))
                summary.failed += 1
                log.warning("failed %s: %s", key, db.scrub_urls(describe_error(e)))
                continue
            if key in failed and not self.retry_failed:
                summary.already_failed += 1
                log.info("skipping %s (failed before; use --retry-failed)", key)
                continue
            try:
                outcome = self.process(item)
            except Exception as e:
                reason = str(e) if isinstance(e, IngestFailure) else describe_error(e)
                db.record_failure(self.conn, key, reason)
                summary.failed += 1
                log.warning("failed %s: %s", key, db.scrub_urls(reason))
                continue
            db.clear_failure(self.conn, key)
            if outcome == "ingested":
                summary.ingested += 1
                log.info("ingested %s", key)
            else:
                summary.skipped += 1
                log.info("skipped %s (already ingested)", key)
        return summary


def make_analyzer(guitar_threshold: float | None = None) -> Analyze:
    """The real analysis: CLAP vibe, librosa features and the htdemucs guitar stem.

    Heavy modules are imported here, and the models load once (cached singletons).
    """
    from sonder.audio.clap import embed_audio
    from sonder.audio.features import extract_features
    from sonder.audio.guitar import (
        GUITAR_MODEL_VERSION,
        analyze_guitar,
        get_separator,
        resample_to_clap,
    )

    def analyze(
        stereo: np.ndarray, mono: np.ndarray, max_windows: int | None, deezer_bpm: float | None
    ) -> Analysis:
        vibe = embed_audio(resample_to_clap(stereo), max_windows)
        feats = extract_features(mono, MONO_SR, deezer_bpm=deezer_bpm)
        guitar = analyze_guitar(
            stereo,
            separator=get_separator(),
            threshold=guitar_threshold,
            max_windows=max_windows,
        )
        return Analysis(
            tempo=feats.tempo,
            key=feats.key,
            mode=feats.mode,
            loudness_db=feats.loudness_db,
            guitar_presence=guitar.presence,
            vibe=vibe,
            guitar=guitar.embedding,
            vibe_model_version=VIBE_MODEL_VERSION,
            guitar_model_version=GUITAR_MODEL_VERSION if guitar.embedding is not None else None,
            features_model_version=db.FEATURES_VERSION,
        )

    return analyze


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("items", nargs="*", help="Deezer track ids or local audio file paths")
    parser.add_argument("--from-file", type=Path, default=None, help="one item per line")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="process at most N items")
    parser.add_argument("--guitar-threshold", type=float, default=None)
    return parser


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx logs every request URL at INFO ("HTTP Request: GET <url> ..."), and the Deezer preview
    # URL carries a signed token. Keep the transport loggers quiet (ground rule 2).
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    from sonder.sources.deezer import DeezerClient
    from sonder.sources.itunes import ItunesClient

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging()
    if args.from_file is not None and not args.from_file.is_file():
        parser.error(f"--from-file not found: {args.from_file}")
    items = read_items(args)
    if not items:
        parser.error("no items given (pass Deezer ids / file paths or --from-file)")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be >= 1")
        items = items[: args.limit]

    settings = get_settings()
    settings.ensure_dirs()
    conn = db.connect()
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as http:
            pipeline = Pipeline(
                conn=conn,
                deezer=DeezerClient(),
                itunes=ItunesClient(),
                http=http,
                analyze=make_analyzer(args.guitar_threshold),
                tmp_dir=settings.tmp_dir,
                retry_failed=args.retry_failed,
            )
            summary = pipeline.run(items)
    finally:
        conn.close()
    print(summary.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
