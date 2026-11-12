"""GATE A: how many seed songs have a playable 30 s preview (Deezer, then iTunes)?

Run from backend/: uv run python ../scripts/coverage.py [--seeds PATH] [--no-durations] [--limit N]
"""

import argparse
import csv
import re
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from sonder.config import get_settings
from sonder.sources.deezer import DeezerTrack
from sonder.sources.itunes import ItunesTrack
from sonder.sources.match import best_match

TARGET_PERCENT = 80.0
SHORT_CLIP_SECONDS = 29.0


@dataclass(frozen=True)
class Seed:
    artist: str
    title: str


@dataclass(frozen=True)
class SeedResult:
    seed: Seed
    deezer_id: int | None
    deezer_preview: bool
    itunes_rescued: bool
    duration: float | None
    reason: str  # "no_deezer_match" | "no_preview" | "ok"
    measure_error: str | None = None


@dataclass(frozen=True)
class Summary:
    n: int
    pct_deezer_match: float
    pct_deezer_preview: float
    pct_itunes_rescued: float
    pct_preview: float
    durations: list[float]
    short_clips: int
    failures_by_artist: dict[str, list[str]]
    measure_errors: list[str]


class DeezerLike(Protocol):
    def search(self, artist: str, title: str, limit: int = 10) -> list[DeezerTrack]: ...

    def fresh_preview_url(self, track_id: int) -> str | None: ...


class ItunesLike(Protocol):
    def search_song(
        self, artist: str, title: str, limit: int = 10, country: str = "US"
    ) -> list[ItunesTrack]: ...


Measure = Callable[[str], float | None]


def read_seeds(path: Path) -> list[Seed]:
    seeds: list[Seed] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            artist = (row.get("artist") or "").strip()
            title = (row.get("title") or "").strip()
            if artist and title:
                seeds.append(Seed(artist, title))
    return seeds


def measure_duration(
    url: str, tmp_dir: Path | None = None, http: httpx.Client | None = None
) -> float:
    """Download the clip to a temp file, read its duration, and always delete the file."""
    import librosa

    directory = tmp_dir if tmp_dir is not None else get_settings().tmp_dir
    directory.mkdir(parents=True, exist_ok=True)
    client = http if http is not None else httpx.Client(timeout=30.0, follow_redirects=True)
    fd, name = tempfile.mkstemp(suffix=".audio", dir=directory)
    path = Path(name)
    try:
        with open(fd, "wb") as f:
            resp = client.get(url)
            resp.raise_for_status()
            f.write(resp.content)
        return float(librosa.get_duration(path=str(path)))
    finally:
        path.unlink(missing_ok=True)


def check_seed(
    seed: Seed,
    deezer: DeezerLike,
    itunes: ItunesLike,
    measure: Measure | None,
) -> SeedResult:
    deezer_id: int | None = None
    deezer_url: str | None = None
    match = best_match(seed.artist, seed.title, deezer.search(seed.artist, seed.title))
    if match is not None:
        deezer_id = match.id
        deezer_url = deezer.fresh_preview_url(match.id)

    url = deezer_url
    itunes_rescued = False
    if url is None:
        found = best_match(seed.artist, seed.title, itunes.search_song(seed.artist, seed.title))
        if found is not None and found.preview_url:
            url = found.preview_url
            itunes_rescued = True

    if url is None:
        reason = "no_deezer_match" if deezer_id is None else "no_preview"
        return SeedResult(seed, deezer_id, False, False, None, reason)

    duration: float | None = None
    measure_error: str | None = None
    if measure is not None:
        try:
            duration = measure(url)
        except Exception as e:
            # The seed still has a preview; record why the clip could not be measured.
            measure_error = _describe_error(e)
    return SeedResult(
        seed, deezer_id, deezer_url is not None, itunes_rescued, duration, "ok", measure_error
    )


def _describe_error(e: Exception) -> str:
    """Exception class and message, with URLs removed (preview URLs must not be recorded)."""
    return f"{type(e).__name__}: {re.sub(r'https?://\S+', '<url>', str(e))}"


def _pct(count: int, n: int) -> float:
    return 100.0 * count / n if n else 0.0


def summarize(results: list[SeedResult]) -> Summary:
    n = len(results)
    failures: dict[str, list[str]] = defaultdict(list)
    for r in results:
        if r.reason != "ok":
            failures[r.seed.artist].append(r.seed.title)
    durations = [r.duration for r in results if r.duration is not None]
    return Summary(
        n=n,
        pct_deezer_match=_pct(sum(r.deezer_id is not None for r in results), n),
        pct_deezer_preview=_pct(sum(r.deezer_preview for r in results), n),
        pct_itunes_rescued=_pct(sum(r.itunes_rescued for r in results), n),
        pct_preview=_pct(sum(r.reason == "ok" for r in results), n),
        durations=durations,
        short_clips=sum(d < SHORT_CLIP_SECONDS for d in durations),
        failures_by_artist=dict(failures),
        measure_errors=[
            f"{r.seed.artist} - {r.seed.title}: {r.measure_error}"
            for r in results
            if r.measure_error is not None
        ],
    )


def format_report(s: Summary) -> str:
    lines = [
        f"Seeds: {s.n}",
        f"Matched on Deezer:      {s.pct_deezer_match:5.1f}%",
        f"Deezer preview:         {s.pct_deezer_preview:5.1f}%",
        f"Rescued by iTunes:      {s.pct_itunes_rescued:5.1f}%",
        f"Total with a preview:   {s.pct_preview:5.1f}%",
    ]
    if s.durations:
        lines.append(
            f"Clip duration (s): min {min(s.durations):.1f}, "
            f"median {statistics.median(s.durations):.1f}, "
            f"mean {statistics.fmean(s.durations):.1f}, max {max(s.durations):.1f} "
            f"({len(s.durations)} measured, {s.short_clips} under {SHORT_CLIP_SECONDS:.0f} s)"
        )
    if s.measure_errors:
        lines.append(f"Duration measurement failed for {len(s.measure_errors)} clip(s):")
        lines.extend(f"  {e}" for e in s.measure_errors)
    if s.failures_by_artist:
        lines.append("Failing seeds by artist:")
        for artist in sorted(s.failures_by_artist):
            lines.append(f"  {artist}: {', '.join(s.failures_by_artist[artist])}")
    verdict = "PASS" if s.pct_preview >= TARGET_PERCENT else "FAIL"
    lines.append(
        f"GATE A: {verdict} ({s.pct_preview:.1f}% with a preview, target {TARGET_PERCENT:.0f}%)"
    )
    return "\n".join(lines)


def write_results_csv(path: Path, results: list[SeedResult]) -> None:
    """Per-seed results. Contains no preview URLs (ground rule 2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "artist",
                "title",
                "deezer_id",
                "deezer_preview",
                "itunes_rescued",
                "duration",
                "reason",
                "measure_error",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.seed.artist,
                    r.seed.title,
                    r.deezer_id if r.deezer_id is not None else "",
                    r.deezer_preview,
                    r.itunes_rescued,
                    f"{r.duration:.2f}" if r.duration is not None else "",
                    r.reason,
                    r.measure_error or "",
                ]
            )


def main() -> None:
    from sonder.sources.deezer import DeezerClient
    from sonder.sources.itunes import ItunesClient

    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=settings.data_dir / "seeds.csv")
    parser.add_argument("--no-durations", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.seeds.exists():
        raise SystemExit(f"Seeds file not found: {args.seeds} (CSV with an artist,title header)")
    seeds = read_seeds(args.seeds)
    if args.limit is not None:
        seeds = seeds[: args.limit]

    deezer = DeezerClient()
    itunes = ItunesClient()
    results: list[SeedResult] = []
    with httpx.Client(timeout=30.0, follow_redirects=True) as http:
        measure: Measure | None = (
            None if args.no_durations else lambda url: measure_duration(url, http=http)
        )
        for i, seed in enumerate(seeds, 1):
            r = check_seed(seed, deezer, itunes, measure)
            results.append(r)
            print(f"[{i}/{len(seeds)}] {seed.artist} - {seed.title}: {r.reason}", flush=True)

    out = settings.data_dir / "coverage_results.csv"
    write_results_csv(out, results)
    print()
    print(format_report(summarize(results)))
    print(f"Per-seed results: {out}")


if __name__ == "__main__":
    main()
