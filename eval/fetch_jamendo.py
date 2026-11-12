"""Download a few MTG-Jamendo audio-low archives (not all 100) and extract them.

Run from backend/:
    uv run python ..\\eval\\fetch_jamendo.py [--archives 8] [--keep-tars]

Clones github.com/MTG/mtg-jamendo-dataset into data/ if missing (plain clone, not a submodule),
then downloads the first N archives one at a time, verifies each SHA-256 and extracts it into
data/jamendo/audio-low. Safe to re-run: finished archives are skipped, partial downloads resume.
"""

import argparse
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jamendo  # noqa: E402

REPO_URL = "https://github.com/MTG/mtg-jamendo-dataset"
CHUNK = 1 << 20
DEFAULT_ARCHIVES = 8


class ArchiveError(Exception):
    """An archive could not be downloaded or failed verification."""


def ensure_repo(repo: Path) -> None:
    if repo.is_dir():
        return
    repo.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)


def _hash_file(path: Path) -> "hashlib._Hash":
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h


def download(client: httpx.Client, url: str, part: Path, expected_sha: str) -> None:
    """Stream `url` into `part` (resuming if it exists), then check the SHA-256."""
    part.parent.mkdir(parents=True, exist_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    hasher = _hash_file(part) if offset else hashlib.sha256()
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", url, headers=headers) as resp:
        if offset and resp.status_code == 200:  # server ignored Range: start over
            offset, hasher = 0, hashlib.sha256()
        elif resp.status_code == 416:  # already complete
            resp.read()
            resp.close()
            hasher = _hash_file(part)
            _verify(hasher, expected_sha, part)
            return
        else:
            resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + offset
        done = offset
        with open(part, "ab" if offset else "wb") as f:
            for chunk in resp.iter_bytes(CHUNK):
                f.write(chunk)
                hasher.update(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:8.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    _verify(hasher, expected_sha, part)


def _verify(hasher: "hashlib._Hash", expected_sha: str, part: Path) -> None:
    if hasher.hexdigest() != expected_sha:
        part.unlink(missing_ok=True)
        raise ArchiveError("SHA-256 mismatch (partial file deleted)")


def extract(tar_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")  # rejects absolute paths and '..' members
        return sum(1 for m in tar.getmembers() if m.isfile())


def fetch_archive(
    client: httpx.Client,
    name: str,
    sha: str,
    *,
    tar_dir: Path,
    dest: Path,
    keep_tar: bool = False,
    base_url: str = jamendo.ARCHIVE_URL,
) -> str:
    """Returns 'skipped' or 'fetched'. Raises ArchiveError / httpx errors on failure."""
    marker = tar_dir / f"{name}.done"
    if marker.exists():
        return "skipped"
    tar_path = tar_dir / name
    part = tar_dir / f"{name}.part"
    download(client, base_url.format(name=name), part, sha)
    part.replace(tar_path)
    try:
        extract(tar_path, dest)
    except (tarfile.TarError, OSError) as e:
        tar_path.unlink(missing_ok=True)
        raise ArchiveError(f"extract failed: {e}") from e
    marker.write_text(sha + "\n", encoding="utf-8")
    if not keep_tar:
        tar_path.unlink(missing_ok=True)
    return "fetched"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=int, default=DEFAULT_ARCHIVES)
    parser.add_argument("--keep-tars", action="store_true")
    args = parser.parse_args(argv)
    if args.archives < 1:
        parser.error("--archives must be >= 1")

    ensure_repo(jamendo.repo_dir())
    archives = jamendo.parse_sha_list(jamendo.sha_list())[: args.archives]
    tar_dir = jamendo.jamendo_dir() / "tars"
    dest = jamendo.audio_dir()
    failures = 0
    with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as client:
        for i, (name, sha) in enumerate(archives, 1):
            print(f"[{i}/{len(archives)}] {name}")
            try:
                result = fetch_archive(
                    client, name, sha, tar_dir=tar_dir, dest=dest, keep_tar=args.keep_tars
                )
            except (ArchiveError, httpx.HTTPError) as e:
                failures += 1
                print(f"  FAILED: {e}")
                continue
            print(f"  {result}")
    tracks = jamendo.available_tracks(jamendo.load_tracks(jamendo.tags_tsv()))
    guitar = sum(jamendo.has_guitar(t) for t in tracks)
    print(f"{len(tracks)} tagged tracks on disk ({guitar} with a guitar tag), {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
