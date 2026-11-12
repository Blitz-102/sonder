import re
import unicodedata
from collections.abc import Callable, Sequence
from typing import Any

from rapidfuzz import fuzz, utils

_BRACKETS = re.compile(r"\s*[(\[][^)\]]*[)\]]")
_FEAT = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+.*$")
_DASH_SUFFIX = re.compile(r"\s+[-–—]\s+.*$")
_SPACES = re.compile(r"\s+")


def _fold(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_title(s: str) -> str:
    s = _fold(s)
    s = _BRACKETS.sub("", s)
    s = _DASH_SUFFIX.sub("", s)
    s = _FEAT.sub("", s)
    return _SPACES.sub(" ", s).strip()


def normalize_artist(s: str) -> str:
    s = _fold(s).strip()
    s = _FEAT.sub("", s)
    s = re.sub(r"^the\s+", "", s)
    return _SPACES.sub(" ", s).strip()


def match_score(q_artist: str, q_title: str, c_artist: str, c_title: str) -> float:
    artist = fuzz.token_set_ratio(
        normalize_artist(q_artist), normalize_artist(c_artist), processor=utils.default_process
    )
    qt, ct = normalize_title(q_title), normalize_title(c_title)
    title = max(
        fuzz.ratio(qt, ct, processor=utils.default_process),
        fuzz.token_sort_ratio(qt, ct, processor=utils.default_process),
    )
    return float(min(artist, title))


def _default_key(c: Any) -> tuple[str, str]:
    return (c.artist, c.title)


def best_match[T](
    artist: str,
    title: str,
    candidates: Sequence[T],
    key: Callable[[T], tuple[str, str]] = _default_key,
    threshold: float = 85.0,
) -> T | None:
    best: T | None = None
    best_score = -1.0
    for c in candidates:
        c_artist, c_title = key(c)
        score = match_score(artist, title, c_artist, c_title)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= threshold else None
