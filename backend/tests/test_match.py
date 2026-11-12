import pytest

from sonder.sources.deezer import DeezerTrack
from sonder.sources.itunes import ItunesTrack
from sonder.sources.match import best_match, match_score, normalize_artist, normalize_title


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Creep", "creep"),
        ("Creep (feat. Someone)", "creep"),
        ("Creep feat. Someone", "creep"),
        ("Creep ft. Someone Else", "creep"),
        ("Creep featuring Someone", "creep"),
        ("Hotel California - Remastered 2011", "hotel california"),
        ("Hotel California (Remastered)", "hotel california"),
        ("Wish You Were Here - Live at Wembley", "wish you were here"),
        ("Song - Radio Edit", "song"),
        ("Song - Mono", "song"),
        ("Song [Deluxe Edition]", "song"),
        ("Café del Mar", "cafe del mar"),
        ("  Spaced   Out  ", "spaced out"),
        ("Self-Esteem", "self-esteem"),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("The Beatles", "beatles"),
        ("Beyoncé", "beyonce"),
        ("Artist feat. Guest", "artist"),
        ("Theory of a Deadman", "theory of a deadman"),
    ],
)
def test_normalize_artist(raw: str, expected: str) -> None:
    assert normalize_artist(raw) == expected


def _dz(track_id: int, artist: str, title: str) -> DeezerTrack:
    return DeezerTrack(track_id, title, title, artist, "", None, 200, None, None, 0)


def test_score_is_high_for_same_song_different_decoration() -> None:
    score = match_score(
        "Pink Floyd", "Wish You Were Here", "Pink Floyd", "Wish You Were Here - 2011 Remaster"
    )
    assert score >= 95


def test_picks_right_candidate_among_near_duplicates() -> None:
    candidates = [
        _dz(1, "Radiohead", "Creep (Acoustic)"),
        _dz(2, "Karaoke Band", "Creep"),
        _dz(3, "Radiohead", "Creep - Live at Glastonbury"),
        _dz(4, "Radiohead", "Creep"),
        _dz(5, "Radiohead", "Creeping"),
    ]
    match = best_match("Radiohead", "Creep", candidates)
    assert match is not None
    assert match.id in (1, 3, 4)
    # An exact title match must score at least as well as decorated variants.
    assert match_score("Radiohead", "Creep", "Radiohead", "Creep") == 100


def test_wrong_artist_is_rejected() -> None:
    assert best_match("Radiohead", "Creep", [_dz(1, "TLC", "Creep")]) is None


def test_weak_title_is_rejected() -> None:
    assert best_match("Radiohead", "Creep", [_dz(1, "Radiohead", "Karma Police")]) is None


def test_empty_candidates() -> None:
    assert best_match("a", "b", []) is None


def test_threshold_is_configurable() -> None:
    cands = [_dz(1, "Radiohead", "Creeps")]
    assert best_match("Radiohead", "Creep", cands, threshold=99.0) is None
    assert best_match("Radiohead", "Creep", cands, threshold=80.0) is not None


def test_works_with_itunes_shaped_candidates() -> None:
    cands = [
        ItunesTrack(1, "Karma Police", "Radiohead", "OK Computer", None, 1, None),
        ItunesTrack(2, "Creep - Remastered", "Radiohead", "Pablo Honey", "u", 1, None),
    ]
    match = best_match("Radiohead", "Creep", cands)
    assert match is not None
    assert match.track_id == 2


def test_custom_key() -> None:
    cands = [("The Beatles", "Let It Be"), ("Other", "Let It Be")]
    assert best_match("Beatles", "Let It Be", cands, key=lambda c: c) == cands[0]
