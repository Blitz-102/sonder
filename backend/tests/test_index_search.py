import sqlite3
from pathlib import Path

import numpy as np
import pytest

from sonder.index import db
from sonder.index.db import Analysis, TrackRecord
from sonder.index.search import VectorIndex


def brute_force(
    ids: np.ndarray, matrix: np.ndarray, query: np.ndarray, k: int, exclude: set[int]
) -> list[tuple[int, float]]:
    mat = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    q = query / np.linalg.norm(query)
    scored = [
        (int(i), float(s)) for i, s in zip(ids, mat @ q, strict=True) if int(i) not in exclude
    ]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:k]


def make_index(
    vibe: np.ndarray, guitar: tuple[np.ndarray, np.ndarray] | None = None
) -> VectorIndex:
    data = {"vibe": (np.arange(1, len(vibe) + 1, dtype=np.int64), vibe)}
    if guitar is not None:
        data["guitar"] = guitar
    return VectorIndex(data)


@pytest.fixture
def random_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return np.arange(1, 201, dtype=np.int64), rng.standard_normal((200, 512)).astype(np.float32)


@pytest.mark.parametrize("k", [1, 5, 10, 199])
def test_matches_brute_force(random_data: tuple[np.ndarray, np.ndarray], k: int) -> None:
    ids, matrix = random_data
    index = make_index(matrix)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.standard_normal(512).astype(np.float32)
        got = index.top_k(q, k=k)
        want = brute_force(ids, matrix, q, k, set())
        assert [i for i, _ in got] == [i for i, _ in want]
        np.testing.assert_allclose([s for _, s in got], [s for _, s in want], atol=1e-5)


def test_exclusion(random_data: tuple[np.ndarray, np.ndarray]) -> None:
    ids, matrix = random_data
    index = make_index(matrix)
    q = matrix[3]
    top = index.top_k(q, k=3)
    assert top[0][0] == 4
    excluded = {4, top[1][0]}
    got = index.top_k(q, k=10, exclude_ids=iter(excluded))
    want = brute_force(ids, matrix, q, 10, excluded)
    assert [i for i, _ in got] == [i for i, _ in want]
    assert not excluded & {i for i, _ in got}
    assert index.top_k(q, k=5, exclude_ids=range(1, 500)) == []


def test_k_larger_than_n_and_non_positive_k() -> None:
    rng = np.random.default_rng(2)
    index = make_index(rng.standard_normal((5, 8)).astype(np.float32))
    q = rng.standard_normal(8).astype(np.float32)
    assert len(index.top_k(q, k=50)) == 5
    assert len(index.top_k(q, k=5)) == 5
    assert index.top_k(q, k=0) == []
    assert index.top_k(q, k=-3) == []


def test_empty_index_and_zero_query() -> None:
    empty = VectorIndex({})
    assert len(empty) == 0 and empty.size("guitar") == 0
    assert empty.top_k(np.ones(4, dtype=np.float32)) == []
    assert empty.top_k(np.ones(4, dtype=np.float32), kind="guitar") == []
    index = make_index(np.eye(4, dtype=np.float32))
    assert index.top_k(np.zeros(4, dtype=np.float32)) == []


def test_wrong_dim_and_kind() -> None:
    index = make_index(np.eye(4, dtype=np.float32))
    with pytest.raises(ValueError, match="dim"):
        index.top_k(np.ones(5, dtype=np.float32))
    with pytest.raises(ValueError):
        index.top_k(np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="kind"):
        index.top_k(np.ones(4, dtype=np.float32), kind="bass")


def test_zero_norm_rows_dropped_and_scores_are_cosine() -> None:
    matrix = np.array([[0, 0, 0], [3, 0, 0], [0, 2, 0], [1, 1, 0]], dtype=np.float32)
    index = make_index(matrix)
    assert len(index) == 3
    got = index.top_k(np.array([10, 0, 0], dtype=np.float32), k=10)
    assert [i for i, _ in got] == [2, 4, 3]
    np.testing.assert_allclose([s for _, s in got], [1.0, 2**-0.5, 0.0], atol=1e-6)


def test_tie_order_is_id_ascending() -> None:
    v = np.array([1.0, 0.0], dtype=np.float32)
    w = np.array([0.0, 1.0], dtype=np.float32)
    matrix = np.stack([w, v, w, v, v])  # ids 1..5; 2, 4, 5 tie for best
    index = make_index(matrix)
    assert [i for i, _ in index.top_k(v, k=5)] == [2, 4, 5, 1, 3]
    # Ties straddling the k boundary still resolve by id.
    assert [i for i, _ in index.top_k(v, k=2)] == [2, 4]
    assert [i for i, _ in index.top_k(v, k=1)] == [2]


def test_guitar_matrix_only_contains_tracks_with_guitar_vectors(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "i.sqlite")
    rng = np.random.default_rng(3)
    ids = []
    for n in range(3):
        t = db.upsert_track(
            conn,
            TrackRecord(n + 1, None, f"t{n}", "a", None, None, 30.0, "deezer", None),
        )
        ids.append(t)
        has_guitar = n != 1
        db.save_analysis(
            conn,
            t,
            Analysis(
                tempo=None,
                key=None,
                mode=None,
                loudness_db=None,
                guitar_presence=0.3,
                vibe=rng.standard_normal(16).astype(np.float32),
                guitar=rng.standard_normal(16).astype(np.float32) if has_guitar else None,
                guitar_model_version="g" if has_guitar else None,
            ),
        )
    for source in (conn, tmp_path / "i.sqlite", str(tmp_path / "i.sqlite")):
        index = VectorIndex.load(source)
        assert index.size("vibe") == 3 and len(index) == 3
        assert index.size("guitar") == 2
        q = rng.standard_normal(16).astype(np.float32)
        assert {i for i, _ in index.top_k(q, "vibe", k=10)} == set(ids)
        assert {i for i, _ in index.top_k(q, "guitar", k=10)} == {ids[0], ids[2]}
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_loaded_matrices_are_unit_norm(random_data: tuple[np.ndarray, np.ndarray]) -> None:
    _, matrix = random_data
    index = make_index(matrix)
    _, normalised = index._data["vibe"]
    assert normalised.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(normalised, axis=1), 1.0, atol=1e-5)
