"""Brute-force cosine search over the stored vectors."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Self

import numpy as np

from sonder.index.db import VECTOR_KINDS, connect, load_vectors


def _normalise_rows(matrix: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1)
    keep = np.isfinite(norms) & (norms > 0)
    matrix = matrix[keep] / norms[keep, None]
    return ids[keep].astype(np.int64, copy=False), np.ascontiguousarray(matrix, dtype=np.float32)


class VectorIndex:
    """L2-normalised matrices per vector kind; cosine similarity is a dot product."""

    def __init__(self, data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self._data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for kind in VECTOR_KINDS:
            ids, matrix = data.get(kind, (np.zeros(0, dtype=np.int64), np.zeros((0, 0))))
            self._data[kind] = _normalise_rows(np.asarray(matrix, dtype=np.float32), ids)

    @classmethod
    def load(cls, conn_or_path: sqlite3.Connection | Path | str) -> Self:
        if isinstance(conn_or_path, sqlite3.Connection):
            return cls({kind: load_vectors(conn_or_path, kind) for kind in VECTOR_KINDS})
        conn = connect(Path(conn_or_path))
        try:
            return cls({kind: load_vectors(conn, kind) for kind in VECTOR_KINDS})
        finally:
            conn.close()

    def size(self, kind: str = "vibe") -> int:
        return int(self._data[kind][0].size)

    def __len__(self) -> int:
        return self.size("vibe")

    def top_k(
        self,
        query: np.ndarray,
        kind: str = "vibe",
        k: int = 10,
        exclude_ids: Iterable[int] = (),
    ) -> list[tuple[int, float]]:
        """The k most similar tracks as (track_id, cosine), best first (ties by id ascending)."""
        if kind not in self._data:
            raise ValueError(f"unknown vector kind {kind!r}")
        ids, matrix = self._data[kind]
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {q.shape}")
        if k <= 0 or ids.size == 0:
            return []
        if q.size != matrix.shape[1]:
            raise ValueError(f"query has dim {q.size}, index has dim {matrix.shape[1]}")
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm == 0.0:
            return []
        q = q / norm

        excluded = np.fromiter(exclude_ids, dtype=np.int64)
        if excluded.size:
            keep = ~np.isin(ids, excluded)
            ids, matrix = ids[keep], matrix[keep]
            if ids.size == 0:
                return []
        scores = matrix @ q
        if k < ids.size:
            # Everything at or above the k-th best score, so ties at the boundary resolve by id.
            kth = np.partition(scores, ids.size - k)[ids.size - k]
            cand = np.flatnonzero(scores >= kth)
        else:
            cand = np.arange(ids.size)
        order = np.lexsort((ids[cand], -scores[cand]))[:k]
        chosen = cand[order]
        return [(int(ids[i]), float(scores[i])) for i in chosen]
