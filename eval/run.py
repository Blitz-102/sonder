"""Evaluation metrics for the Jamendo set: clip-vs-full agreement, tag precision@10, presence AUC.

Run from backend/ after eval\\build.py:
    uv run python ..\\eval\\run.py [--db PATH] [--out PATH] [--seed N]

Pure numpy/scipy; no ML libraries. Writes numbers only to eval/results.md. Judging the numbers
against the Gate B targets is done by a person, not by this script.
"""

import argparse
import datetime
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jamendo  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent / "results.md"
RECALL_K = 5
PRECISION_K = 10
MIN_TRACKS = 200  # below this the numbers are only a smoke test
MOOD_PREFIX = "mood/theme---"
GENRE_PREFIX = "genre---"
# Approximate negatives for the presence AUC (a track also needs no guitar-like instrument tag).
NEGATIVE_TAGS = frozenset({"instrument---piano", "instrument---synthesizer", "genre---electronic"})
GENRE_TIE_NOISE = 1e-7  # far below the smallest gap between distinct Jaccard values


# ---- data -------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSet:
    track_ids: np.ndarray
    artist_ids: np.ndarray
    tags: list[set[str]]  # full 'category---name' strings
    clip_vibe: np.ndarray
    full_vibe: np.ndarray
    clip_guitar: np.ndarray
    presence: np.ndarray
    versions: dict[str, str]
    dropped: int  # tracks stored but missing a vector (e.g. a silent clip has no guitar vector)

    @property
    def n(self) -> int:
        return int(self.track_ids.shape[0])


def load_eval(conn: sqlite3.Connection) -> EvalSet:
    """Tracks that have all three vectors, aligned row for row."""
    matrices = {kind: jamendo.load_matrix(conn, kind) for kind in jamendo.VECTOR_KINDS}
    common = matrices[jamendo.VECTOR_KINDS[0]][0]
    for kind in jamendo.VECTOR_KINDS[1:]:
        common = np.intersect1d(common, matrices[kind][0])
    aligned = {}
    for kind, (ids, mat) in matrices.items():
        aligned[kind] = mat[np.searchsorted(ids, common)]  # ids are sorted by track_id

    meta = {
        r["track_id"]: (r["artist_id"], json.loads(r["tags_json"]))
        for r in conn.execute("SELECT track_id, artist_id, tags_json FROM tracks")
    }
    presence = dict(conn.execute("SELECT track_id, guitar_presence FROM presence").fetchall())
    versions = {
        r["kind"]: r["versions"]
        for r in conn.execute(
            "SELECT kind, group_concat(DISTINCT model_version) AS versions FROM vectors "
            "GROUP BY kind"
        )
    }
    ids = [int(t) for t in common]
    tags = [{f"{cat}---{name}" for cat, names in meta[t][1].items() for name in names} for t in ids]
    return EvalSet(
        track_ids=common,
        artist_ids=np.array([meta[t][0] for t in ids], dtype=np.int64),
        tags=tags,
        clip_vibe=aligned["clip_vibe"],
        full_vibe=aligned["full_vibe"],
        clip_guitar=aligned["clip_guitar"],
        presence=np.array([presence[t] for t in ids], dtype=np.float64),
        versions=versions,
        dropped=len(meta) - len(ids),
    )


# ---- clip vs full -----------------------------------------------------------------------


def recall_at_k(query: np.ndarray, gallery: np.ndarray, k: int = RECALL_K) -> float:
    """Share of queries whose own gallery row (same index) ranks in the top k by dot product.

    Rows beat the own row only when strictly more similar, so exact ties count in its favour.
    """
    sims = query @ gallery.T
    own = np.diag(sims)[:, None]
    return float(np.mean((sims > own).sum(axis=1) < k))


def _top_neighbours(x: np.ndarray, k: int) -> np.ndarray:
    sims = x @ x.T
    np.fill_diagonal(sims, -np.inf)
    return np.argsort(-sims, axis=1, kind="stable")[:, :k]


def neighbour_overlap(a: np.ndarray, b: np.ndarray, k: int = PRECISION_K) -> float:
    """Mean |top-k(a) & top-k(b)| / k over rows, each space excluding the row itself."""
    k = min(k, a.shape[0] - 1)
    if k < 1:
        raise ValueError("need at least 2 tracks")
    top_a, top_b = _top_neighbours(a, k), _top_neighbours(b, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(top_a, top_b, strict=True)]))


# ---- tag precision ----------------------------------------------------------------------


def tag_matrix(tag_sets: list[set[str]]) -> np.ndarray:
    vocab = {t: i for i, t in enumerate(sorted(set().union(*tag_sets)))}
    m = np.zeros((len(tag_sets), len(vocab)), dtype=np.float32)
    for row, tags in enumerate(tag_sets):
        for t in tags:
            m[row, vocab[t]] = 1.0
    return m


@dataclass(frozen=True)
class Pool:
    """Queries and candidates for one tag category (tracks with >= 1 tag in it)."""

    idx: np.ndarray  # indices into the eval set
    eligible: np.ndarray  # (n, n) bool: candidate j may be returned for query i
    hit: np.ndarray  # (n, n) bool: i and j share a tag in the category
    queries: np.ndarray  # (n,) bool: query i has at least one eligible candidate


def build_pool(category_tags: list[set[str]], artist_ids: np.ndarray) -> Pool:
    idx = np.array([i for i, s in enumerate(category_tags) if s], dtype=np.int64)
    if idx.size == 0:
        empty = np.zeros((0, 0), dtype=bool)
        return Pool(idx, empty, empty, np.zeros(0, dtype=bool))
    t = tag_matrix([category_tags[i] for i in idx])
    artists = artist_ids[idx]
    eligible = artists[:, None] != artists[None, :]  # also drops the query itself
    return Pool(idx, eligible, (t @ t.T) > 0, eligible.any(axis=1))


def precision_at_k(sub_sims: np.ndarray, pool: Pool, k: int = PRECISION_K) -> float | None:
    """Mean over queries of hits in the top min(k, eligible) eligible neighbours.

    `sub_sims` is the (n, n) similarity matrix restricted to pool.idx.
    """
    if not pool.queries.any():
        return None
    masked = np.where(pool.eligible, sub_sims, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    n_eligible = pool.eligible.sum(axis=1)
    kq = np.minimum(k, n_eligible)
    in_top = np.arange(order.shape[1])[None, :] < kq[:, None]
    hits = (np.take_along_axis(pool.hit, order, axis=1) & in_top).sum(axis=1)
    q = pool.queries
    return float(np.mean(hits[q] / kq[q]))


def random_precision(pool: Pool) -> float | None:
    """Exact expected precision of a random ranking (hit share among eligible candidates)."""
    if not pool.queries.any():
        return None
    q = pool.queries
    share = (pool.hit & pool.eligible).sum(axis=1)[q] / pool.eligible.sum(axis=1)[q]
    return float(np.mean(share))


def genre_similarity(genre_tags: list[set[str]], rng: np.random.Generator) -> np.ndarray:
    """Jaccard similarity of genre tag sets, ties broken at random per query."""
    g = tag_matrix(genre_tags).astype(np.float64)
    inter = g @ g.T
    sizes = g.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    jaccard = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return jaccard + rng.random(jaccard.shape) * GENRE_TIE_NOISE


# ---- presence AUC -----------------------------------------------------------------------


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """P(random positive scores above random negative), ties count half (Mann-Whitney U)."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    ranks = rankdata(np.concatenate([pos, neg]))
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def presence_groups(tags: list[set[str]]) -> tuple[np.ndarray, np.ndarray]:
    """(positive idx, negative idx). Negatives: piano/synth/electronic and no guitar-like tag."""
    pos, neg = [], []
    for i, s in enumerate(tags):
        if s & jamendo.GUITAR_TAGS:
            pos.append(i)
            continue
        guitar_like = any("guitar" in t for t in s if t.startswith("instrument---"))
        if not guitar_like and s & NEGATIVE_TAGS:
            neg.append(i)
    return np.array(pos, dtype=np.int64), np.array(neg, dtype=np.int64)


# ---- orchestration ----------------------------------------------------------------------


def evaluate(ev: EvalSet, seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sim_vibe = ev.clip_vibe @ ev.clip_vibe.T
    sim_full = ev.full_vibe @ ev.full_vibe.T
    sim_guitar = ev.clip_guitar @ ev.clip_guitar.T
    method_sims = {
        "vibe (clip)": sim_vibe,
        "vibe (full track)": sim_full,
        "guitar (clip)": sim_guitar,
        "blend 0.5 (clip)": 0.5 * (sim_vibe + sim_guitar),
    }

    clip_vs_full = {
        "recall@5": recall_at_k(ev.clip_vibe, ev.full_vibe),
        "recall@5 random": min(1.0, RECALL_K / ev.n),
        "overlap@10": neighbour_overlap(ev.clip_vibe, ev.full_vibe),
    }

    categories = {
        "mood/theme": [{t for t in s if t.startswith(MOOD_PREFIX)} for s in ev.tags],
        "guitar": [s & jamendo.GUITAR_TAGS for s in ev.tags],
    }
    genres = [{t for t in s if t.startswith(GENRE_PREFIX)} for s in ev.tags]
    precision: dict[str, Any] = {}
    for name, cat_tags in categories.items():
        pool = build_pool(cat_tags, ev.artist_ids)
        results: dict[str, float | None] = {"random": random_precision(pool)}
        sub = np.ix_(pool.idx, pool.idx)
        results["genre tags"] = precision_at_k(
            genre_similarity([genres[i] for i in pool.idx], rng), pool
        )
        for method, sims in method_sims.items():
            results[method] = precision_at_k(sims[sub], pool)
        precision[name] = {"queries": int(pool.queries.sum()), "methods": results}

    pos, neg = presence_groups(ev.tags)
    return {
        "n": ev.n,
        "dropped": ev.dropped,
        "versions": ev.versions,
        "clip_vs_full": clip_vs_full,
        "precision": precision,
        "presence_auc": {
            "auc": auc(ev.presence[pos], ev.presence[neg]),
            "n_positive": int(pos.size),
            "n_negative": int(neg.size),
        },
    }


# ---- report -----------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_vs_random(value: float | None, random: float | None) -> str:
    if value is None:
        return "n/a"
    if not random:
        return _fmt(value)
    return f"{value:.3f} ({value / random:.2f}x)"


def render(results: dict[str, Any], date: datetime.date | None = None) -> str:
    date = date or datetime.date.today()
    cvf = results["clip_vs_full"]
    versions = ", ".join(f"{k}={v}" for k, v in sorted(results["versions"].items()))
    lines = [
        "# sonder evaluation results",
        "",
        f"Date: {date.isoformat()}. Tracks: {results['n']} "
        f"({results['dropped']} dropped for a missing vector). Models: {versions}.",
        "",
        "## Clip vs full track",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| recall@5 (clip finds its own full track) | {_fmt(cvf['recall@5'])} |",
        f"| recall@5, random | {_fmt(cvf['recall@5 random'])} |",
        f"| neighbour overlap@10 | {_fmt(cvf['overlap@10'])} |",
        "",
        "## Tag precision@10",
        "",
        "Multiples in brackets are against the random row of the same column.",
        "",
    ]
    cats = list(results["precision"])
    queries = ", ".join(f"{c} {results['precision'][c]['queries']}" for c in cats)
    lines += [
        f"Queries: {queries}.",
        "",
        "| Method | " + " | ".join(cats) + " |",
        "| --- | " + " | ".join("---" for _ in cats) + " |",
    ]
    for method in results["precision"][cats[0]]["methods"]:
        cells = []
        for c in cats:
            methods = results["precision"][c]["methods"]
            value = methods[method]
            cells.append(
                _fmt(value) if method == "random" else _fmt_vs_random(value, methods["random"])
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    p = results["presence_auc"]
    lines += [
        "",
        "## Guitar presence AUC",
        "",
        "| AUC | Guitar-tagged | Piano/synth/electronic, no guitar |",
        "| --- | --- | --- |",
        f"| {_fmt(p['auc'])} | {p['n_positive']} | {p['n_negative']} |",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=None, help="eval store (default: settings)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    conn = jamendo.connect(args.db)
    try:
        ev = load_eval(conn)
    finally:
        conn.close()
    if ev.n < 2:
        parser.error("fewer than 2 tracks with all three vectors; run eval\\build.py first")
    if ev.n < MIN_TRACKS:
        print(
            f"WARNING: only {ev.n} tracks (< {MIN_TRACKS}); these numbers are a smoke test only.",
            file=sys.stderr,
        )
    text = render(evaluate(ev, args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
