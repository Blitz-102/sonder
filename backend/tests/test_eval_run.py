import datetime
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from sonder.config import REPO_ROOT

EVAL_DIR = REPO_ROOT / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import jamendo as jm  # noqa: E402


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_run", EVAL_DIR / "run.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_run"] = module
    spec.loader.exec_module(module)
    return module


er: Any = _load()


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _angles(degrees: list[float]) -> np.ndarray:
    rad = np.radians(degrees)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


# ---- clip vs full -----------------------------------------------------------------------


def test_recall_is_one_when_clip_equals_full() -> None:
    x = _unit(np.random.default_rng(0).standard_normal((20, 16)))
    assert er.recall_at_k(x, x, k=1) == 1.0


def test_recall_drops_when_gallery_is_shifted() -> None:
    eye = np.eye(10, dtype=np.float32)
    shifted = np.roll(eye, 1, axis=0)  # query i is now closest to gallery row i + 1
    assert er.recall_at_k(eye, shifted, k=1) == 0.0
    assert er.recall_at_k(eye, shifted, k=2) == 1.0


def test_overlap_identical_and_disjoint() -> None:
    a = _angles([0, 10, 90, 100])
    assert er.neighbour_overlap(a, a, k=1) == 1.0
    b = _angles([0, 90, 10, 100])  # every nearest neighbour changes
    assert er.neighbour_overlap(a, b, k=1) == 0.0


def test_overlap_clamps_k_to_available_neighbours() -> None:
    x = _unit(np.random.default_rng(1).standard_normal((3, 8)))
    assert er.neighbour_overlap(x, x, k=10) == 1.0  # k becomes 2
    with pytest.raises(ValueError):
        er.neighbour_overlap(x[:1], x[:1])


# ---- tag precision ----------------------------------------------------------------------


def test_pool_excludes_same_artist_and_self() -> None:
    pool = er.build_pool([{"x"}] * 4, np.array([1, 1, 2, 3]))
    assert pool.eligible[0].tolist() == [False, False, True, True]
    assert not pool.eligible.diagonal().any()


def test_pool_uses_only_tracks_with_tags() -> None:
    pool = er.build_pool([{"a"}, set(), {"a"}], np.array([1, 2, 3]))
    assert pool.idx.tolist() == [0, 2]
    assert pool.hit.shape == (2, 2)


SIMS = np.array(
    [[1, 0.5, 0.9, 0.1], [0.5, 1, 0.2, 0.3], [0.9, 0.2, 1, 0.4], [0.1, 0.3, 0.4, 1]],
    dtype=np.float32,
)
POOL_TAGS = [{"a"}, {"a"}, {"b"}, {"a"}]


def test_precision_exact_values() -> None:
    pool = er.build_pool(POOL_TAGS, np.arange(4))
    assert er.precision_at_k(SIMS, pool, k=1) == pytest.approx(0.25)
    assert er.precision_at_k(SIMS, pool, k=2) == pytest.approx(0.5)


def test_random_precision_is_exact_hit_share() -> None:
    pool = er.build_pool(POOL_TAGS, np.arange(4))
    assert er.random_precision(pool) == pytest.approx(0.5)


def test_random_precision_matches_simulation() -> None:
    pool = er.build_pool(POOL_TAGS, np.arange(4))
    rng = np.random.default_rng(0)
    runs = [er.precision_at_k(rng.random((4, 4)), pool, k=2) for _ in range(3000)]
    assert np.mean(runs) == pytest.approx(er.random_precision(pool), abs=0.03)


def test_precision_uses_fewer_neighbours_when_few_are_eligible() -> None:
    # queries 0 and 1 only have track 2 eligible (same artist), query 2 has two candidates
    pool = er.build_pool([{"a"}] * 3, np.array([1, 1, 2]))
    sims = np.random.default_rng(0).random((3, 3))
    assert er.precision_at_k(sims, pool, k=10) == 1.0
    assert er.random_precision(pool) == 1.0


def test_precision_is_none_without_any_valid_query() -> None:
    pool = er.build_pool([{"a"}] * 3, np.array([1, 1, 1]))
    assert er.precision_at_k(np.eye(3), pool) is None
    assert er.random_precision(pool) is None


def test_genre_baseline_prefers_shared_genres_with_negligible_noise() -> None:
    sims = er.genre_similarity([{"g1"}, {"g2"}, {"g1", "g3"}], np.random.default_rng(0))
    assert sims[0, 2] == pytest.approx(0.5, abs=1e-6)
    assert sims[0, 1] < 1e-6
    assert sims[0, 2] > sims[0, 1]


# ---- AUC --------------------------------------------------------------------------------


def test_auc_known_values() -> None:
    assert er.auc(np.array([2, 3]), np.array([0, 1])) == 1.0
    assert er.auc(np.array([0, 1]), np.array([2, 3])) == 0.0
    assert er.auc(np.array([1, 1]), np.array([1, 1])) == 0.5
    assert er.auc(np.array([1, 3]), np.array([2])) == 0.5


def test_auc_none_when_a_class_is_empty() -> None:
    assert er.auc(np.array([]), np.array([1.0])) is None
    assert er.auc(np.array([1.0]), np.array([])) is None


def test_presence_groups_exclude_guitar_like_tracks_from_negatives() -> None:
    tags = [
        {"instrument---electricguitar", "instrument---piano"},  # positive
        {"instrument---piano"},  # negative
        {"genre---electronic", "instrument---acousticbassguitar"},  # guitar-like: neither
        {"genre---rock"},  # neither
        {"instrument---synthesizer", "instrument---guitar"},  # positive
        {"genre---electronic"},  # negative
    ]
    pos, neg = er.presence_groups(tags)
    assert pos.tolist() == [0, 4]
    assert neg.tolist() == [1, 5]


# ---- evaluate / report ------------------------------------------------------------------

V = {"full_vibe": "clap", "clip_vibe": "clap", "clip_guitar": "demucs+clap"}


def _tags(i: int) -> set[str]:
    guitar = {0: "electricguitar", 1: "piano", 2: "synthesizer"}[i % 3]
    return {
        f"mood/theme---{'happy' if i % 2 else 'sad'}",
        f"genre---{'rock' if i % 2 else 'electronic'}",
        f"instrument---{guitar}",
    }


def _make_eval(n: int = 30, same_guitar: bool = False) -> Any:
    rng = np.random.default_rng(0)
    vibe = _unit(rng.standard_normal((n, 8)))
    guitar = vibe if same_guitar else _unit(rng.standard_normal((n, 8)))
    return er.EvalSet(
        track_ids=np.arange(n),
        artist_ids=np.arange(n) % 10,
        tags=[_tags(i) for i in range(n)],
        clip_vibe=vibe,
        full_vibe=vibe,
        clip_guitar=guitar,
        presence=np.array([0.8 if i % 3 == 0 else 0.1 for i in range(n)]),
        versions=V,
        dropped=0,
    )


def test_evaluate_structure_and_ranges() -> None:
    res = er.evaluate(_make_eval())
    assert res["clip_vs_full"]["recall@5"] == 1.0
    assert res["clip_vs_full"]["overlap@10"] == 1.0
    assert res["clip_vs_full"]["recall@5 random"] == pytest.approx(5 / 30)
    assert res["presence_auc"] == {"auc": 1.0, "n_positive": 10, "n_negative": 20}
    for cat in ("mood/theme", "guitar"):
        block = res["precision"][cat]
        assert block["queries"] > 0
        assert set(block["methods"]) == {
            "random",
            "genre tags",
            "vibe (clip)",
            "vibe (full track)",
            "guitar (clip)",
            "blend 0.5 (clip)",
        }
        assert all(0.0 <= v <= 1.0 for v in block["methods"].values())


def test_blend_equals_both_when_vibe_and_guitar_vectors_match() -> None:
    res = er.evaluate(_make_eval(same_guitar=True))
    for cat in ("mood/theme", "guitar"):
        m = res["precision"][cat]["methods"]
        assert m["blend 0.5 (clip)"] == pytest.approx(m["vibe (clip)"])
        assert m["guitar (clip)"] == pytest.approx(m["vibe (clip)"])


def test_evaluate_is_deterministic_for_a_seed() -> None:
    assert er.evaluate(_make_eval(), seed=3) == er.evaluate(_make_eval(), seed=3)


def test_render_has_all_sections_and_the_given_date() -> None:
    text = er.render(er.evaluate(_make_eval()), date=datetime.date(2026, 1, 2))
    assert "Date: 2026-01-02" in text
    for heading in ("## Clip vs full track", "## Tag precision@10", "## Guitar presence AUC"):
        assert heading in text
    assert "| random |" in text
    assert "x)" in text  # multiples of random shown for the other methods


def test_render_shows_na_for_missing_values() -> None:
    res = er.evaluate(_make_eval())
    res["presence_auc"]["auc"] = None
    assert "| n/a |" in er.render(res)


# ---- load + CLI -------------------------------------------------------------------------


def _store(tmp_path: Path, n: int = 30) -> Path:
    path = tmp_path / "eval.sqlite"
    conn = jm.connect(path)
    rng = np.random.default_rng(1)
    for i in range(n):
        tags: dict[str, list[str]] = {}
        for full in _tags(i):
            cat, _, name = full.partition("---")
            tags.setdefault(cat, []).append(name)
        track = jm.JamendoTrack(i + 1, i % 10, 1, f"{i % 100:02d}/{i}.mp3", 100.0, tags)
        vecs = {
            "full_vibe": _unit(rng.standard_normal((1, 8)))[0],
            "clip_vibe": _unit(rng.standard_normal((1, 8)))[0],
        }
        if i != n - 1:  # the last track has a silent clip: no guitar vector
            vecs["clip_guitar"] = _unit(rng.standard_normal((1, 8)))[0]
        jm.save_track_result(conn, track, vecs, V, 0.8 if i % 3 == 0 else 0.1)
    conn.close()
    return path


def test_load_eval_aligns_rows_and_counts_dropped(tmp_path: Path) -> None:
    conn = jm.connect(_store(tmp_path))
    ev = er.load_eval(conn)
    assert ev.n == 29 and ev.dropped == 1
    assert ev.track_ids.tolist() == list(range(1, 30))
    assert ev.clip_vibe.shape == ev.clip_guitar.shape == (29, 8)
    assert ev.versions["clip_guitar"] == "demucs+clap"
    assert ev.tags[0] == _tags(0)
    assert ev.presence[0] == pytest.approx(0.8)


def test_main_writes_results_and_warns_on_small_sets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "results.md"
    assert er.main(["--db", str(_store(tmp_path)), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "29 (1 dropped" in text
    assert "## Guitar presence AUC" in text
    assert "smoke test only" in capsys.readouterr().err


def test_main_errors_on_empty_store(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        er.main(["--db", str(tmp_path / "empty.sqlite"), "--out", str(tmp_path / "r.md")])


# ---- import hygiene ---------------------------------------------------------------------


def test_run_does_not_import_ml_libs() -> None:
    code = (
        "import importlib.util, sys\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "spec = importlib.util.spec_from_file_location('s', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "bad = [n for n in ('torch', 'transformers', 'demucs_infer') if n in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(EVAL_DIR / "run.py"), str(EVAL_DIR)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
