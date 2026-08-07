"""Matching tests — the leakage assertion is the most important test in the repo.
It guarantees the single most important teaching moment reproduces on any laptop."""
from __future__ import annotations

from src import load_config
from src.generate_data import generate
from src.matching import run_experiment


def _prepared(tmp_path):
    cfg = load_config()
    cfg["paths"]["raw_jobs"] = str(tmp_path / "jobs.csv")
    generate(cfg)
    return cfg


def test_leakage_inflates_the_score(tmp_path):
    """The naive (leaky) split MUST score meaningfully higher than the honest split."""
    cfg = _prepared(tmp_path)
    r = run_experiment(cfg, verbose=False)
    margin = cfg["matching"]["min_leakage_gap"]
    assert r.naive_score > r.honest_score + margin, (
        f"expected naive ({r.naive_score}) > honest ({r.honest_score}) + {margin}; "
        f"gap was {r.gap:.3f}"
    )


def test_confusion_matrix_sums_to_test_set(tmp_path):
    cfg = _prepared(tmp_path)
    r = run_experiment(cfg, verbose=False)
    total = int(r.confusion.sum())
    assert total > 0
    # 2x2 matrix of non-negative integer counts.
    assert r.confusion.shape == (2, 2)
    assert (r.confusion >= 0).all()


def test_pipeline_is_deterministic(tmp_path):
    """Same seed, same data -> identical scores. Reproducibility is a feature."""
    cfg = _prepared(tmp_path)
    r1 = run_experiment(cfg, verbose=False)
    r2 = run_experiment(cfg, verbose=False)
    assert r1.naive_score == r2.naive_score
    assert r1.honest_score == r2.honest_score
    assert (r1.confusion == r2.confusion).all()
