"""Quality-scorecard tests. Deterministic: fixed frames and a fixed 'today'."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.quality import WEIGHTS, score_dataframe


def _clean_frame():
    # Two distinct, complete, recent, consistently-located, currency-stamped posts.
    return pd.DataFrame([
        {"title": "Junior Developer", "company": "Acme", "location": "Nairobi, Kenya",
         "salary": "KES 90,000", "deadline": "2026-09-01",
         "description": "python and sql", "source_post_date": "2026-08-01",
         "currency": "KES"},
        {"title": "Data Analyst", "company": "Baobab", "location": "Lagos, Nigeria",
         "salary": "NGN 400,000", "deadline": "2026-09-15",
         "description": "excel dashboards", "source_post_date": "2026-08-02",
         "currency": "NGN"},
    ])


TODAY = date(2026, 8, 8)


def test_all_dimensions_in_unit_interval():
    score = score_dataframe(_clean_frame(), today=TODAY)
    assert set(score.dimensions) == set(WEIGHTS)
    for v in score.dimensions.values():
        assert 0.0 <= v <= 1.0
    assert 0.0 <= score.overall <= 1.0
    assert score.grade in {"A", "B", "C", "D", "F"}


def test_clean_frame_scores_well():
    score = score_dataframe(_clean_frame(), today=TODAY)
    assert score.dimensions["completeness"] == 1.0
    assert score.dimensions["uniqueness"] == 1.0
    assert score.overall >= 0.85


def test_detects_injected_missingness():
    df = _clean_frame()
    df.loc[:, "deadline"] = ""            # wipe a whole field
    score = score_dataframe(df, today=TODAY)
    assert score.dimensions["completeness"] < 1.0
    assert "deadline=0%" in score.details["completeness"]


def test_detects_duplicates():
    df = _clean_frame()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)   # 3 rows, 2 unique
    score = score_dataframe(dup, today=TODAY)
    assert score.dimensions["uniqueness"] < 1.0


def test_stale_posts_lower_freshness():
    df = _clean_frame()
    df.loc[:, "source_post_date"] = "2020-01-01"   # far older than freshness window
    score = score_dataframe(df, today=TODAY)
    assert score.dimensions["freshness"] == 0.0


def test_deterministic():
    a = score_dataframe(_clean_frame(), today=TODAY).dimensions
    b = score_dataframe(_clean_frame(), today=TODAY).dimensions
    assert a == b
