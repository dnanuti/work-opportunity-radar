"""Radar ranking tests. The radar is a transparent rule, so every assertion here maps
to a reason a human could read in the 'why' column."""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.radar import rank_opportunities

TODAY = date(2026, 8, 8)

PROFILE = {
    "name": "Amina (fictional)",
    "skills": ["Python", "SQL"],
    "location_preference": ["Nairobi", "Remote"],
    "career_stage": "new_graduate",
    "preferred_levels": ["graduate", "junior", "entry_level"],
    "target_roles": ["Backend Developer", "Data Analyst"],
}


def _frame():
    return pd.DataFrame([
        # Strong fit: junior, remote, both skills, recent.
        {"title": "Junior Backend Developer", "company": "Zanzibar Cloud",
         "location": "Remote", "salary": "USD 2,000",
         "description": "python and sql, learn from senior engineers",
         "source_post_date": "2026-08-01"},
        # Weak fit: senior, unrelated skills, no salary, far location.
        {"title": "Senior Java Architect", "company": "Old Corp",
         "location": "Berlin, Germany", "salary": "",
         "description": "10+ years of java and kubernetes",
         "source_post_date": "2026-08-01"},
    ])


def test_adds_score_and_why():
    ranked = rank_opportunities(_frame(), PROFILE, today=TODAY)
    assert "fit_score" in ranked.columns and "why" in ranked.columns
    assert (ranked["fit_score"].between(0.0, 1.0)).all()
    assert (ranked["why"].str.len() > 0).all()


def test_strong_fit_ranks_first():
    ranked = rank_opportunities(_frame(), PROFILE, today=TODAY)
    assert ranked.iloc[0]["company"] == "Zanzibar Cloud"
    assert ranked.iloc[0]["fit_score"] > ranked.iloc[1]["fit_score"]


def test_sorted_descending():
    ranked = rank_opportunities(_frame(), PROFILE, today=TODAY)
    scores = ranked["fit_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_title_is_authoritative_for_level():
    # Title says Junior, description mentions senior engineers -> still a junior match.
    ranked = rank_opportunities(_frame(), PROFILE, today=TODAY)
    junior = ranked[ranked["title"] == "Junior Backend Developer"].iloc[0]
    assert "experience level fits (junior)" in junior["why"]
    senior = ranked[ranked["title"] == "Senior Java Architect"].iloc[0]
    assert "experience level may not fit (senior)" in senior["why"]


def test_skill_match_is_explained():
    ranked = rank_opportunities(_frame(), PROFILE, today=TODAY)
    top = ranked.iloc[0]
    assert "Python" in top["why"] and "SQL" in top["why"]


def test_senior_candidate_is_not_forced_into_junior_roles():
    senior = {**PROFILE, "preferred_levels": ["senior"],
              "target_roles": ["Java Architect"], "skills": ["Java", "Kubernetes"]}
    ranked = rank_opportunities(_frame(), senior, today=TODAY)
    assert ranked.iloc[0]["company"] == "Old Corp"
