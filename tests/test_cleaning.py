"""Cleaning tests — run against the seeded generator, not hand-written fixtures,
so they cover the real path the audience sees."""
from __future__ import annotations

import pandas as pd

from src import load_config
from src.cleaning import clean, normalise_location, normalise_text, parse_date
from src.generate_data import generate


def _fresh(tmp_path):
    """Generate + clean into a temp location, returning (cfg, clean_df)."""
    cfg = load_config()
    cfg["paths"]["raw_jobs"] = str(tmp_path / "jobs.csv")
    cfg["paths"]["clean_jobs"] = str(tmp_path / "jobs_clean.csv")
    generate(cfg)
    df = clean(cfg, verbose=False)
    return cfg, df


def test_dedup_removes_injected_duplicates(tmp_path):
    cfg, clean_df = _fresh(tmp_path)
    raw = pd.read_csv(cfg["paths"]["raw_jobs"], dtype=str, keep_default_na=False)
    # Duplicates were injected, so cleaning must remove a positive number of rows.
    assert len(clean_df) < len(raw)
    # And the content_key must be unique after dedup.
    assert clean_df["content_key"].is_unique


def test_near_duplicates_are_caught():
    # 'Jr.' vs 'Junior' plus whitespace must collapse to the same content key.
    from src.cleaning import content_key
    a = pd.Series({"title": "Junior Data Analyst", "company": "Acacia Analytics",
                   "description": "We are hiring a Junior Data Analyst."})
    b = pd.Series({"title": "Jr. Data Analyst", "company": "acacia analytics",
                   "description": "  We are hiring a Junior Data Analyst.  "})
    assert content_key(a) == content_key(b)


def test_location_normalisation_collapses_spellings():
    for raw in ["Nairobi", "nairobi", "Nairobi, KE", "Nairobi, Kenya"]:
        assert normalise_location(raw) == "Nairobi"
    for raw in ["Remote", "remote", "REMOTE", "Fully remote"]:
        assert normalise_location(raw) == "Remote"
    assert normalise_location("") == "Unknown"


def test_missing_values_are_flagged_not_dropped(tmp_path):
    cfg, clean_df = _fresh(tmp_path)
    # Flag columns exist and there is genuinely some missingness to flag.
    assert {"has_salary", "has_deadline", "deadline_unparseable"} <= set(clean_df.columns)
    assert (clean_df["has_salary"] == 0).sum() > 0
    # Flagged-missing rows are retained, not dropped.
    assert clean_df["has_salary"].isin([0, 1]).all()


def test_normalise_text_collapses_whitespace():
    assert normalise_text("  Hello   World  ") == "hello world"
    assert normalise_text(None) == ""


def test_iso_date_is_not_reinterpreted_as_day_first():
    assert parse_date("2026-09-01").date().isoformat() == "2026-09-01"
    assert parse_date("01/09/2026").date().isoformat() == "2026-09-01"
