"""A reusable data-quality scorecard.

'Evaluate the quality of your data' is easy to say and hard to make concrete. This turns
it into five measured dimensions, each scored 0-1, plus an overall grade — so a student
can point at a number, not a vibe. It runs on ANY DataFrame of job posts (collected,
synthetic, or their own), which is the point: quality is measurable before any ML.

Dimensions:
  * completeness — are the fields actually filled in?
  * validity     — do filled-in values parse (dates parse, salary carries a currency)?
  * uniqueness   — how much is duplicated (exact + near, via a content key)?
  * consistency  — how many redundant spellings of the same category (locations)?
  * freshness    — how recent are the posts?

Run: python -m src.quality
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from src import load_config, resolve
from src.cleaning import parse_date

# Columns we grade for completeness, and the weight of each dimension in the overall.
KEY_COLUMNS = ["title", "company", "location", "salary", "deadline", "description"]
WEIGHTS = {
    "completeness": 0.30,
    "validity": 0.20,
    "uniqueness": 0.25,
    "consistency": 0.15,
    "freshness": 0.10,
}


@dataclass
class QualityScore:
    dimensions: dict          # name -> score in [0, 1]
    details: dict = field(default_factory=dict)   # human-readable notes per dimension
    issues: list = field(default_factory=list)    # the biggest problems, worst first

    @property
    def overall(self) -> float:
        return round(sum(WEIGHTS[k] * v for k, v in self.dimensions.items()), 3)

    @property
    def grade(self) -> str:
        o = self.overall
        return ("A" if o >= 0.90 else "B" if o >= 0.80 else "C" if o >= 0.70
                else "D" if o >= 0.60 else "F")


def _norm(s) -> str:
    import re
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _content_key(row: pd.Series) -> str:
    title = _norm(row.get("title", "")).replace("jr.", "junior")
    return f"{title}|{_norm(row.get('company', ''))}|{_norm(row.get('description', ''))}"


def _completeness(df: pd.DataFrame, details: dict) -> float:
    cols = [c for c in KEY_COLUMNS if c in df.columns]
    if not cols or len(df) == 0:
        return 0.0
    per_col = {}
    total_filled = total_cells = 0
    for c in cols:
        filled = (df[c].astype(str).str.strip() != "").sum()
        per_col[c] = round(filled / len(df), 2)
        total_filled += filled
        total_cells += len(df)
    details["completeness"] = "filled rate per field: " + ", ".join(
        f"{c}={v:.0%}" for c, v in per_col.items()
    )
    return round(total_filled / total_cells, 3)


def _validity(df: pd.DataFrame, details: dict) -> float:
    checks = []
    # Deadlines that are present should parse to a real date.
    if "deadline" in df.columns:
        present = df["deadline"].astype(str).str.strip() != ""
        if present.any():
            parsed = df.loc[present, "deadline"].map(parse_date)
            checks.append(parsed.notna().mean())
    # Salaries that are present should carry a currency (a bare number is ambiguous).
    if "salary" in df.columns:
        present = df["salary"].astype(str).str.strip() != ""
        if present.any():
            if "currency" in df.columns:
                has_cur = (df.loc[present, "currency"].astype(str).str.strip() != "")
            else:
                has_cur = df.loc[present, "salary"].astype(str).str.contains(
                    r"[$€₦]|usd|kes|ngn|ghs|zar|egp|eur", case=False, regex=True)
            checks.append(has_cur.mean())
    # Post dates that are present should parse.
    if "source_post_date" in df.columns:
        present = df["source_post_date"].astype(str).str.strip() != ""
        if present.any():
            parsed = pd.to_datetime(df.loc[present, "source_post_date"],
                                    errors="coerce", format="mixed", utc=True)
            checks.append(parsed.notna().mean())
    score = float(sum(checks) / len(checks)) if checks else 1.0
    details["validity"] = (f"{len(checks)} value-format checks "
                           f"(dates parse, salary has a currency)")
    return round(score, 3)


def _uniqueness(df: pd.DataFrame, details: dict) -> float:
    if len(df) == 0:
        return 0.0
    keys = df.apply(_content_key, axis=1)
    n_unique = keys.nunique()
    dup = len(df) - n_unique
    details["uniqueness"] = (f"{dup} duplicate/near-duplicate rows "
                             f"({n_unique} unique of {len(df)})")
    return round(n_unique / len(df), 3)


def _consistency(df: pd.DataFrame, details: dict) -> float:
    if "location" not in df.columns or len(df) == 0:
        return 1.0
    from src.collect.africa import canonical_location
    raw_spellings = df["location"].astype(str).str.strip()
    raw_spellings = raw_spellings[raw_spellings != ""]
    if raw_spellings.empty:
        return 1.0
    distinct_raw = raw_spellings.nunique()
    distinct_canon = raw_spellings.map(canonical_location).nunique()
    # Perfectly consistent -> raw spellings == canonical count -> score 1.
    score = distinct_canon / distinct_raw if distinct_raw else 1.0
    details["consistency"] = (f"{distinct_raw} raw location spellings collapse to "
                              f"{distinct_canon} canonical")
    return round(float(score), 3)


def _freshness(df: pd.DataFrame, details: dict, freshness_days: int, today: date) -> float:
    col = "source_post_date" if "source_post_date" in df.columns else None
    if col is None or len(df) == 0:
        return 1.0
    parsed = pd.to_datetime(df[col], errors="coerce", format="mixed", utc=True)
    present = parsed.notna()
    if not present.any():
        details["freshness"] = "no parseable post dates"
        return 0.0
    ages = (pd.Timestamp(today, tz="UTC") - parsed[present]).dt.days
    fresh = (ages <= freshness_days).mean()
    details["freshness"] = (f"{int((ages <= freshness_days).sum())}/{present.sum()} posts "
                            f"within {freshness_days} days (median age {int(ages.median())}d)")
    return round(float(fresh), 3)


def score_dataframe(df: pd.DataFrame, cfg: dict | None = None,
                    today: date | None = None) -> QualityScore:
    """Compute the five-dimension quality scorecard for a job-post DataFrame."""
    cfg = cfg or load_config()
    freshness_days = cfg.get("collect", {}).get("freshness_days", 45)
    today = today or datetime.now().date()

    details: dict = {}
    dims = {
        "completeness": _completeness(df, details),
        "validity": _validity(df, details),
        "uniqueness": _uniqueness(df, details),
        "consistency": _consistency(df, details),
        "freshness": _freshness(df, details, freshness_days, today),
    }
    issues = sorted(
        ({"dimension": k, "score": v, "note": details.get(k, "")}
         for k, v in dims.items() if v < 0.85),
        key=lambda d: d["score"],
    )
    return QualityScore(dimensions=dims, details=details, issues=issues)


def print_scorecard(score: QualityScore) -> None:
    bar = lambda v: "█" * int(round(v * 20)) + "·" * (20 - int(round(v * 20)))
    print("=" * 60)
    print("DATA QUALITY SCORECARD")
    print("=" * 60)
    for name in WEIGHTS:
        v = score.dimensions[name]
        print(f"  {name:<13} {bar(v)} {v:>5.0%}   {score.details.get(name, '')}")
    print("-" * 60)
    print(f"  OVERALL       {bar(score.overall)} {score.overall:>5.0%}   grade {score.grade}")
    if score.issues:
        print("\n  Fix first (lowest scores):")
        for it in score.issues[:3]:
            print(f"    - {it['dimension']} ({it['score']:.0%}): {it['note']}")
    print("=" * 60)


def _load_default_frame(cfg: dict) -> pd.DataFrame:
    """Prefer collected real data; fall back to raw synthetic if not collected yet."""
    collected = resolve(cfg["paths"]["collected_jobs"])
    path = collected if collected.exists() else resolve(cfg["paths"]["raw_jobs"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


if __name__ == "__main__":
    cfg = load_config()
    df = _load_default_frame(cfg)
    print_scorecard(score_dataframe(df, cfg))
