"""Cleaning: dedup, missing-value handling, and category normalisation.

The teaching point: cleaning is not busywork before the 'real' work — every choice
here changes what the model is allowed to learn. Run: python -m src.cleaning
"""
from __future__ import annotations

import re

import pandas as pd

from src import load_config, rel, resolve

# Canonical locations and the messy spellings that map to them.
LOCATION_CANON = {
    "nairobi": "Nairobi",
    "nairobi, ke": "Nairobi",
    "nairobi, kenya": "Nairobi",
    "remote": "Remote",
    "fully remote": "Remote",
    "remote (africa)": "Remote",
    "lagos": "Lagos",
    "lagos, ng": "Lagos",
    "lagos, nigeria": "Lagos",
    "accra": "Accra",
    "accra, gh": "Accra",
    "kampala": "Kampala",
    "kampala, ug": "Kampala",
}


def normalise_text(s: str) -> str:
    """Lowercase, strip, collapse whitespace — the basis for both dedup and matching."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip().lower()


def normalise_location(raw: str) -> str:
    """Map a messy location spelling to its canonical form (or Title-case fallback)."""
    key = normalise_text(raw)
    if key == "":
        return "Unknown"
    return LOCATION_CANON.get(key, key.title())


def parse_date(raw) -> pd.Timestamp:
    """Parse ISO dates unambiguously, then accept day-first human-entered dates.

    ``dayfirst=True`` is useful for values such as ``01/09/2026``, but applying it
    to an ISO value such as ``2026-09-01`` can swap the month and day. Source APIs
    commonly use ISO, so recognise that shape before using the human-input fallback.
    """
    value = str(raw or "").strip()
    if not value:
        return pd.NaT
    if re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[T ])", value):
        return pd.to_datetime(value, errors="coerce", format="ISO8601")
    return pd.to_datetime(value, errors="coerce", dayfirst=True)


def content_key(row: pd.Series) -> str:
    """A stable key for 'the same job post', robust to whitespace / Jr. vs Junior /
    upper-vs-lower. This is what lets us catch NEAR-duplicates that drop_duplicates()
    on raw columns would miss."""
    title = normalise_text(str(row["title"])).replace("jr.", "junior")
    company = normalise_text(str(row["company"]))
    desc = normalise_text(str(row["description"]))
    return f"{title}|{company}|{desc}"


def load_raw(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(resolve(cfg["paths"]["raw_jobs"]), dtype=str, keep_default_na=False)


def clean_frame(df: pd.DataFrame, has_labels: bool = True) -> tuple[pd.DataFrame, dict]:
    """Clean a job-post DataFrame in memory (no file IO). Returns (clean_df, stats).

    Used by both the synthetic pipeline (has_labels=True) and the collected real-world
    data (has_labels=False — real feeds carry no is_opportunity ground truth)."""
    df = df.copy()
    n_start = len(df)

    # 1) A content key for every row (before we drop anything).
    df["content_key"] = df.apply(content_key, axis=1)

    # 2) Deduplicate on the content key — this removes exact AND near duplicates.
    n_exact = int(df.duplicated(subset=["title", "company", "location",
                                        "description"]).sum())
    df_dedup = df.drop_duplicates(subset=["content_key"], keep="first").copy()
    n_removed = n_start - len(df_dedup)

    # 3) Normalise location spellings.
    df_dedup["location_clean"] = df_dedup["location"].apply(normalise_location)

    # 4) Missing values: we FLAG rather than silently drop, so nothing disappears
    #    without the audience seeing it. salary -> numeric with has_salary flag.
    df_dedup["has_salary"] = (df_dedup["salary"].astype(str).str.strip() != "").astype(int)
    df_dedup["salary_num"] = pd.to_numeric(df_dedup["salary"], errors="coerce")
    df_dedup["has_deadline"] = (df_dedup["deadline"].astype(str).str.strip() != "").astype(int)

    # 5) Parse deadlines where possible; "unparseable" is a visible, valid outcome.
    parsed = df_dedup["deadline"].map(parse_date)
    df_dedup["deadline_parsed"] = parsed
    df_dedup["deadline_unparseable"] = (
        (df_dedup["deadline"].astype(str).str.strip() != "") & parsed.isna()
    ).astype(int)

    if has_labels:
        df_dedup["is_opportunity"] = pd.to_numeric(df_dedup["is_opportunity"]).astype(int)

    stats = {"n_start": n_start, "n_exact": n_exact, "n_removed": n_removed}
    return df_dedup, stats


def clean(cfg: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = load_raw(cfg)
    df_dedup, stats = clean_frame(df, has_labels=True)

    out_path = resolve(cfg["paths"]["clean_jobs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_dedup.to_csv(out_path, index=False)

    if verbose:
        _report(df, df_dedup, stats["n_start"], stats["n_exact"],
                stats["n_removed"], out_path)
    return df_dedup


def clean_collected(cfg: dict | None = None, df: pd.DataFrame | None = None,
                    verbose: bool = True) -> pd.DataFrame:
    """Create the trusted layer for unlabelled collected opportunities."""
    cfg = cfg or load_config()
    if df is None:
        df = pd.read_csv(resolve(cfg["paths"]["collected_jobs"]), dtype=str,
                         keep_default_na=False)
    trusted, stats = clean_frame(df, has_labels=False)
    out_path = resolve(cfg["paths"]["trusted_jobs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trusted.to_csv(out_path, index=False)
    if verbose:
        _report(df, trusted, stats["n_start"], stats["n_exact"],
                stats["n_removed"], out_path)
    return trusted


def _report(df_raw, df_clean, n_start, n_exact, n_removed, out_path) -> None:
    print("=" * 60)
    print("CLEANING REPORT")
    print("=" * 60)
    print(f"Rows in                         : {n_start}")
    print(f"Exact duplicates (raw columns)  : {n_exact}")
    print(f"Removed by content-key dedup    : {n_removed}  "
          f"(catches near-dups too)")
    print(f"Rows out                        : {len(df_clean)}  -> {rel(out_path)}")
    print("\nLocation spellings collapsed:")
    print(f"    raw distinct  : {df_raw['location'].nunique()}")
    print(f"    clean distinct: {df_clean['location_clean'].nunique()}  "
          f"-> {sorted(df_clean['location_clean'].unique())}")
    print("\nMissing-value flags (kept, not dropped):")
    print(f"    without salary  : {(df_clean['has_salary'] == 0).sum()}")
    print(f"    without deadline: {(df_clean['has_deadline'] == 0).sum()}")
    print(f"    unparseable date: {(df_clean['deadline_unparseable'] == 1).sum()}")
    print("=" * 60)


if __name__ == "__main__":
    clean()
