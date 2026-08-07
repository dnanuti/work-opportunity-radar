"""Collection layer — turn messy, nested, multi-source job feeds into one flat table.

Public API:
    fetch(cfg=None, source=None, write=True, verbose=True) -> pd.DataFrame

The flow the audience sees:
    native provider JSON  (semi-structured, nested, inconsistent keys)
        -> normalise_record()        (flatten to a fixed schema)
        -> Africa relevance filter    (keep what a Nairobi grad can actually take)
        -> keyword filter + cap
        -> a tidy DataFrame + a short collection report

Real feeds have **no `is_opportunity` label** — that column is intentionally blank here.
That is a teaching point, not a bug: supervised ML (notebook 02) needs labels, so it
uses the synthetic generator; the radar (notebook 00) ranks *unlabelled* real posts by
profile fit instead. You cannot train a supervised model on data with no ground truth.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src import load_config, rel, resolve
from src.collect import africa
from src.collect.adapters import fetch_raw

# The flat schema every source is normalised into. Superset of the synthetic columns so
# the same cleaning helpers work, plus provenance (source, url) and Africa tags.
CANON_COLUMNS = [
    "id", "title", "company", "location", "salary", "deadline", "description",
    "source_post_date", "is_opportunity", "source", "url",
    "location_canonical", "region", "is_remote", "currency",
]


def _first(record: dict, *keys, default=""):
    """Return the first present, non-empty value among candidate keys (feeds disagree
    on names: 'company' vs 'company_name', 'title' vs 'position', ...)."""
    for k in keys:
        if k in record and record[k] not in (None, ""):
            return record[k]
    return default


def _salary_string(record: dict) -> str:
    """Feeds express salary differently: a string, or min/max numbers. Unify to text."""
    s = _first(record, "salary", "salary_text", default="")
    if s:
        return str(s).strip()
    lo = _first(record, "salary_min", default="")
    hi = _first(record, "salary_max", default="")
    if lo or hi:
        return f"{lo}-{hi}".strip("-")
    return ""


def normalise_record(record: dict, source: str) -> dict:
    """Flatten one native provider record into the canonical schema."""
    title = str(_first(record, "title", "position", "role")).strip()
    company = str(_first(record, "company", "company_name", "employer")).strip()
    location = str(_first(
        record, "location", "candidate_required_location", "region", default=""
    )).strip()
    description = str(_first(record, "description", "desc", "summary", default="")).strip()
    salary = _salary_string(record)
    post_date = str(_first(record, "source_post_date", "publication_date", "date",
                           "created_at", default="")).strip()
    url = str(_first(record, "url", "apply_url", "link", default="")).strip()
    rid = str(_first(record, "id", "slug", default="")).strip()

    canon_loc = africa.canonical_location(location)
    remote = africa.is_remote(f"{location} {description} {title}")
    if remote and canon_loc in ("Unknown", ""):
        canon_loc = "Remote"

    return {
        "id": rid,
        "title": title,
        "company": company,
        "location": location,
        "salary": salary,
        "deadline": str(_first(record, "deadline", "closing_date", default="")).strip(),
        "description": description,
        "source_post_date": post_date,
        "is_opportunity": "",                      # real feeds carry no ground-truth label
        "source": source,
        "url": url,
        "location_canonical": canon_loc,
        "region": africa.region_of(canon_loc),
        "is_remote": int(remote),
        "currency": africa.detect_currency(f"{salary} {description}") or "",
    }


def _matches_query(row: dict, query: str) -> bool:
    if not query:
        return True
    q = query.lower().strip()
    return q in row["title"].lower() or q in row["description"].lower()


def fetch(cfg: dict | None = None, source: str | None = None,
          write: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Collect, normalise, filter, and (optionally) write real-world job posts."""
    cfg = cfg or load_config()
    conf = cfg["collect"]
    native, effective = fetch_raw(cfg, source)

    # Raw layer: preserve exactly what the provider returned, plus provenance. Cleaning
    # never overwrites this evidence, so a transformation can be explained or replayed.
    if write:
        raw_path = resolve(cfg["paths"]["raw_source_records"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps({
            "source_requested": source or conf["source"],
            "source_used": effective,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "records": native,
        }, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    rows = [normalise_record(r, effective.split(" ")[0]) for r in native]
    # Drop rows with no title at all — unusable noise.
    rows = [r for r in rows if r["title"]]

    kept = []
    for r in rows:
        if conf.get("africa_only", True) and not africa.is_africa_relevant(
            r["location"], r["description"], conf.get("include_remote", True)
        ):
            continue
        if not _matches_query(r, conf.get("query", "")):
            continue
        kept.append(r)

    kept = kept[: conf.get("max_results", 60)]
    # Assign stable ids where the feed didn't provide one.
    for i, r in enumerate(kept):
        if not r["id"]:
            r["id"] = f"{r['source']}_{i:04d}"

    df = pd.DataFrame(kept, columns=CANON_COLUMNS)

    if write:
        out = resolve(cfg["paths"]["collected_jobs"])
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    if verbose:
        _report(df, native, effective, conf, cfg)
    return df


def _report(df: pd.DataFrame, native: list, effective: str, conf: dict,
            cfg: dict) -> None:
    print("=" * 60)
    print("COLLECTION REPORT  (semi-structured feed -> structured table)")
    print("=" * 60)
    print(f"Source requested          : {conf['source']}")
    print(f"Source used               : {effective}")
    print(f"Native records pulled     : {len(native)}")
    print(f"Kept after Africa+query   : {len(df)}  "
          f"(query={conf.get('query', '')!r}, africa_only={conf.get('africa_only')})")
    if len(df):
        print(f"Raw evidence              : {rel(cfg['paths']['raw_source_records'])}")
        print(f"Structured records        : {rel(cfg['paths']['collected_jobs'])}")
        print("\nRepresentation by region (who is in the data / who is missing):")
        for region, n in df["region"].value_counts().items():
            print(f"    {region:<22}: {n}")
        remote_n = int(df["is_remote"].sum())
        print(f"\nRemote-friendly roles     : {remote_n}/{len(df)}")
        with_salary = int((df["salary"].str.strip() != "").sum())
        print(f"Posts stating a salary    : {with_salary}/{len(df)}  "
              f"(the rest hide it — a real-world data gap)")
        print(f"Posts with a label        : 0/{len(df)}  "
              f"(real feeds have no ground truth — see notebook 02)")
    print("=" * 60)


if __name__ == "__main__":
    fetch()
