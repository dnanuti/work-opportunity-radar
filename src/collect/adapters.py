"""Source adapters — where raw job posts come from.

Each adapter returns a list of **native** provider records (nested, inconsistent — the
real shape an API gives you). Normalisation into flat rows happens in __init__.py, so
the "semi-structured/nested JSON -> structured table" step is visible and testable.

Design rule for the venue: **never crash**. The live adapters (remotive, remoteok) try
a real HTTP call and, on any failure — no network, timeout, rate limit, changed schema —
fall back to the bundled sample fixtures with a printed warning. The default source is
`sample`, which is fully offline.

We do NOT scrape LinkedIn or any site behind auth or ToS restrictions. The live sources
here are public JSON job APIs.
"""
from __future__ import annotations

import glob
import html
import json
import re
from pathlib import Path

from src import resolve

# Public JSON endpoints (no key required). Used only when the user opts in.
REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
REMOTEOK_URL = "https://remoteok.com/api"
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"
LEVER_URL = "https://api.lever.co/v0/postings/{site}"


def _plain_text(value: str) -> str:
    """Make HTML-rich ATS descriptions readable without another dependency."""
    no_tags = re.sub(r"<[^>]+>", " ", value or "")
    text = re.sub(r"\s+", " ", html.unescape(no_tags)).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


# --- Offline fixtures (the default) ---------------------------------------------

def _load_fixtures(cfg: dict) -> list[dict]:
    """Read every *.json in the fixtures dir. Files may be a provider envelope
    ({"jobs": [...]}) or a bare list — we flatten both."""
    fx_dir = resolve(cfg["collect"]["fixtures_dir"])
    records: list[dict] = []
    for fp in sorted(glob.glob(str(Path(fx_dir) / "*.json"))):
        with open(fp, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "jobs" in payload:
            records.extend(payload["jobs"])
        elif isinstance(payload, list):
            # RemoteOK-style: the first element is a legal/notice object — skip it.
            records.extend(r for r in payload if isinstance(r, dict) and
                           ("position" in r or "title" in r))
        elif isinstance(payload, dict):
            records.append(payload)
    return records


# --- User files ------------------------------------------------------------------

def _load_user_file(cfg: dict) -> list[dict]:
    """Read the user's own CSV or JSON export (source: csv | json)."""
    path = cfg["collect"].get("user_file", "")
    if not path:
        raise ValueError(
            "collect.source is 'csv'/'json' but collect.user_file is empty — "
            "set it to the path of your exported file."
        )
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"collect.user_file not found: {p}")
    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "jobs" in payload:
            return payload["jobs"]
        return payload if isinstance(payload, list) else [payload]
    # CSV -> list of row dicts.
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- Live adapters (opt-in; fall back to fixtures on any failure) ----------------

def _http_get_json(url: str, params: dict, timeout: int):
    import requests  # imported lazily so the offline path needs no dependency at import
    headers = {"User-Agent": "work-opportunity-radar/teaching-demo (+offline-first)"}
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _remotive(cfg: dict) -> list[dict]:
    data = _http_get_json(
        REMOTIVE_URL,
        params={"search": cfg["collect"].get("query", "")},
        timeout=cfg["collect"].get("timeout_seconds", 8),
    )
    return data.get("jobs", []) if isinstance(data, dict) else []


def _remoteok(cfg: dict) -> list[dict]:
    data = _http_get_json(
        REMOTEOK_URL, params={}, timeout=cfg["collect"].get("timeout_seconds", 8)
    )
    # First element is a legal notice; keep the actual posts.
    return [r for r in data if isinstance(r, dict) and "position" in r]


def _africa_ats(cfg: dict) -> list[dict]:
    """Read public employer feeds from Africa-active Greenhouse/Lever boards.

    These are documented public job-posting APIs, not scraped HTML. The company list is
    explicit in config.yaml so learners can see exactly which sources are represented.
    """
    from datetime import datetime, timezone

    records: list[dict] = []
    timeout = cfg["collect"].get("timeout_seconds", 8)
    for board in cfg["collect"].get("greenhouse_boards", []):
        payload = _http_get_json(
            GREENHOUSE_URL.format(board=board), {"content": "true"}, timeout
        )
        for job in payload.get("jobs", []):
            records.append({
                "id": f"greenhouse-{board}-{job.get('id', '')}",
                "title": job.get("title", ""),
                "company": board,
                "location": (job.get("location") or {}).get("name", ""),
                "description": _plain_text(job.get("content", "")),
                "source_post_date": job.get("updated_at", ""),
                "url": job.get("absolute_url", ""),
            })
    for site in cfg["collect"].get("lever_sites", []):
        payload = _http_get_json(
            LEVER_URL.format(site=site), {"mode": "json"}, timeout
        )
        for job in payload if isinstance(payload, list) else []:
            created = job.get("createdAt")
            created_iso = ""
            if isinstance(created, (int, float)):
                created_iso = datetime.fromtimestamp(
                    created / 1000, tz=timezone.utc
                ).isoformat()
            records.append({
                "id": f"lever-{site}-{job.get('id', '')}",
                "title": job.get("text", ""),
                "company": site,
                "location": (job.get("categories") or {}).get("location", ""),
                "description": job.get("descriptionPlain", ""),
                "source_post_date": created_iso,
                "url": job.get("hostedUrl", ""),
            })
    return records


_LIVE = {"remotive": _remotive, "remoteok": _remoteok, "africa_ats": _africa_ats}


def fetch_raw(cfg: dict, source: str | None = None) -> tuple[list[dict], str]:
    """Return (native_records, effective_source). Live sources fall back to fixtures."""
    source = source or cfg["collect"]["source"]

    if source == "sample":
        return _load_fixtures(cfg), "sample"
    if source in ("csv", "json"):
        return _load_user_file(cfg), source
    if source in _LIVE:
        try:
            records = _LIVE[source](cfg)
            if records:
                return records, source
            print(f"[collect] WARNING: '{source}' returned no rows — "
                  f"falling back to offline sample.")
        except Exception as e:  # noqa: BLE001 — never crash the demo on a bad network
            print(f"[collect] WARNING: live source '{source}' failed ({e}) — "
                  f"falling back to offline sample.")
        return _load_fixtures(cfg), f"sample (fallback from {source})"

    raise ValueError(f"Unknown collect.source: {source!r}. "
                     f"Use sample | africa_ats | csv | json | remotive | remoteok.")
