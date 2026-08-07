"""Collection tests — run against the bundled offline fixtures, so no network needed.
They prove: nested feed -> flat schema, the Africa filter, and graceful live fallback."""
from __future__ import annotations

from src import load_config
from src.collect import CANON_COLUMNS, fetch, normalise_record
from src.collect import adapters
from src.collect.africa import canonical_location, detect_currency, is_remote
import json


def _cfg():
    cfg = load_config()
    cfg["collect"]["source"] = "sample"
    cfg["collect"]["query"] = ""          # keep all, then assert on the full set
    cfg["collect"]["africa_only"] = True
    return cfg


def test_fetch_returns_canonical_schema():
    df = fetch(_cfg(), write=False, verbose=False)
    assert list(df.columns) == CANON_COLUMNS
    assert len(df) > 0
    assert (df["title"].str.strip() != "").all()          # no empty titles survive


def test_fetch_preserves_raw_payload_separately(tmp_path):
    cfg = _cfg()
    cfg["paths"]["raw_source_records"] = str(tmp_path / "source_records.json")
    cfg["paths"]["collected_jobs"] = str(tmp_path / "collected.csv")
    fetch(cfg, write=True, verbose=False)
    raw = json.loads((tmp_path / "source_records.json").read_text(encoding="utf-8"))
    assert raw["source_requested"] == "sample"
    assert raw["records"]
    assert (tmp_path / "collected.csv").exists()


def test_real_feed_has_no_labels():
    df = fetch(_cfg(), write=False, verbose=False)
    # Real collected data carries no ground truth — the teaching point behind notebook 02.
    assert (df["is_opportunity"].astype(str).str.strip() == "").all()


def test_africa_filter_excludes_non_reachable():
    df = fetch(_cfg(), write=False, verbose=False)
    locs = " ".join(df["location"].tolist()).lower()
    # Berlin is on-site in Germany with "no remote" — must be filtered out.
    assert "berlin" not in locs
    # But a Nairobi role must be kept.
    assert (df["location_canonical"] == "Nairobi, Kenya").any()


def test_near_duplicate_is_preserved_for_cleaning():
    df = fetch(_cfg(), write=False, verbose=False)
    # Two Acacia Analytics Nairobi analyst posts (Junior vs Jr.) come in as separate rows;
    # dedup happens later in cleaning, not at collection time.
    acacia = df[df["company"] == "Acacia Analytics"]
    assert len(acacia) >= 2


def test_normalise_record_maps_both_provider_shapes():
    remotive = {"id": 1, "title": "Data Analyst", "company_name": "X",
                "candidate_required_location": "Lagos, NG",
                "salary": "NGN 400,000", "description": "SQL role",
                "publication_date": "2026-07-01", "url": "u"}
    r = normalise_record(remotive, "remotive")
    assert r["company"] == "X" and r["location_canonical"] == "Lagos, Nigeria"
    assert r["currency"] == "NGN"

    remoteok = {"id": "ro-9", "position": "Backend Developer", "company": "Y",
                "location": "Remote", "salary_min": 1000, "salary_max": 2000,
                "description": "python", "date": "2026-08-01"}
    r2 = normalise_record(remoteok, "remoteok")
    assert r2["title"] == "Backend Developer" and r2["is_remote"] == 1
    assert r2["salary"] == "1000-2000"


def test_live_source_falls_back_to_sample(monkeypatch):
    """If a live API fails, we must fall back to the offline sample, never crash."""
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(adapters, "_http_get_json", boom)
    cfg = _cfg()
    records, effective = adapters.fetch_raw(cfg, source="remotive")
    assert effective.startswith("sample")
    assert len(records) > 0


def test_africa_ats_reads_public_greenhouse_and_lever_feeds(monkeypatch):
    def fake_get(url, params, timeout):
        if "greenhouse" in url:
            return {"jobs": [{
                "id": 7, "title": "Graduate Data Analyst",
                "location": {"name": "Cape Town, South Africa"},
                "content": "<p>Use <strong>SQL</strong>.</p>",
                "updated_at": "2026-08-01T00:00:00Z",
                "absolute_url": "https://example.org/gh/7",
            }]}
        return [{
            "id": "abc", "text": "Junior Developer",
            "categories": {"location": "Nairobi, Kenya"},
            "descriptionPlain": "Build with Python",
            "createdAt": 1785542400000,
            "hostedUrl": "https://example.org/lever/abc",
        }]

    monkeypatch.setattr(adapters, "_http_get_json", fake_get)
    cfg = _cfg()
    cfg["collect"]["greenhouse_boards"] = ["jumo"]
    cfg["collect"]["lever_sites"] = ["Yassir"]
    records, effective = adapters.fetch_raw(cfg, source="africa_ats")
    assert effective == "africa_ats"
    assert len(records) == 2
    assert records[0]["description"] == "Use SQL."
    assert records[1]["company"] == "Yassir"


def test_africa_helpers():
    assert canonical_location("nairobi, ke") == "Nairobi, Kenya"
    assert canonical_location("Fully Remote") == "Remote"
    assert is_remote("This is a remote role") is True
    assert is_remote("On-site only, no remote") is False     # negation handled
    assert detect_currency("KES 90,000") == "KES"
    assert detect_currency("junior backend developer") is None   # no false positive
