"""Candidate-facing evidence card: evidence and uncertainty, never an instruction."""
from __future__ import annotations

from datetime import date, datetime

from src.application_assistant import ExtractedJob


def build_evidence_card(job: dict, extraction: ExtractedJob,
                        verified_on: date | None = None) -> dict:
    verified_on = verified_on or datetime.now().date()
    missing = sorted(set(extraction.missing_information))
    url = str(job.get("url", "") or extraction.application_link or "").strip()
    source = str(job.get("source", "") or "provided job record").strip()
    next_step = (
        "Open the original source, confirm the role is still available, and verify the "
        "missing details before sharing personal information."
    )
    return {
        "source": {"name": source, "url": url or None},
        "verified": verified_on.isoformat(),
        "opportunity_status": job.get("opportunity_status", "unclassified"),
        "why_it_may_match": job.get("why", "Not ranked against a candidate profile."),
        "missing": missing,
        "uncertainty": {"high": "low", "medium": "medium", "low": "high"}[
            extraction.confidence
        ],
        "next_step": next_step,
    }


def format_evidence_card(card: dict) -> str:
    source = card["source"]["name"]
    if card["source"].get("url"):
        source += f" - {card['source']['url']}"
    missing = ", ".join(card["missing"]) if card["missing"] else "None identified"
    return "\n".join([
        "EVIDENCE, NOT INSTRUCTIONS",
        f"SOURCE - {source}",
        f"VERIFIED - {card['verified']}",
        f"STATUS - {card['opportunity_status']}",
        f"WHY IT MAY MATCH - {card['why_it_may_match']}",
        f"MISSING - {missing}",
        f"UNCERTAINTY - {card['uncertainty']}",
        f"NEXT STEP - {card['next_step']}",
    ])
