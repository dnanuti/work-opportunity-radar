"""Evidence-based opportunity classification, separate from candidate relevance.

The labels deliberately say *likely*, *uncertain*, and *not current*: a classifier can
organise evidence, but it cannot prove that a company or post is genuine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re

import pandas as pd

from src.cleaning import normalise_text, parse_date

LIKELY = "likely_opportunity"
UNCERTAIN = "uncertain"
NOT_CURRENT = "not_current"

_JOB_WORDS = {
    "job", "role", "position", "hiring", "vacancy", "intern", "internship",
    "analyst", "developer", "engineer", "technician", "representative", "manager",
}
_EVENT_WORDS = {"event", "webinar", "career fair", "meetup", "workshop", "conference"}


@dataclass(frozen=True)
class OpportunityDecision:
    label: str
    uncertainty: str
    reasons: tuple[str, ...]


def classify_opportunity(row: pd.Series | dict, today: date | None = None) -> OpportunityDecision:
    """Classify one record using only visible evidence, never candidate attributes."""
    today = today or datetime.now().date()
    get = row.get
    title = str(get("title", "") or "").strip()
    description = str(get("description", "") or "").strip()
    text = normalise_text(f"{title} {description}")
    url = str(get("url", "") or "").strip()
    company = str(get("company", "") or "").strip()
    deadline = parse_date(get("deadline", ""))

    if pd.notna(deadline) and deadline.date() < today:
        return OpportunityDecision(
            NOT_CURRENT, "low", (f"closing date passed ({deadline.date().isoformat()})",)
        )

    looks_like_event = any(word in text for word in _EVENT_WORDS)
    past_marker = any(word in text for word in ("ended", "expired", "took place", "last month"))
    if looks_like_event and past_marker:
        return OpportunityDecision(
            NOT_CURRENT, "low", ("job-related content describes a past event",)
        )

    job_signals = sorted(
        word for word in _JOB_WORDS
        if re.search(rf"\b{re.escape(word)}\b", text)
    )
    evidence = []
    if title:
        evidence.append("title provided")
    if company:
        evidence.append("company provided")
    if url:
        evidence.append("source/application link provided")
    if job_signals:
        evidence.append(f"job language: {', '.join(job_signals[:3])}")

    if job_signals and title and company and url:
        return OpportunityDecision(LIKELY, "low", tuple(evidence))

    missing = []
    if not company:
        missing.append("company")
    if not url:
        missing.append("source/application link")
    if not job_signals:
        missing.append("clear role or application language")
    reason = "not enough evidence; missing " + ", ".join(missing or ["verification"])
    return OpportunityDecision(UNCERTAIN, "medium", tuple([*evidence, reason]))


def classify_frame(df: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    out = df.copy()
    decisions = [classify_opportunity(row, today=today) for _, row in out.iterrows()]
    out["opportunity_status"] = [decision.label for decision in decisions]
    out["status_uncertainty"] = [decision.uncertainty for decision in decisions]
    out["status_why"] = ["; ".join(decision.reasons) for decision in decisions]
    return out
