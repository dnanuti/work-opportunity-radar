import json
from datetime import date

from src.application_assistant import ExtractedJob
from src.evidence import build_evidence_card, format_evidence_card
from src.feedback import record_feedback


def _extraction():
    return ExtractedJob(
        is_opportunity=True,
        role="Junior Data Analyst",
        level="junior",
        skills=["Python", "SQL"],
        location="Nairobi",
        salary=None,
        deadline=None,
        application_link="https://example.org/apply",
        confidence="medium",
        missing_information=["salary", "closing_date"],
        evidence={"role": "Junior Data Analyst", "level": "Junior Data Analyst",
                  "skills": "Python and SQL", "location": "Nairobi", "salary": None,
                  "deadline": None, "application_link": "https://example.org/apply"},
    )


def test_evidence_card_exposes_source_gaps_and_next_step():
    job = {"source": "sample", "url": "https://example.org/apply",
           "opportunity_status": "likely_opportunity", "why": "skills match"}
    card = build_evidence_card(job, _extraction(), verified_on=date(2026, 8, 8))
    text = format_evidence_card(card)
    assert "SOURCE - sample" in text
    assert "closing_date" in text and "salary" in text
    assert "NEXT STEP" in text


def test_feedback_is_append_only_and_needs_review(tmp_path):
    path = tmp_path / "feedback.jsonl"
    record_feedback("job-1", True, "Useful after source review", path=path)
    record_feedback("job-2", None, "Not enough information", path=path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["job_id"] for row in rows] == ["job-1", "job-2"]
    assert all(row["review_status"] == "needs_review" for row in rows)
