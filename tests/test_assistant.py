"""Assistant tests — the grounded output MUST validate, and malformed output MUST
be rejected. 'missing_information' MUST be populated when a fact is absent."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import load_config
from src.application_assistant import (
    ExtractedJob,
    extract_grounded,
    validate_extraction,
)
from src.generate_data import generate


def test_valid_extraction_passes():
    good = {
        "is_opportunity": True,
        "role": "Junior Data Analyst",
        "level": "junior",
        "skills": ["SQL", "Python"],
        "location": "Nairobi, Kenya",
        "salary": None,
        "deadline": None,
        "application_link": None,
        "confidence": "medium",
        "missing_information": ["closing_date"],
        "evidence": {"role": "Junior Data Analyst", "level": "Junior Data Analyst",
                     "skills": "SQL and Python", "location": "Nairobi, Kenya",
                     "salary": None, "deadline": None, "application_link": None},
    }
    job = validate_extraction(good)
    assert isinstance(job, ExtractedJob)
    assert job.role == "Junior Data Analyst"


def test_malformed_output_is_rejected():
    # confidence not in the allowed set, skills not a list -> must raise.
    bad = {
        "is_opportunity": True,
        "role": "X",
        "level": None,
        "skills": "SQL",                 # should be a list
        "location": None,
        "salary": None,
        "deadline": None,
        "application_link": None,
        "confidence": "very-sure",       # not low/medium/high
        "missing_information": [],
        "evidence": {},
    }
    with pytest.raises(ValidationError):
        validate_extraction(bad)


def test_missing_field_rejected():
    incomplete = {"is_opportunity": True, "role": "X"}   # missing required fields
    with pytest.raises(ValidationError):
        validate_extraction(incomplete)


def test_grounded_declares_missing_information(tmp_path):
    """On a real generated post (no application link), missing_information is populated."""
    cfg = load_config()
    cfg["paths"]["raw_jobs"] = str(tmp_path / "jobs.csv")
    generate(cfg)
    from src.application_assistant import pick_sample
    sample = pick_sample(cfg)
    sample["url"] = ""  # make the grounding claim explicit for this test
    job = extract_grounded(sample, cfg)
    # With no supplied link, the assistant must expose the gap rather than invent one.
    assert "application_link" in job.missing_information
    assert isinstance(job.skills, list)
