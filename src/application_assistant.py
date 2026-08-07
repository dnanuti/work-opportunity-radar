"""GenAI application assistant — vague vs grounded, with schema validation.

Two demos over the SAME job post:
  * Demo A (vague)   — a loose prompt; fluent output that invents facts.
  * Demo B (grounded)— structured JSON, facts-only, with 'what's missing' declared.

Then we use Amina's profile + the validated fields to draft a short message, showing
unknown facts as explicit gaps instead of inventing them.

Run: python -m src.application_assistant
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

from src import load_config, resolve
from src.llm import complete
from src.profile import load_profile


class FieldEvidence(BaseModel):
    """Fixed nullable keys keep the API schema strict and machine-checkable."""
    role: Optional[str]
    level: Optional[str]
    skills: Optional[str]
    location: Optional[str]
    salary: Optional[str]
    deadline: Optional[str]
    application_link: Optional[str]


class ExtractedJob(BaseModel):
    """The contract the grounded GenAI output MUST satisfy. Anything else is rejected."""
    is_opportunity: bool
    role: Optional[str]
    level: Optional[str]
    skills: list[str]
    location: Optional[str]
    salary: Optional[str]
    deadline: Optional[str]
    application_link: Optional[str]
    confidence: Literal["low", "medium", "high"]
    missing_information: list[str]
    evidence: FieldEvidence

    @field_validator("skills", "missing_information", mode="before")
    @classmethod
    def _must_be_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("expected a list")
        return v


def validate_extraction(data: dict) -> ExtractedJob:
    """Validate a raw dict against the schema. Raises ValidationError if malformed."""
    return ExtractedJob(**data)


def _load_profile(cfg: dict) -> dict:
    """Backward-compatible alias used by the deep-dive notebook."""
    return load_profile(cfg)


def _load_prompt(path_str: str, cfg: dict) -> str:
    with open(resolve(path_str), encoding="utf-8") as f:
        return f.read()


def job_context(job: str | Mapping) -> str:
    """Render metadata and description so source facts stay distinct and visible."""
    if isinstance(job, str):
        return f"Description: {job}"
    labels = [
        ("Title", "title"), ("Company", "company"), ("Location", "location"),
        ("Salary", "salary"), ("Deadline", "deadline"),
        ("Published", "source_post_date"), ("Source", "source"),
        ("Application URL", "url"), ("Description", "description"),
    ]
    return "\n".join(f"{label}: {str(job.get(key, '') or '').strip()}" for label, key in labels)


def extract_grounded(job: str | Mapping, cfg: dict) -> ExtractedJob:
    """Run the grounded prompt and validate — the checkable path."""
    template = _load_prompt(cfg["paths"]["grounded_prompt"], cfg)
    # .replace (not .format) — the grounded prompt contains a literal JSON example
    # whose { } braces would otherwise be read as format fields.
    raw = complete(template.replace("{job_post}", job_context(job)),
                   schema=ExtractedJob, cfg=cfg)
    return validate_extraction(raw)


def summarise_vague(job: str | Mapping, cfg: dict) -> str:
    """Run the vague prompt — fluent, unchecked, may hallucinate."""
    template = _load_prompt(cfg["paths"]["vague_prompt"], cfg)
    return complete(template.replace("{job_post}", job_context(job)), cfg=cfg)


def draft_message(job: ExtractedJob, profile: dict) -> str:
    """Draft a short application note. Unknown facts are shown as gaps, not invented."""
    role = job.role or "the advertised role"
    extracted_skills = {skill.casefold() for skill in job.skills}
    matched = [s for s in profile["skills"] if s.casefold() in extracted_skills]
    skills_line = (", ".join(matched) if matched
                   else "skills relevant to the role")
    lines = [
        f"Hello,",
        f"",
        f"I'm {profile['name'].replace(' (fictional)', '')}, a {profile['career_stage'].replace('_', ' ')} "
        f"based in {profile['location']}. I'd like to apply for {role}.",
        f"My relevant skills include {skills_line}.",
    ]
    if job.deadline:
        lines.append(f"I understand the closing date is {job.deadline}.")
    if job.missing_information:
        lines.append(
            f"[Gaps to confirm before sending: "
            f"{', '.join(job.missing_information)} — not stated in the post.]"
        )
    lines += ["", "Best regards,", profile["name"].replace(" (fictional)", "")]
    return "\n".join(lines)


def pick_sample(cfg: dict) -> dict:
    """Pick one strong, Amina-relevant post to run both demos on (deterministic).

    We look for a junior data/software role that mentions one of Amina's skills, so the
    grounded extraction and the drafted message both have something real to show. Falls
    back to the first post if nothing matches. We do NOT rely on is_opportunity here —
    that column carries label noise; the demo is about the text, not the label."""
    df = pd.read_csv(resolve(cfg["paths"]["raw_jobs"]), dtype=str,
                     keep_default_na=False).sort_values("id")
    profile = _load_profile(cfg)
    skills = [s.lower() for s in profile["skills"]]
    for _, r in df.iterrows():
        title = str(r["title"]).lower()
        desc = str(r["description"]).lower()
        is_junior_tech = ("data analyst" in title or "developer" in title) and \
                         "junior" in title
        if is_junior_tech and any(s in desc for s in skills):
            return r.to_dict()
    return df.iloc[0].to_dict()


def run_demo(cfg: dict | None = None, verbose: bool = True,
             job_record: dict | None = None, profile: dict | None = None) -> dict:
    cfg = cfg or load_config()
    profile = profile or load_profile(cfg)
    job_record = job_record or pick_sample(cfg)

    vague = summarise_vague(job_record, cfg)
    grounded = extract_grounded(job_record, cfg)
    message = draft_message(grounded, profile)

    if verbose:
        print("=" * 60)
        print("GENAI APPLICATION ASSISTANT  (offline stub unless provider=openai)")
        print("=" * 60)
        print("JOB POST (unstructured input):")
        print(job_context(job_record) + "\n")
        print("-" * 60)
        print("DEMO A — VAGUE PROMPT (fluent, but watch the facts):")
        print(vague + "\n")
        print("-" * 60)
        print("DEMO B — GROUNDED PROMPT (structured, validated, honest):")
        print(json.dumps(grounded.model_dump(), indent=2))
        print("\n    ^ 'missing_information' turns silent guessing into a visible list.")
        print("-" * 60)
        print("DRAFTED MESSAGE (gaps shown, not invented):")
        print(message)
        print("=" * 60)

    return {"vague": vague, "grounded": grounded, "message": message,
            "job_record": job_record}


if __name__ == "__main__":
    run_demo()
