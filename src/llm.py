"""Offline-first language-model adapter used by the teaching demo.

``offline`` is deterministic and needs no account. ``openai`` is optional and falls
back safely when a key, package, network, or valid response is unavailable.
"""
from __future__ import annotations

import os
import re

from src import load_config

SKILL_VOCAB = [
    "Python", "SQL", "JavaScript", "Git", "Excel", "Tableau", "React", "CSS",
    "HTML", "statistics", "networking", "hardware", "Windows", "CRM", "SEO",
    "distributed systems", "logistics", "forklift", "strategy", "communication",
]

OPPORTUNITY_TITLES = [
    "data analyst", "software developer", "software engineer", "frontend developer",
    "backend developer", "data science intern", "developer", "analyst",
]


def _extract_job_post(prompt: str) -> str:
    for marker in ("JOB POST:", "Job post:"):
        idx = prompt.rfind(marker)
        if idx != -1:
            return prompt[idx + len(marker):].strip()
    return prompt.strip()


def _labelled_context(job_post: str) -> dict[str, str]:
    values: dict[str, str] = {}
    labels = {"Title", "Company", "Location", "Salary", "Deadline", "Published",
              "Source", "Application URL", "Description"}
    for line in job_post.splitlines():
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        if label in labels:
            values[label] = value.strip()
    if "Description" not in values:
        values["Description"] = job_post.strip()
    return values


def _offline_grounded(job_post: str) -> dict:
    context = _labelled_context(job_post)
    description = context.get("Description", "")
    title = context.get("Title", "")
    text = f"{title} {description}".lower()
    skills = [skill for skill in SKILL_VOCAB if skill.lower() in text]

    role = title or None
    if role is None:
        for candidate in OPPORTUNITY_TITLES:
            if candidate in text:
                role = candidate.title()
                break

    level = None
    level_text = title.lower()
    for value, markers in {
        "internship": ("intern", "internship", "trainee"),
        "graduate": ("graduate", "new grad"),
        "junior": ("junior", "jr."),
        "senior": ("senior", "staff", "lead", "principal"),
    }.items():
        if any(marker in level_text for marker in markers):
            level = value
            break
    if level is None:
        for value, markers in {
            "internship": ("internship", "trainee"),
            "graduate": ("new graduate", "graduate programme"),
            "junior": ("junior role", "entry level"),
            "senior": ("senior role", "staff level", "lead role"),
        }.items():
            if any(marker in description.lower() for marker in markers):
                level = value
                break

    event_or_course = any(word in text for word in (
        "event", "workshop", "bootcamp", "course", "registration has ended"
    ))
    is_opportunity = bool(role) and any(
        candidate in text for candidate in OPPORTUNITY_TITLES
    ) and not event_or_course

    # Keep publication date and deadline separate. A random date is not a closing date.
    deadline = context.get("Deadline") or None
    if deadline is None:
        match = re.search(
            r"(?:deadline|closing date|apply by)\D{0,12}"
            r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})",
            description, re.IGNORECASE,
        )
        deadline = match.group(1) if match else None

    location = context.get("Location") or None
    salary = context.get("Salary") or None
    application_link = context.get("Application URL") or None

    missing = []
    if deadline is None:
        missing.append("closing_date")
    if application_link is None and not re.search(r"https?://|apply at|email|@", text):
        missing.append("application_link")
    if salary is None:
        missing.append("salary")
    if location is None:
        missing.append("location")
    if not skills:
        missing.append("required_skills")

    if role and skills and deadline:
        confidence = "high"
    elif role and skills:
        confidence = "medium"
    else:
        confidence = "low"

    evidence = {
        "role": None, "level": None, "skills": None, "location": None,
        "salary": None, "deadline": None, "application_link": None,
    }
    # The short fixture descriptions are themselves a single exact source fragment and
    # may mention different skills in adjacent sentences; keeping them together proves
    # every extracted skill instead of citing only the first one.
    skill_evidence = (description or None) if skills else None
    for field, value in {
        "role": role,
        "level": title if level else None,
        "skills": skill_evidence,
        "location": location,
        "salary": salary,
        "deadline": deadline,
        "application_link": application_link,
    }.items():
        if value:
            evidence[field] = value

    return {
        "is_opportunity": bool(is_opportunity),
        "role": role,
        "level": level,
        "skills": skills,
        "location": location,
        "salary": salary,
        "deadline": deadline,
        "application_link": application_link,
        "confidence": confidence,
        "missing_information": missing,
        "evidence": evidence,
    }


def _offline_vague(job_post: str) -> str:
    """Fluent, confident, and deliberately invented: the anti-pattern."""
    role_match = re.search(
        r"(Junior|Graduate|Senior)\s+[A-Za-z]+(?:\s[A-Za-z]+)?\s"
        r"(Analyst|Developer|Engineer)", job_post,
    )
    role = role_match.group(0).strip() if role_match else "this role"
    return (
        f"Great news - you're a perfect fit for {role}! The position offers a "
        f"competitive salary of $72,000 and the closing date is 2026-09-15. "
        f"Simply email your CV to careers@company.example.\n"
        f"[DEMO WARNING: those salary, date, and contact claims were not grounded in "
        f"the supplied post.]"
    )


def complete(prompt: str, schema: dict | None = None, cfg: dict | None = None):
    cfg = cfg or load_config()
    if cfg["llm"]["provider"] == "openai":
        result = _try_openai(prompt, schema, cfg)
        if result is not None:
            return result

    job_post = _extract_job_post(prompt)
    return _offline_grounded(job_post) if schema is not None else _offline_vague(job_post)


def _try_openai(prompt: str, schema: dict | None, cfg: dict):
    key = os.environ.get(cfg["llm"]["api_key_env"])
    if not key:
        print(f"[llm] WARNING: {cfg['llm']['api_key_env']} not set - using offline mode.")
        return None
    try:
        from openai import OpenAI
    except ImportError:
        print("[llm] WARNING: optional 'openai' package not installed - using offline mode.")
        return None
    try:
        client = OpenAI(api_key=key)
        if schema is not None:
            response = client.responses.parse(
                model=cfg["llm"]["api_model"], input=prompt, text_format=schema,
            )
            if response.output_parsed is None:
                raise ValueError("model returned no structured output")
            parsed = response.output_parsed
            return parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
        response = client.responses.create(model=cfg["llm"]["api_model"], input=prompt)
        return response.output_text
    except Exception as error:  # noqa: BLE001 - live demo must fall back safely
        print(f"[llm] WARNING: API call failed ({error}) - using offline mode.")
        return None
