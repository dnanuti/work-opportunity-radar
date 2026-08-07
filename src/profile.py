"""Candidate-profile helpers and a small cross-platform setup wizard.

The repository ships with fictional Amina. Running ``python run.py configure`` creates
``profiles/candidate.json`` locally; that file is git-ignored and automatically takes
precedence. The profile intentionally contains job-search preferences, not a CV or
contact details.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src import PROJECT_ROOT, load_config, rel, resolve

LOCAL_PROFILE = PROJECT_ROOT / "profiles" / "candidate.json"
ALLOWED_LEVELS = {
    "internship", "graduate", "entry_level", "junior", "mid_level", "senior",
}


def _items(value: str) -> list[str]:
    """Turn a comma-separated answer into a clean, de-duplicated list."""
    seen: set[str] = set()
    result: list[str] = []
    for item in (part.strip() for part in value.split(",")):
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def profile_path(cfg: dict | None = None) -> Path:
    """Resolve the active profile: env override, local candidate, then Amina."""
    cfg = cfg or load_config()
    override = os.environ.get("RADAR_PROFILE")
    if override:
        return resolve(override)
    if LOCAL_PROFILE.exists():
        return LOCAL_PROFILE
    return resolve(cfg["paths"]["profile"])


def load_profile(cfg: dict | None = None) -> dict:
    path = profile_path(cfg)
    with open(path, encoding="utf-8") as stream:
        profile = json.load(stream)
    required = ["name", "location", "skills", "location_preference"]
    missing = [field for field in required if not profile.get(field)]
    if missing:
        raise ValueError(f"Profile {rel(path)} is missing: {', '.join(missing)}")
    return profile


def build_profile(
    name: str,
    location: str,
    skills: str,
    target_roles: str,
    location_preference: str,
    preferred_levels: str,
) -> dict:
    levels = [item.casefold().replace(" ", "_") for item in _items(preferred_levels)]
    invalid = sorted(set(levels) - ALLOWED_LEVELS)
    if invalid:
        allowed = ", ".join(sorted(ALLOWED_LEVELS))
        raise ValueError(f"Unknown level(s): {', '.join(invalid)}. Use: {allowed}")
    return {
        "note": "Local candidate profile. Keep contact details and CV content out of this file.",
        "name": name.strip() or "Candidate",
        "location": location.strip(),
        "location_preference": _items(location_preference),
        "career_stage": levels[0] if levels else "entry_level",
        "preferred_levels": levels,
        "skills": _items(skills),
        "target_roles": _items(target_roles),
        "interests": [],
    }


def configure(input_fn=input, output_path: Path | None = None) -> Path:
    """Ask six non-sensitive questions and save a local candidate profile."""
    print("Create your candidate profile (comma-separate lists).")
    print("Use a nickname if you prefer; do not paste a CV, email, phone, or API key.\n")
    profile = build_profile(
        name=input_fn("Name or nickname: "),
        location=input_fn("Current city and country: "),
        skills=input_fn("Skills (for example Python, SQL, Git): "),
        target_roles=input_fn("Target roles (for example Data Analyst, QA Engineer): "),
        location_preference=input_fn("Preferred locations (include Remote if wanted): "),
        preferred_levels=input_fn(
            "Levels (internship, graduate, entry_level, junior, mid_level, senior): "
        ),
    )
    path = output_path or LOCAL_PROFILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSaved {rel(path)}. Future commands will use this profile automatically.")
    return path

