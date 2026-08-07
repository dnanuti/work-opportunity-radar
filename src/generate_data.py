"""Deterministic synthetic job-post generator.

Builds data/raw/jobs.csv — a scraped-style dataset that DELIBERATELY contains the
messes real data has: exact and near-duplicates, missing values, inconsistent category
spellings, mixed date formats, free-text descriptions, and a little label noise.

Everything is driven by config.yaml and a single seed, so two laptops produce a
byte-identical file. Run:  python -m src.generate_data
"""
from __future__ import annotations

import csv
import random

from src import load_config, rel, resolve, seed_everything

# --- Building blocks for realistic-but-fake posts --------------------------------

ROLES = [
    ("Junior Data Analyst", ["SQL", "Excel", "Python"], True),
    ("Data Analyst", ["SQL", "Python", "Tableau"], True),
    ("Junior Software Developer", ["JavaScript", "Git", "HTML"], True),
    ("Backend Developer (Junior)", ["Python", "SQL", "Git"], True),
    ("Graduate Software Engineer", ["Python", "Git", "JavaScript"], True),
    ("Frontend Developer", ["JavaScript", "CSS", "React"], True),
    ("Data Science Intern", ["Python", "SQL", "statistics"], True),
    # These are genuine jobs even when they are irrelevant to Amina. Classification
    # asks "is it an opportunity?"; candidate-aware ranking asks "is it for me?".
    ("IT Support Technician", ["networking", "hardware", "Windows"], True),
    ("Senior Staff Engineer", ["distributed systems", "10y experience"], True),
    ("Sales Representative", ["communication", "CRM"], True),
    ("Warehouse Operative", ["forklift", "logistics"], True),
    ("Marketing Manager", ["SEO", "5y experience", "strategy"], True),
    # Non-opportunities are job-adjacent content, not simply jobs for someone else.
    ("Career Fair Recap", ["networking"], False),
    ("Python Bootcamp", ["Python"], False),
    ("Data Team Update", ["SQL"], False),
    ("CV Workshop", ["communication"], False),
]

COMPANIES = [
    "Acacia Analytics", "Savanna Software", "Rift Valley Tech", "Baobab Labs",
    "Harambee Digital", "Nile Data Co", "Sahara Systems", "Zanzibar Cloud",
    "Kilimanjaro AI", "Lagos Logic", "Accra Apps", "Kampala Code",
]

# Same city, written many different ways — the normalisation lesson.
LOCATION_VARIANTS = {
    "Nairobi": ["Nairobi", "Nairobi, KE", "nairobi", "Nairobi, Kenya"],
    "Remote": ["Remote", "remote", "REMOTE", "Fully remote", "Remote (Africa)"],
    "Lagos": ["Lagos", "Lagos, NG", "Lagos, Nigeria"],
    "Accra": ["Accra", "Accra, GH", "accra"],
    "Kampala": ["Kampala", "Kampala, UG"],
}

DESCRIPTION_TEMPLATES = [
    "We are hiring a {role} to join {company}. You will work with {skills}. "
    "This is a great role for someone early in their career.",
    "{company} is looking for a {role}. Ideal candidates know {skills}. "
    "Apply with your CV.",
    "Join {company} as a {role}! Day to day you'll use {skills}. "
    "We value curiosity over years of experience.",
    "{role} wanted at {company}. Must be comfortable with {skills}. "
    "Send us a short note about a project you built.",
]

NON_OPPORTUNITY_TEMPLATES = [
    "This {role} event took place last month at {company}. Registration has ended.",
    "Join {company}'s paid {role}. Learn {skills}; this is a course, not a job vacancy.",
    "{company} shared a {role}: our team is growing. Follow us for news; no role or application link was provided.",
    "A recording from the {role} hosted by {company}. The event has ended.",
]


def _make_deadline(rng: random.Random, unparseable_rate: float) -> str:
    """Return a deadline in a MIX of formats, sometimes unparseable on purpose."""
    day = rng.randint(1, 28)
    month = rng.randint(1, 12)
    year = 2026
    style = rng.random()
    if style < unparseable_rate:
        return rng.choice(["next Friday", "end of month", "ASAP", "rolling"])
    if style < 0.5:
        return f"{year}-{month:02d}-{day:02d}"          # ISO
    return f"{day:02d}/{month:02d}/{year}"              # DD/MM/YYYY


def _base_posts(cfg: dict, rng: random.Random) -> list[dict]:
    n = cfg["data"]["n_base_jobs"]
    unparseable = cfg["data"]["unparseable_date_rate"]
    posts = []
    for i in range(n):
        role, skills, is_opp = rng.choice(ROLES)
        company = rng.choice(COMPANIES)
        canonical_loc = rng.choice(list(LOCATION_VARIANTS.keys()))
        location = rng.choice(LOCATION_VARIANTS[canonical_loc])
        skills_str = ", ".join(skills)
        templates = DESCRIPTION_TEMPLATES if is_opp else NON_OPPORTUNITY_TEMPLATES
        description = rng.choice(templates).format(
            role=role, company=company, skills=skills_str
        )
        # A stable posting reference so each genuine base post is distinct, while a
        # re-post (exact or near-dup) copies this text and stays identifiable as the
        # SAME posting. Without it, templated text makes unrelated posts look identical.
        description = f"{description} (Posting ref: {i})"
        salary = rng.choice([40000, 55000, 60000, 75000, 90000, 120000])
        posts.append(
            {
                "id": f"job_{i:04d}",
                "title": role,
                "company": company,
                "location": location,
                "salary": salary,
                "deadline": _make_deadline(rng, unparseable),
                "description": description,
                "source_post_date": f"2026-{rng.randint(1,7):02d}-{rng.randint(1,28):02d}",
                "is_opportunity": int(is_opp),
                "url": f"https://example.org/jobs/{i}" if is_opp else "",
            }
        )
    return posts


def _make_near_dup(post: dict, rng: random.Random, new_id: str) -> dict:
    """A near-duplicate: same content, tiny edits that defeat naive drop_duplicates()."""
    dup = dict(post)
    dup["id"] = new_id
    # Small, realistic mutations.
    dup["title"] = dup["title"].replace("Junior", "Jr.")
    dup["description"] = "  " + dup["description"].replace("  ", " ") + " "  # whitespace noise
    if rng.random() < 0.5:
        dup["location"] = dup["location"].upper()
    return dup


def _inject_duplicates(posts: list[dict], cfg: dict, rng: random.Random) -> list[dict]:
    dup_rate = cfg["data"]["duplicate_rate"]
    near_frac = cfg["data"]["near_dup_fraction"]
    n_dups = int(len(posts) * dup_rate)
    out = list(posts)
    next_idx = len(posts)
    for _ in range(n_dups):
        original = rng.choice(posts)
        new_id = f"job_{next_idx:04d}"
        next_idx += 1
        if rng.random() < near_frac:
            out.append(_make_near_dup(original, rng, new_id))
        else:
            exact = dict(original)          # verbatim re-post, new id only
            exact["id"] = new_id
            out.append(exact)
    return out


def _inject_missing(posts: list[dict], cfg: dict, rng: random.Random) -> None:
    miss = cfg["data"]["missing"]
    for p in posts:
        for col, rate in miss.items():
            if rng.random() < rate:
                p[col] = ""          # empty string = missing in a CSV


def _inject_label_noise(posts: list[dict], cfg: dict, rng: random.Random) -> int:
    rate = cfg["data"]["label_noise"]
    flipped = 0
    for p in posts:
        if rng.random() < rate:
            p["is_opportunity"] = 1 - int(p["is_opportunity"])
            flipped += 1
    return flipped


FIELDNAMES = [
    "id", "title", "company", "location", "salary", "deadline",
    "description", "source_post_date", "is_opportunity", "url",
]


def generate(cfg: dict | None = None) -> list[dict]:
    """Generate the full messy dataset and write it to the raw path. Returns the rows."""
    cfg = cfg or load_config()
    seed_everything(cfg["seed"])
    rng = random.Random(cfg["seed"])

    posts = _base_posts(cfg, rng)
    n_unique = len(posts)
    # Apply label noise BEFORE duplication so a re-posted job keeps the same (sometimes
    # wrong) label as its twin. That shared label is exactly what a leaky split lets the
    # model memorise — and what an honest, grouped split refuses to reward.
    n_flipped = _inject_label_noise(posts, cfg, rng)
    posts = _inject_duplicates(posts, cfg, rng)
    n_after_dup = len(posts)
    _inject_missing(posts, cfg, rng)

    out_path = resolve(cfg["paths"]["raw_jobs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(posts)

    _print_report(posts, n_unique, n_after_dup, n_flipped, out_path)
    return posts


def _print_report(posts, n_unique, n_after_dup, n_flipped, out_path) -> None:
    """The 'data problems report' — the mess, quantified, for the audience."""
    n = len(posts)
    nulls = {c: sum(1 for p in posts if p[c] == "") for c in FIELDNAMES}
    loc_spellings = sorted({p["location"] for p in posts if p["location"] != ""})
    print("=" * 60)
    print("DATA PROBLEMS REPORT  (this is the job, not the exception)")
    print("=" * 60)
    print(f"Rows written              : {n}  -> {rel(out_path)}")
    print(f"Unique base posts         : {n_unique}")
    print(f"After duplicate injection : {n_after_dup}  (+{n_after_dup - n_unique} dupes)")
    print(f"Labels flipped (noise)    : {n_flipped}")
    print("\nMissing values per column :")
    for c in ["salary", "deadline", "location"]:
        print(f"    {c:<10}: {nulls[c]:>3} missing  ({nulls[c]/n:.0%})")
    print(f"\nDistinct 'location' spellings on disk : {len(loc_spellings)}")
    print("    e.g. " + ", ".join(repr(s) for s in loc_spellings[:6]) + " ...")
    print("=" * 60)


if __name__ == "__main__":
    generate()
