"""Work Opportunity Radar — rank real (unlabelled) job posts by fit to a candidate.

Notebook 02 trains a *supervised* model, which needs labels. Real collected feeds have
none. So the radar scores fit with a transparent, **explainable rule** instead — no
training, every point of the score is attributable to a reason you can read out loud.
That honesty is the whole ethos of the talk: a simpler method you can explain beats a
fancy one you can't.

fit_score (0-1) blends:
  * skill overlap with the candidate's skills
  * target-role similarity
  * location fit (preferred city or remote)
  * the candidate's preferred experience levels
  * salary transparency (a small nudge)
  * recency of the post

Output: a ranked table + a one-line 'why' per post + data/processed/matches.csv.
Run: python -m src.radar
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src import load_config, rel, resolve
from src.classification import NOT_CURRENT, classify_frame
from src.cleaning import clean_frame, normalise_text
from src.collect.africa import canonical_location, is_remote
from src.profile import load_profile

WEIGHTS = {"skills": 0.35, "target_role": 0.15, "location": 0.20,
           "level": 0.15, "salary": 0.05, "recency": 0.10}

_SENIOR_MARKERS = ["senior", "staff", "lead", "principal", "manager", "head of",
                   "10+ years", "10 years", "6+ years", "5 years", "5+ years"]
_JUNIOR_MARKERS = ["junior", "jr.", "jr ", "graduate", "intern", "entry", "trainee",
                   "new grad"]


_LEVEL_DEFAULTS = {
    "student": {"internship", "entry_level"},
    "new_graduate": {"internship", "graduate", "entry_level", "junior"},
    "graduate": {"graduate", "entry_level", "junior"},
    "entry_level": {"entry_level", "junior"},
    "junior": {"entry_level", "junior"},
    "mid_level": {"mid_level"},
    "senior": {"senior"},
}


def infer_level(title: str, description: str = "") -> str:
    """Infer a coarse level for transparent comparison with candidate preferences."""
    title_text = normalise_text(title)
    all_text = normalise_text(f"{title} {description}")
    if any(marker in title_text for marker in ("intern", "internship", "trainee")):
        return "internship"
    if any(marker in title_text for marker in ("graduate", "new grad")):
        return "graduate"
    if any(marker in title_text for marker in _JUNIOR_MARKERS):
        return "junior"
    if any(marker in title_text for marker in _SENIOR_MARKERS):
        return "senior"
    if any(marker in all_text for marker in ("3+ years", "3 years", "4+ years",
                                              "4 years", "mid-level", "mid level")):
        return "mid_level"
    if any(marker in all_text for marker in _SENIOR_MARKERS):
        return "senior"
    return "unknown"


def _target_role_score(title: str, targets: list[str]) -> tuple[float, str | None]:
    """Token overlap keeps role preferences inspectable instead of hiding embeddings."""
    if not targets:
        return 0.5, None
    stop = {"junior", "graduate", "senior", "entry", "level", "intern", "internship"}
    title_tokens = set(normalise_text(title).replace("-", " ").split()) - stop
    best_score, best_target = 0.0, None
    for target in targets:
        target_tokens = set(normalise_text(target).replace("-", " ").split()) - stop
        score = len(title_tokens & target_tokens) / len(target_tokens) if target_tokens else 0.0
        if score > best_score:
            best_score, best_target = score, target
    return min(best_score, 1.0), best_target


def _score_row(row: pd.Series, profile: dict, today) -> tuple[float, list[str]]:
    text = normalise_text(f"{row.get('title', '')} {row.get('description', '')}")
    reasons: list[str] = []

    # --- skills: share of the candidate's skills the post mentions ---
    prof_skills = [s for s in profile.get("skills", [])]
    matched = [s for s in prof_skills if normalise_text(s) in text]
    skill_score = len(matched) / len(prof_skills) if prof_skills else 0.0
    if matched:
        reasons.append(f"matches your skills: {', '.join(matched)}")
    else:
        reasons.append("no listed skills found in the post")

    # --- location: preferred city, or remote if the candidate accepts remote ---
    canon = canonical_location(str(row.get("location", "")))
    prefs = [normalise_text(p) for p in profile.get("location_preference", [])]
    remote_ok = "remote" in prefs
    loc_score = 0.0
    if canon != "Unknown" and any(p in normalise_text(canon) for p in prefs if p):
        loc_score = 1.0
        reasons.append(f"location fits ({canon})")
    elif (canon == "Remote" or is_remote(text)) and remote_ok:
        loc_score = 1.0
        reasons.append("remote — reachable from your location")
    else:
        reasons.append(f"location may not fit ({canon})")

    # --- target role ---
    title_text = normalise_text(str(row.get("title", "")))
    role_score, target = _target_role_score(
        str(row.get("title", "")), profile.get("target_roles", [])
    )
    if target and role_score > 0:
        reasons.append(f"similar to target role: {target}")
    elif profile.get("target_roles"):
        reasons.append("title is outside your stated target roles")

    # --- experience level: candidate-specific, not hard-coded to junior ---
    job_level = infer_level(str(row.get("title", "")), str(row.get("description", "")))
    preferred = set(profile.get("preferred_levels", []))
    if not preferred:
        preferred = _LEVEL_DEFAULTS.get(profile.get("career_stage", ""), set())
    if job_level == "unknown":
        level_score = 0.5
        reasons.append("experience level not clearly stated")
    elif job_level in preferred:
        level_score = 1.0
        reasons.append(f"experience level fits ({job_level.replace('_', ' ')})")
    else:
        level_score = 0.0
        reasons.append(f"experience level may not fit ({job_level.replace('_', ' ')})")

    # --- salary transparency (small nudge; hidden salary is a real-world red flag) ---
    salary_score = 1.0 if str(row.get("salary", "")).strip() else 0.0
    if not salary_score:
        reasons.append("no salary stated")

    # --- recency ---
    recency_score = 0.5
    parsed = pd.to_datetime(row.get("source_post_date", ""), errors="coerce",
                            format="mixed", utc=True)
    if pd.notna(parsed):
        age = (pd.Timestamp(today, tz="UTC") - parsed).days
        recency_score = 1.0 if age <= 30 else 0.5 if age <= 60 else 0.2
        if age <= 30:
            reasons.append("posted recently")

    fit = (WEIGHTS["skills"] * skill_score + WEIGHTS["target_role"] * role_score
           + WEIGHTS["location"] * loc_score + WEIGHTS["level"] * level_score
           + WEIGHTS["salary"] * salary_score
           + WEIGHTS["recency"] * recency_score)
    return round(float(fit), 3), reasons


def rank_opportunities(df: pd.DataFrame, profile: dict, today=None) -> pd.DataFrame:
    """Add fit_score + why to each post and return them sorted best-first."""
    today = today or datetime.now().date()
    scores, whys = [], []
    for _, row in df.iterrows():
        fit, reasons = _score_row(row, profile, today)
        scores.append(fit)
        whys.append("; ".join(reasons))
    out = df.copy()
    out["fit_score"] = scores
    out["why"] = whys
    return out.sort_values("fit_score", ascending=False).reset_index(drop=True)


def build_radar(df: pd.DataFrame, profile: dict, today=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify first, then rank current/uncertain posts for this candidate.

    Returns ``(ranked, excluded)`` so discarded evidence remains visible.
    """
    classified = classify_frame(df, today=today)
    excluded = classified[classified["opportunity_status"] == NOT_CURRENT].copy()
    eligible = classified[classified["opportunity_status"] != NOT_CURRENT].copy()
    ranked = rank_opportunities(eligible, profile, today=today)
    return ranked, excluded


def run(cfg: dict | None = None, verbose: bool = True, top_n: int = 10) -> pd.DataFrame:
    cfg = cfg or load_config()
    collected = resolve(cfg["paths"]["collected_jobs"])
    if not collected.exists():
        # Nothing collected yet — do it now (offline sample by default).
        from src.collect import fetch
        fetch(cfg, verbose=False)
    raw = pd.read_csv(collected, dtype=str, keep_default_na=False)

    clean_df, _ = clean_frame(raw, has_labels=False)
    profile = load_profile(cfg)
    ranked, excluded = build_radar(clean_df, profile)

    out_path = resolve(cfg["paths"]["matches"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_cols = [c for c in ["fit_score", "opportunity_status", "status_uncertainty",
                               "title", "company", "location_clean", "is_remote",
                               "salary", "url", "why", "status_why"]
                   if c in ranked.columns]
    ranked[export_cols].to_csv(out_path, index=False)

    if verbose:
        _report(ranked, profile, out_path, top_n, len(excluded))
    return ranked


def _report(ranked: pd.DataFrame, profile: dict, out_path, top_n: int,
            excluded_count: int = 0) -> None:
    name = profile.get("name", "the candidate").replace(" (fictional)", "")
    print("=" * 72)
    print(f"WORK OPPORTUNITY RADAR  —  ranked for {name}")
    print("=" * 72)
    print(f"Ranked {len(ranked)} current/uncertain posts; kept {excluded_count} not-current "
          f"record(s) out of the ranking.\nFull ranking -> {rel(out_path)}\n")
    for i, row in ranked.head(top_n).iterrows():
        loc = row.get("location_clean", row.get("location", ""))
        tag = "  [remote]" if int(row.get("is_remote", 0) or 0) else ""
        print(f"{i+1:>2}. {row['fit_score']:.2f}  {row['title']}  @ {row['company']}"
              f"  ({loc}){tag}")
        print(f"       status: {row.get('opportunity_status', 'unclassified')} — "
              f"{row.get('status_why', '')}")
        print(f"       match : {row['why']}")
    print("=" * 72)
    print("Note: fit_score is a transparent rule, not a trained model — every point is\n"
          "explained in 'why'. Real feeds have no labels, so we rank by fit, not predict.")
    print("=" * 72)


if __name__ == "__main__":
    run()
