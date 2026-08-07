"""Manual ChatGPT bridge: prepare a prompt, copy/paste, then validate locally."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import load_config, rel, resolve
from src.application_assistant import draft_message, job_context, validate_extraction
from src.cleaning import clean_frame
from src.collect import fetch
from src.evidence import build_evidence_card, format_evidence_card
from src.profile import load_profile
from src.radar import build_radar


def _top_job(cfg: dict) -> dict:
    collected = resolve(cfg["paths"]["collected_jobs"])
    if not collected.exists():
        fetch(cfg, verbose=False)
    jobs = pd.read_csv(collected, dtype=str, keep_default_na=False)
    clean, _ = clean_frame(jobs, has_labels=False)
    ranked, _ = build_radar(clean, load_profile(cfg))
    return ranked.iloc[0].to_dict()


def build_prompt(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    template = resolve(cfg["paths"]["grounded_prompt"]).read_text(encoding="utf-8")
    return template.replace("{job_post}", job_context(_top_job(cfg)))


def write_prompt(cfg: dict | None = None) -> Path:
    out = resolve("output/chatgpt_prompt.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_prompt(cfg), encoding="utf-8")
    print(f"Prompt written to {rel(out)}")
    print("1. Open ChatGPT and paste the entire prompt.")
    print("2. Copy only the returned JSON into output/chatgpt_response.json.")
    print("3. Run: python run.py validate output/chatgpt_response.json")
    return out


def _parse_json_response(text: str) -> dict:
    """Accept plain JSON or the common ```json fenced response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def validate_response_file(path: str | Path, cfg: dict | None = None):
    cfg = cfg or load_config()
    raw = _parse_json_response(Path(path).read_text(encoding="utf-8"))
    result = validate_extraction(raw)
    print("Valid grounded response:\n")
    print(json.dumps(result.model_dump(), indent=2))
    job = _top_job(cfg)
    print("\n" + format_evidence_card(build_evidence_card(job, result)))
    print("\nDRAFT FOR HUMAN REVIEW\n")
    print(draft_message(result, load_profile(cfg)))
    return result


def interactive_chatgpt(cfg: dict | None = None):
    """Cross-platform copy/paste flow; no browser automation and no API key."""
    cfg = cfg or load_config()
    prompt_path = write_prompt(cfg)
    print(f"\nOpen {rel(prompt_path)}, paste it into ChatGPT, then copy the JSON reply.")
    print("Paste the reply below. On a new line type END and press Enter.\n")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    out = resolve("output/chatgpt_response.json")
    out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return validate_response_file(out, cfg)

