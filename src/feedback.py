"""Append-only, reviewable human feedback for the continuous-learning loop."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src import load_config, resolve


def record_feedback(job_id: str, relevant: bool | None, reason: str = "",
                    path: str | Path | None = None) -> dict:
    """Record judgement as feedback, not as automatic training truth."""
    if relevant not in (True, False, None):
        raise ValueError("relevant must be True, False, or None (unsure)")
    cfg = load_config()
    target = Path(path) if path else resolve(cfg["paths"]["feedback"])
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "job_id": str(job_id),
        "relevant": relevant,
        "reason": reason.strip(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "needs_review",
    }
    with open(target, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row

