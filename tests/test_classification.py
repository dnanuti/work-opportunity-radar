from datetime import date

import pandas as pd

from src.classification import LIKELY, NOT_CURRENT, UNCERTAIN, classify_frame


def test_talk_examples_keep_evidence_and_uncertainty_visible():
    examples = pd.DataFrame([
        {"title": "Junior Data Analyst", "company": "Acacia", "url": "https://x/apply",
         "deadline": "2026-09-01", "description": "We are hiring an analyst role."},
        {"title": "Our data team is growing", "company": "", "url": "", "deadline": "",
         "description": "Our data team is growing - DM me."},
        {"title": "Graduate careers event", "company": "Community", "url": "https://x/event",
         "deadline": "2026-06-01", "description": "The career fair event has ended."},
    ])
    result = classify_frame(examples, today=date(2026, 8, 8))
    assert result["opportunity_status"].tolist() == [LIKELY, UNCERTAIN, NOT_CURRENT]
    assert result["status_why"].str.len().gt(0).all()

