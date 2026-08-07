"""Opportunity classification + the data-leakage demo.

The matcher is a nearest-neighbour classifier: to score a post it finds the most
similar post it has already seen and copies that label. Intuitive — and exactly why
leakage bites. Features:

  * interpretable post-evidence features (job language, application signal, ...), and
  * a bag-of-words view of the post description — the text the neighbour search runs on.

We score the SAME model on the SAME features two ways, changing only the split:

  * naive split  — a plain random split. Near-duplicate posts land in BOTH train and
                   test, so a test post's identical twin is sitting in the training set:
                   the model just echoes the twin's label and looks brilliant.
  * honest split — a grouped split on content_key, so no variant of the same post can
                   appear on both sides. Now the neighbour is a genuinely different post.
                   This is the real score.

The gap between the two is the lie. For the 'why did it decide that' readout we also fit
a small decision tree on the interpretable features alone. Run: python -m src.matching
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src import load_config, resolve, seed_everything
from src.cleaning import content_key, normalise_text

# The interpretable features, in column order (used for the 'why' readout).
CLASSIFICATION_FEATURES = [
    "has_company", "has_source_link", "has_apply_language", "has_job_language",
    "event_or_course_signal", "desc_len_norm",
]


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add candidate-independent evidence features, label, and leakage group key."""
    rows = []
    for _, r in df.iterrows():
        text = normalise_text(f"{r['title']} {r['description']}")
        rows.append(
            {
                "has_company": int(str(r.get("company", "")).strip() != ""),
                "has_source_link": int(str(r.get("url", "")).strip() != ""),
                "has_apply_language": int(any(
                    phrase in text for phrase in ("apply", "send us", "hiring", "vacancy")
                )),
                "has_job_language": int(any(re.search(rf"\b{word}\b", text) for word in (
                    "job", "role", "developer", "engineer", "analyst", "technician",
                    "operative", "representative", "manager"
                ))),
                "event_or_course_signal": int(any(
                    phrase in text for phrase in ("event", "workshop", "bootcamp",
                                                  "course", "registration has ended")
                )),
                "desc_len_norm": len(str(r["description"])) / 200.0,
                "description": str(r["description"]),
                "is_opportunity": int(r["is_opportunity"]),
                "content_key": content_key(r),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class MatchingResults:
    naive_score: float
    honest_score: float
    naive_f1: float
    honest_f1: float
    confusion: np.ndarray            # honest split: [[TN, FP], [FN, TP]]
    coefficients: dict = field(default_factory=dict)

    @property
    def gap(self) -> float:
        return self.naive_score - self.honest_score


def _score_split(frame, tr_idx, te_idx):
    """Fit the text vectoriser on TRAIN ONLY (so vocabulary itself doesn't leak), stack
    it with the dense profile features, and score a 1-NN matcher. Returns (yte, preds)."""
    dense = frame[CLASSIFICATION_FEATURES].to_numpy(dtype=float)

    vec = CountVectorizer(min_df=1)
    bow_tr = vec.fit_transform(frame.iloc[tr_idx]["description"])
    bow_te = vec.transform(frame.iloc[te_idx]["description"])

    Xtr = hstack([csr_matrix(dense[tr_idx]), bow_tr]).tocsr()
    Xte = hstack([csr_matrix(dense[te_idx]), bow_te]).tocsr()
    ytr = frame.iloc[tr_idx]["is_opportunity"].to_numpy()
    yte = frame.iloc[te_idx]["is_opportunity"].to_numpy()

    # 1-NN: copy the label of the single most similar post seen in training. If an
    # identical twin is in the training set (leakage), that twin IS the neighbour.
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(Xtr, ytr)
    return yte, model.predict(Xte)


def run_experiment(cfg: dict | None = None, verbose: bool = True) -> MatchingResults:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    seed_everything(seed)

    # Use the RAW (still-duplicated) data on purpose — the duplicates are the point.
    raw = pd.read_csv(resolve(cfg["paths"]["raw_jobs"]), dtype=str,
                      keep_default_na=False)
    frame = build_frame(raw)

    y = frame["is_opportunity"].to_numpy()
    groups = frame["content_key"].to_numpy()
    idx = np.arange(len(frame))
    test_size = cfg["split"]["test_size"]

    # --- Naive split: plain random. Near-dups straddle train/test -> leakage. ---
    n_tr, n_te = train_test_split(idx, test_size=test_size, random_state=seed,
                                  stratify=y)
    yn_te, pn = _score_split(frame, n_tr, n_te)
    naive_acc = accuracy_score(yn_te, pn)
    naive_f1 = f1_score(yn_te, pn, zero_division=0)

    # --- Honest split: grouped by content_key. No post variant spans the split. ---
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    h_tr, h_te = next(gss.split(idx, y, groups))
    yh_te, ph = _score_split(frame, h_tr, h_te)
    honest_acc = accuracy_score(yh_te, ph)
    honest_f1 = f1_score(yh_te, ph, zero_division=0)
    cm = confusion_matrix(yh_te, ph, labels=[0, 1])

    # Companion model, honest split, interpretable features only -> a clean 'why' readout.
    tree = DecisionTreeClassifier(max_depth=4, random_state=seed)
    tree.fit(frame[CLASSIFICATION_FEATURES].to_numpy(float)[h_tr], y[h_tr])
    coefs = dict(zip(CLASSIFICATION_FEATURES, tree.feature_importances_.round(3)))

    results = MatchingResults(
        naive_score=round(float(naive_acc), 4),
        honest_score=round(float(honest_acc), 4),
        naive_f1=round(float(naive_f1), 4),
        honest_f1=round(float(honest_f1), 4),
        confusion=cm,
        coefficients=coefs,
    )
    if verbose:
        _report(results, cfg)
    return results


def _report(r: MatchingResults, cfg: dict) -> None:
    tn, fp, fn, tp = r.confusion.ravel()
    print("=" * 60)
    print("OPPORTUNITY CLASSIFICATION + LEAKAGE EXPERIMENT")
    print("=" * 60)
    print(f"Naive split  accuracy : {r.naive_score:.3f}   (looks great — it's cheating)")
    print(f"Honest split accuracy : {r.honest_score:.3f}   (the real number)")
    print(f"THE GAP              : {r.gap:.3f}   <- this is the lie leakage tells you")
    print(f"    (min gap this demo guarantees: {cfg['matching']['min_leakage_gap']})")
    print(f"\nF1 — naive {r.naive_f1:.3f}  vs  honest {r.honest_f1:.3f}")
    print("\nConfusion matrix (honest split), rows=actual, cols=predicted:")
    print(f"                 pred: not-opp   pred: opp")
    print(f"    actual not-opp   {tn:>6}      {fp:>6}   (FP = wasted application)")
    print(f"    actual opp       {fn:>6}      {tp:>6}   (FN = missed real job)")
    print("\nPost-evidence feature importances (share of the tree's decisions):")
    for name, c in r.coefficients.items():
        print(f"    {name:<16}: {c:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
