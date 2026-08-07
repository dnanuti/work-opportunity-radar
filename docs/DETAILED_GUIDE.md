# Mentor guide

Start with the root `README.md`, then use `notebooks/01_data_foundations.ipynb` and
`notebooks/02_work_opportunity_radar.ipynb`.

## Architecture

```text
provider JSON -> raw evidence -> structured records -> trusted records
                                                 |-> opportunity classification
                                                 |-> candidate-aware ranking
                                                 |-> grounded GenAI extraction
                                                 |-> evidence card -> human feedback
```

The boundaries are intentional:

- `src/collect/` preserves and normalises permitted source data.
- `src/quality.py` measures completeness, validity, uniqueness, consistency, and
  freshness before a model is involved.
- `src/cleaning.py` creates a separate trusted layer and exposes missingness.
- `src/classification.py` decides whether a post looks current and opportunity-like; it
  never reads a candidate profile.
- `src/radar.py` ranks eligible posts for the active candidate with inspectable weights.
- `src/matching.py` is a labelled ML/evaluation exercise, not the production radar.
- `src/application_assistant.py` contrasts vague generation with schema-validated,
  grounded extraction.
- `src/evidence.py` renders the candidate-facing evidence card.
- `src/feedback.py` records append-only feedback with `needs_review` status.

## Why two data paths exist

Collected feeds are useful for collection, quality, cleaning, classification, and
ranking, but they do not contain defensible ground-truth labels. The ML exercise
therefore generates a labelled synthetic dataset containing duplicates, missing values,
inconsistent categories, and label noise. That keeps the evaluation lesson honest.

## Extension review checklist

- Is every source permitted, attributable, and resilient to failure?
- Is the original payload retained separately from trusted transformations?
- Does classification remain independent of candidate identity and preferences?
- Can every ranking contribution be explained in plain language?
- Are uncertainty and missing evidence visible?
- Are false positives and false negatives evaluated separately?
- Is human feedback reviewed before it becomes a label?
- Do tests run offline and deterministically on Windows, macOS, and Linux?
