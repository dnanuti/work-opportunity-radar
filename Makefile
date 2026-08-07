# Work Opportunity Radar — one command per stage, and `make all` for everything.
# Uses a local virtualenv in .venv so the demo never touches system packages.

PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: all setup data collect quality clean-data train assistant radar test demo notebooks clean

all: setup
	$(PY) run.py demo
	$(PY) run.py test
	@echo ""
	@echo "==> make all complete. See README.md for what each stage showed."

setup: .venv/.installed

.venv/.installed: requirements.txt
	python3 -m venv .venv
	$(PIP) install --upgrade pip >/dev/null
	$(PIP) install -r requirements.txt
	@touch .venv/.installed

# Stage 1 — generate the messy synthetic dataset (deterministic).
data: setup
	$(PY) -m src.generate_data

# Stage 0 — collect real-world posts (offline sample by default; live sources opt-in).
collect: setup
	$(PY) -m src.collect

# Stage 0b — score the quality of whatever data we have (collected, else synthetic).
quality: setup
	$(PY) -m src.quality

# Stage 1b — clean it (dedup, missing values, category normalisation).
clean-data: data
	$(PY) -m src.cleaning

# Stage 4 — the Radar: rank real collected opportunities by fit to the profile.
radar: collect
	$(PY) -m src.radar

# Stage 2 — train the matcher and run the leakage experiment.
train: clean-data
	$(PY) -m src.matching

# Stage 3 — GenAI application assistant (vague vs grounded), offline by default.
assistant: clean-data
	$(PY) -m src.application_assistant

# Full test suite — includes the leakage assertion and schema validation.
test: setup
	$(PY) -m pytest -q

# Convenience: launch Jupyter on the notebooks.
notebooks: setup
	$(PY) -m jupyter notebook notebooks/

# 'demo' uses the same talk-aligned cross-platform narrative as run.py.
demo: setup
	$(PY) run.py demo

clean:
	rm -rf .venv data/raw/*.csv data/processed/*.csv .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
