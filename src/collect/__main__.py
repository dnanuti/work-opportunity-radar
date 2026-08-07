"""Allow `python -m src.collect` to run the collector (used by the Makefile)."""
from src.collect import fetch

if __name__ == "__main__":
    fetch()
