#!/usr/bin/env bash
# One-command launcher for the Work Opportunity Radar.
#   ./run.sh            -> set up, run every stage, print the story end to end
#   ./run.sh notebooks  -> set up, then open the Jupyter notebooks
#   ./run.sh radar      -> just collect real jobs and print your ranked opportunities
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-all}" in
  notebooks) make notebooks ;;
  radar)     make collect radar ;;
  all)       make setup
             .venv/bin/python run.py demo
             .venv/bin/python run.py test
             echo ""
             echo "==> Done. Open notebooks/ for the narrated version:  ./run.sh notebooks" ;;
  *)         echo "usage: ./run.sh [all|notebooks|radar]"; exit 2 ;;
esac
