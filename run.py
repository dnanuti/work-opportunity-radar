"""Cross-platform entry point for the Work Opportunity Radar.

Examples:
  python run.py demo
  python run.py notebook
  python run.py configure
  python run.py chatgpt
  python run.py prompt
  python run.py validate output/chatgpt_response.json
  python run.py test

On Windows, ``py`` can be used instead of ``python``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _demo() -> None:
    from src import load_config
    from src.application_assistant import run_demo
    from src.cleaning import clean, clean_collected
    from src.collect import fetch
    from src.generate_data import generate
    from src.evidence import build_evidence_card, format_evidence_card
    from src.matching import run_experiment
    from src.quality import _load_default_frame, print_scorecard, score_dataframe
    from src.radar import run as radar

    cfg = load_config()
    print("\n1/6  Collect and preserve the raw source evidence")
    collected = fetch(cfg)
    print("\n2/6  Measure quality and create the trusted layer")
    frame = _load_default_frame(cfg)
    print_scorecard(score_dataframe(frame, cfg))
    clean_collected(cfg, collected)
    print("\n3/6  Classify first, then rank for the active candidate")
    ranked = radar(cfg)
    print("\n4/6  Train/evaluate on labelled examples - including the leakage trap")
    generate(cfg)
    clean(cfg)
    run_experiment(cfg)
    print("\n5/6  Compare vague and grounded GenAI on the top-ranked post")
    assistant = run_demo(cfg, job_record=ranked.iloc[0].to_dict())
    print("\n6/6  Review the evidence card, then record human feedback")
    print(format_evidence_card(build_evidence_card(
        ranked.iloc[0].to_dict(), assistant["grounded"]
    )))
    print("Run 'python run.py feedback JOB_ID relevant --reason "
          "\"why it helped\"' after reviewing a result.")


def _notebook() -> None:
    subprocess.run(
        [sys.executable, "-m", "notebook", "notebooks/"],
        cwd=ROOT,
        check=True,
    )


def _tests() -> None:
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Work Opportunity Radar on Windows, macOS, or Linux."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="run the complete offline teaching demo")
    sub.add_parser("configure", help="create a local candidate profile interactively")
    sub.add_parser("radar", help="collect and rank jobs for the configured profile")
    sub.add_parser("notebook", help="open the two guided notebooks")
    sub.add_parser("prompt", help="create a prompt to paste into ChatGPT")
    sub.add_parser("chatgpt", help="guided ChatGPT copy/paste and validation flow")
    validate = sub.add_parser("validate", help="validate JSON copied back from ChatGPT")
    validate.add_argument("file", help="path to the JSON response")
    feedback = sub.add_parser("feedback", help="record reviewable candidate feedback")
    feedback.add_argument("job_id")
    feedback.add_argument("judgement", choices=["relevant", "not_relevant", "unsure"])
    feedback.add_argument("--reason", default="")
    sub.add_parser("test", help="run the automated tests")
    args = parser.parse_args()

    if args.command == "demo":
        _demo()
    elif args.command == "notebook":
        _notebook()
    elif args.command == "test":
        _tests()
    elif args.command == "configure":
        from src.profile import configure

        configure()
    elif args.command == "radar":
        from src.collect import fetch
        from src.radar import run

        fetch()
        run()
    elif args.command == "prompt":
        from src.manual_chatgpt import write_prompt

        write_prompt()
    elif args.command == "chatgpt":
        from src.manual_chatgpt import interactive_chatgpt

        interactive_chatgpt()
    elif args.command == "validate":
        from src.manual_chatgpt import validate_response_file

        validate_response_file(args.file)
    elif args.command == "feedback":
        from src.feedback import record_feedback

        value = {"relevant": True, "not_relevant": False, "unsure": None}[args.judgement]
        print(record_feedback(args.job_id, value, args.reason))


if __name__ == "__main__":
    main()
