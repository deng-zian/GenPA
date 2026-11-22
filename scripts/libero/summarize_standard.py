#!/usr/bin/env python3
"""Report standard LIBERO only after validating all 40 tasks and 2,000 episodes."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "libero"))
from summarize_results import summarize_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    # Ignore any narrower tasks.txt: this entry point always validates the full benchmark.
    from summarize_results import full_task_set, collect_results
    collect_results(args.results, benchmark="libero", expected=full_task_set("libero"),
                    num_trials=50, allow_partial=args.allow_partial)
    summarize_results(args.results, benchmark="libero", num_trials=50,
                      allow_partial=args.allow_partial, auto_task_file=False)
