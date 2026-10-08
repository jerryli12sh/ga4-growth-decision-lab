"""Reproduce the public GA4 analysis from the included compressed CSV exports."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["all", "data", "analysis", "validation"],
        default="all",
        help="analysis and validation rebuild prerequisites; all also builds the report",
    )
    args = parser.parse_args()
    env = dict(
        os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg"
    )
    start = time.monotonic()

    def run(script, *options):
        print(f"Running {script}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "src" / f"{script}.py"), *map(str, options)],
            cwd=ROOT,
            env=env,
            check=True,
        )

    run("data")
    if args.stage == "data":
        return
    run("checks")
    run("measurement")
    run("transactions")
    run(
        "baseline",
        "--sessions",
        "data/processed/session_facts.csv",
        "--purchases",
        "data/processed/transaction_events.csv",
        "--quality-flags",
        "results/analysis/transaction_event_quality.csv",
        "--expected-sessions",
        "360129",
        "--output",
        "data/processed/validation",
        "--reports",
        "results/validation",
    )
    run("sensitivity")
    run("overview")
    run("journeys")
    run("merchandise")
    if args.stage in ["all", "validation"]:
        run(
            "validation",
            "--baseline",
            "data/processed/validation/first_checkout_positive_merchandise.csv",
            "--output",
            "results/validation",
            "--effect",
            "0.05",
            "--seed",
            "20261006",
            "--aa-reps",
            "500",
            "--rct-reps",
            "500",
            "--causal-reps",
            "300",
            "--sim-n",
            "6000",
        )
        run(
            "business_threshold",
            "--baseline",
            "data/processed/validation/first_checkout_positive_merchandise.csv",
            "--output",
            "results/validation/business_threshold.json",
        )
    if args.stage == "all":
        run("report")
    run("organize_results")
    print(f"Completed in {time.monotonic()-start:.1f} seconds")


if __name__ == "__main__":
    main()
