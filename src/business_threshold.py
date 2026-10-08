"""Compare detecting a benefit with showing that benefit exceeds a commercial floor."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from cohort import read_table
from scipy.stats import norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--true-effect", type=float, default=0.05)
    ap.add_argument("--floor", type=float, default=0.025)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()
    b = read_table(args.baseline)
    p0 = float(b.y7.mean())
    p1 = p0 + args.true_effect
    if not 0 <= args.floor < args.true_effect < 1 - p0:
        raise ValueError("Require 0 <= floor < true effect and valid probabilities")
    n_arm = int(
        np.ceil(
            (norm.ppf(0.975) + norm.ppf(0.8)) ** 2
            * (p0 * (1 - p0) + p1 * (1 - p1))
            / (args.true_effect - args.floor) ** 2
        )
    )
    rng = np.random.default_rng(args.seed)
    reps = 10000
    c = rng.binomial(n_arm, p0, reps) / n_arm
    t = rng.binomial(n_arm, p1, reps) / n_arm
    lower = t - c - 1.96 * np.sqrt(t * (1 - t) / n_arm + c * (1 - c) / n_arm)
    probability = float((lower > args.floor).mean())
    result = dict(
        evidence="synthetic Bernoulli precision/design scenario, not observed treatment effect",
        baseline_path=args.baseline,
        baseline_rate=p0,
        assumed_true_effect=args.true_effect,
        illustrative_commercial_floor=args.floor,
        target_power=0.8,
        criterion="two-sided 95% CI lower endpoint greater than commercial floor",
        approximate_n_per_arm=n_arm,
        total_n=2 * n_arm,
        simulation_repetitions=reps,
        seed=args.seed,
        simulated_probability_meeting_criterion=probability,
        mcse=float(np.sqrt(probability * (1 - probability) / reps)),
        commercial_note="The 2.5pp floor is illustrative and has not been justified by actual implementation cost or margin data.",
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
