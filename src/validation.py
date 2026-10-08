"""Actual offline replay + explicitly semi-synthetic simulations on real covariates.

Run only after prepare_baseline.py produced a reviewed, mature one-user fact table.
No output from this script is an observed product treatment effect.
"""

from __future__ import annotations
import argparse, json, platform, time, importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from cohort import read_table
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm
from methods import (
    difference,
    srm,
    weighted_cluster_difference,
    fit_logistic,
    crossfit_nuisance,
    causal_estimates,
    overlap_diagnostics,
    summarize_runs,
    Estimate,
)


def raw_features(b):
    return np.column_stack(
        [
            np.log1p(b.pre_sessions_14d),
            b.device_category.eq("mobile").astype(float),
            np.log1p(b.pre_purchases_14d),
            b.pre_history_present.astype(float),
            b.ga_session_number.fillna(1).gt(1).astype(float),
            b.first_user_medium.eq("organic").astype(float),
            b.ga_session_number.isna().astype(float),
        ]
    )


def features(b):
    raw = raw_features(b)
    scale = np.std(raw, axis=0)
    scale[scale < 1e-06] = 1
    return np.clip((raw - np.mean(raw, axis=0)) / scale, -3, 3)


def fixed_preperiod_score(b):
    """Fit on earlier users whose complete outcome windows end before eval starts."""
    order = np.argsort(b.index_ts.to_numpy())
    cutoff = int(b.index_ts.iloc[order[int(0.35 * len(b))]])
    train = b.index_ts.to_numpy() + 7 * 86400000000 < cutoff
    evaluate = b.index_ts.to_numpy() >= cutoff
    if train.sum() < 100 or evaluate.sum() < 100:
        raise ValueError("Insufficient disjoint mature train/evaluation cohort.")
    x = raw_features(b)
    mu = x[train].mean(axis=0)
    sd = x[train].std(axis=0)
    sd[sd < 1e-08] = 1
    x = (x - mu) / sd
    predict = fit_logistic(x[train], b.y7.to_numpy()[train])
    score = predict(x)
    return (
        score,
        evaluate,
        dict(
            training_users=int(train.sum()),
            evaluation_users=int(evaluate.sum()),
            evaluation_start_ts=cutoff,
            purged_or_excluded_users=int((~(train | evaluate)).sum()),
            preperiod_features=[
                "log1p(pre_sessions_14d)",
                "mobile",
                "log1p(pre_purchases_14d)",
                "pre_history_present",
                "session_number_gt1",
                "first_user_medium_organic",
                "session_number_missing",
            ],
        ),
    )


def run_aa(b, score, evaluate, rng, reps):
    y = b.y7.to_numpy(dtype=float)[evaluate]
    pred = score[evaluate]
    records = []
    for rep in range(reps):
        a = rng.binomial(1, 0.5, len(y))
        p_srm = srm(a)
        for method, response in [
            ("difference_in_means", y),
            ("fixed_preperiod_score", y - pred),
        ]:
            r = difference(response, a).as_dict(0.0)
            r.update(
                scenario="historical_aa_replay",
                method=method,
                rep=rep,
                n=len(y),
                srm_assignment=p_srm,
            )
            records.append(r)
    raw, summary = summarize_runs(records, ["scenario", "method"])
    return (raw, summary)


def calibrated_probabilities(score, target):
    logits = logit(np.clip(score, 0.0001, 0.9999))
    if not 0.0001 < target < 0.9:
        raise ValueError(
            "Simulation calibration requires baseline rate in (0.0001,0.90)"
        )
    shift = brentq(
        lambda v: np.clip(expit(logits + v), 0.0001, 0.9).mean() - target, -30, 30
    )
    return np.clip(expit(logits + shift), 0.0001, 0.9)


def run_rct(b, score, rng, reps, n, effect):
    p0 = calibrated_probabilities(score, float(b.y7.mean()))
    mobile = b.device_category.eq("mobile").to_numpy(dtype=float)
    hetero = 0.3 + 1.4 * mobile
    hetero = hetero / hetero.mean()
    all_records = []
    gates = []
    for scenario in [
        "zero_effect",
        "business_mde",
        "heterogeneous_effect",
        "lost_treated_converters",
        "immature_24h_window",
        "revenue_guardrail_harm",
    ]:
        delta = 0 if scenario in ["zero_effect", "lost_treated_converters"] else effect
        tau = delta * (
            hetero if scenario == "heterogeneous_effect" else np.ones(len(b))
        )
        p1 = np.minimum(p0 + tau, 0.995)
        truth = float(np.mean(p1 - p0))
        for rep in range(reps):
            sample = rng.integers(0, len(b), n)
            a = rng.binomial(1, 0.5, n)
            y = rng.binomial(1, np.where(a, p1[sample], p0[sample])).astype(float)
            observed = np.ones(n, dtype=bool)
            if scenario == "lost_treated_converters":
                observed = ~((a == 1) & (y == 1) & (rng.random(n) < 0.3))
            if scenario == "immature_24h_window":
                y = y * (rng.random(n) < np.where(a, 0.35, 0.7))
            p_srm = srm(a)
            p_observed = srm(a[observed])
            for method, response in [
                ("difference_in_means", y),
                ("fixed_preperiod_score", y - score[sample]),
            ]:
                r = difference(response[observed], a[observed]).as_dict(truth)
                r.update(
                    scenario=scenario,
                    method=method,
                    rep=rep,
                    n=n,
                    observed_n=int(observed.sum()),
                    srm_assignment=p_srm,
                    srm_observed=p_observed,
                )
                all_records.append(r)
            if scenario == "revenue_guardrail_harm":
                order_value = rng.lognormal(np.log(25) - 0.5 * 0.6**2, 0.6, n)
                revenue = y * order_value * np.where(a, 0.75, 1.0)
                guardrail = difference(revenue, a)
                primary = difference(y, a)
                margin = 0.05 * float(p0.mean()) * 25
                ship = (
                    primary.estimate - 1.96 * primary.se > 0.5 * effect
                    and guardrail.estimate - 1.645 * guardrail.se > -margin
                )
                gates.append(
                    dict(
                        rep=rep,
                        scenario=scenario,
                        synthetic_rpu_truth=float(
                            p1.mean() * 25 * 0.75 - p0.mean() * 25
                        ),
                        estimated_rpu_effect=guardrail.estimate,
                        rpu_ci_low=guardrail.estimate - 1.96 * guardrail.se,
                        rpu_ci_high=guardrail.estimate + 1.96 * guardrail.se,
                        ship=bool(ship),
                        main_positive_significant=bool(
                            primary.estimate - 1.96 * primary.se > 0
                        ),
                    )
                )
    for rep in range(reps):
        sample = rng.integers(0, len(b), n)
        a = rng.binomial(1, 0.5, n)
        y = rng.binomial(1, p0[sample])
        repeats = (
            1
            + np.minimum(b.pre_sessions_14d.to_numpy(dtype=int)[sample], 5)
            + rng.poisson(1.0, n)
        )
        naive, cluster = weighted_cluster_difference(y, a, repeats)
        for method, est in [
            ("correct_one_user_one_row", difference(y, a)),
            ("wrong_repeated_rows_iid", naive),
            ("user_cluster_ratio_se", cluster),
        ]:
            r = est.as_dict(0)
            r.update(scenario="repeated_sessions_null", method=method, rep=rep, n=n)
            all_records.append(r)
    raw, summary = summarize_runs(all_records, ["scenario", "method"])
    return (raw, summary, pd.DataFrame(gates))


def dgp(b, x, hidden_u, scenario, effect):
    nonlinear = x[:, 0] * x[:, 1] + 0.45 * x[:, 0] ** 2
    rich = np.column_stack([x, nonlinear])
    risk = (
        0.6 * x[:, 0] - 0.3 * x[:, 1] + 0.5 * x[:, 2] + 0.25 * x[:, 4] + 0.6 * nonlinear
    )
    if scenario == "unmeasured_intent":
        risk = risk + 0.9 * hidden_u
    base = float(np.clip(b.y7.mean(), 0.005, 0.8))
    intercept = brentq(lambda v: expit(v + risk).mean() - base, -40, 40)
    p0 = expit(intercept + risk)
    treatment_shift = brentq(
        lambda v: (expit(logit(p0) + v) - p0).mean() - effect, 0, 20
    )
    p1 = expit(logit(p0) + treatment_shift)
    propensity_logit = (
        -0.3 + 0.7 * x[:, 0] - 0.5 * x[:, 1] + 0.5 * x[:, 2] + 0.7 * nonlinear
    )
    if scenario == "randomized":
        e = np.full(len(x), 0.5)
    elif scenario == "unmeasured_intent":
        e = expit(propensity_logit + 1.25 * hidden_u)
    elif scenario in ["poor_overlap", "positivity_violation"]:
        e = expit(3 * propensity_logit)
    else:
        e = expit(propensity_logit)
    if scenario == "positivity_violation":
        e[propensity_logit <= np.quantile(propensity_logit, 0.15)] = 0
        e[propensity_logit >= np.quantile(propensity_logit, 0.85)] = 1
        observed_probability = e * p1 + (1 - e) * p0
        p0 = np.where(e == 1, np.maximum(0.0001, p0 - 0.1), p0)
        p1 = np.where(e == 0, np.minimum(0.9999, p1 + 0.1), p1)
        assert np.allclose(e * p1 + (1 - e) * p0, observed_probability)
    poor = x[:, [1, 5]]
    xprop = (
        poor if scenario in ["propensity_misspecified", "both_misspecified"] else rich
    )
    xout = poor if scenario in ["outcome_misspecified", "both_misspecified"] else rich
    return (p0, p1, e, xprop, xout, rich)


def run_causal(b, rng, reps, n, effect):
    x = features(b)
    hidden_u = rng.normal(size=len(b))
    rows = []
    diagnostics = []
    population = []
    for scenario in [
        "randomized",
        "observed_confounding",
        "propensity_misspecified",
        "outcome_misspecified",
        "both_misspecified",
        "unmeasured_intent",
        "poor_overlap",
        "positivity_violation",
    ]:
        p0, p1, e, xprop, xout, rich = dgp(b, x, hidden_u, scenario, effect)
        truth = float(np.mean(p1 - p0))
        population.append(
            dict(
                scenario=scenario,
                truth=truth,
                baseline_probability=float(p0.mean()),
                treated_probability=float(p1.mean()),
                true_propensity_below_005=float((e < 0.05).mean()),
                true_propensity_above_095=float((e > 0.95).mean()),
                true_positivity_failure_fraction=float(((e == 0) | (e == 1)).mean()),
            )
        )
        for rep in range(reps):
            sample = rng.integers(0, len(b), n)
            a = rng.binomial(1, e[sample])
            y = rng.binomial(1, np.where(a, p1[sample], p0[sample]))
            if min(a.sum(), n - a.sum()) < 20:
                raise RuntimeError(f"Insufficient arm observations: {scenario}")
            pred, m0, m1 = crossfit_nuisance(
                xprop[sample], xout[sample], a, y, int(rng.integers(0, 2**31 - 1))
            )
            for method, estimate in causal_estimates(
                a, y, pred, m0, m1, clip=0.01
            ).items():
                row = estimate.as_dict(truth)
                row.update(scenario=scenario, method=method, rep=rep, n=n, clip=0.01)
                rows.append(row)
            if scenario != "positivity_violation":
                estimate = causal_estimates(
                    a, y, e[sample], p0[sample], p1[sample], clip=1e-08
                )["aipw"]
                row = estimate.as_dict(truth)
                row.update(
                    scenario=scenario,
                    method="oracle_aipw_simulator_only",
                    rep=rep,
                    n=n,
                    clip=1e-08,
                )
                rows.append(row)
            if scenario in ["poor_overlap", "positivity_violation"]:
                for clip in [0.001, 0.05]:
                    estimate = causal_estimates(a, y, pred, m0, m1, clip=clip)["aipw"]
                    row = estimate.as_dict(truth)
                    row.update(
                        scenario=scenario,
                        method=f"aipw_clip_{clip}",
                        rep=rep,
                        n=n,
                        clip=clip,
                    )
                    rows.append(row)
            diagnostic = overlap_diagnostics(a, pred, rich[sample], clip=0.01)
            diagnostic.update(scenario=scenario, rep=rep)
            diagnostics.append(diagnostic)
        print(f"Finished causal scenario {scenario}: {reps} repetitions", flush=True)
    raw, summary = summarize_runs(rows, ["scenario", "method"])
    return (raw, summary, pd.DataFrame(diagnostics), pd.DataFrame(population))


def power_table(b, effect, extra_absolute_effects=()):
    p0 = float(b.y7.mean())
    dates = pd.to_datetime(b.index_date)
    calendar_days = max(1, (dates.max() - dates.min()).days + 1)
    daily = len(b) / calendar_days
    rows = []
    week = dates.dt.to_period("W-SUN")
    full_week_counts = b.groupby(week).size()
    full_week_counts = full_week_counts[
        [
            w.start_time >= dates.min() and w.end_time.normalize() <= dates.max()
            for w in full_week_counts.index
        ]
    ]
    p25_daily = (
        float(full_week_counts.quantile(0.25) / 7) if len(full_week_counts) else daily
    )
    relatives = [0.05, 0.1, 0.15, 0.2, 0.3]
    if not any((np.isclose(effect / p0, value) for value in relatives)):
        relatives.append(effect / p0)
    for absolute in extra_absolute_effects:
        if not any((np.isclose(absolute / p0, value) for value in relatives)):
            relatives.append(absolute / p0)
    for relative in sorted(relatives):
        delta = p0 * relative
        p1 = p0 + delta
        if p1 >= 1:
            continue
        for power in [0.8, 0.9]:
            pbar = (p0 + p1) / 2
            n_arm = int(
                np.ceil(
                    (
                        norm.ppf(0.975) * np.sqrt(2 * pbar * (1 - pbar))
                        + norm.ppf(power) * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
                    )
                    ** 2
                    / delta**2
                )
            )
            enroll_weeks = int(np.ceil(2 * n_arm / daily / 7))
            rows.append(
                dict(
                    baseline_p7=p0,
                    relative_mde=relative,
                    absolute_mde_pp=100 * delta,
                    power=power,
                    n_per_arm=n_arm,
                    total_n=2 * n_arm,
                    historical_new_eligible_users_per_day=daily,
                    enrollment_weeks=enroll_weeks,
                    total_weeks_with_7d_followup=enroll_weeks + 1,
                    complete_week_p25_users_per_day=p25_daily,
                    total_weeks_at_p25_flow=int(np.ceil(2 * n_arm / p25_daily / 7)) + 1,
                    selected_simulation_mde=bool(np.isclose(delta, effect)),
                )
            )
    return pd.DataFrame(rows)


def validate_sample_size(table, rng, repetitions=10000):
    """Binomial RCT power validation at the exact tabulated design sample sizes."""
    records = []
    for row in table.itertuples(index=False):
        p0 = row.baseline_p7
        delta = row.absolute_mde_pp / 100
        p1 = p0 + delta
        n = row.n_per_arm
        successes0 = rng.binomial(n, p0, repetitions)
        successes1 = rng.binomial(n, p1, repetitions)
        phat0 = successes0 / n
        phat1 = successes1 / n
        estimate = phat1 - phat0
        se = np.sqrt(phat0 * (1 - phat0) / n + phat1 * (1 - phat1) / n)
        reject = np.abs(estimate) > 1.96 * se
        coverage = np.abs(estimate - delta) <= 1.96 * se
        power = float(reject.mean())
        records.append(
            dict(
                relative_mde=row.relative_mde,
                target_power=row.power,
                n_per_arm=n,
                repetitions=repetitions,
                simulated_power=power,
                power_mcse=float(np.sqrt(power * (1 - power) / repetitions)),
                coverage=float(coverage.mean()),
                selected_simulation_mde=row.selected_simulation_mde,
                evidence_label="synthetic independent Bernoulli RCT at observed baseline rate",
            )
        )
    return pd.DataFrame(records)


def save(frame, path):
    frame.to_csv(path, index=False)


def make_figures(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    causal = pd.read_csv(out / "semi_synthetic_causal_summary.csv")
    chosen = causal[causal.method.isin(["naive", "aipw", "oracle_aipw_simulator_only"])]
    order = [
        "randomized",
        "observed_confounding",
        "propensity_misspecified",
        "outcome_misspecified",
        "both_misspecified",
        "unmeasured_intent",
        "poor_overlap",
        "positivity_violation",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    for offset, method, color in [
        (-0.16, "naive", "#cb5b5b"),
        (0, "aipw", "#2b7bba"),
        (0.16, "oracle_aipw_simulator_only", "#42865a"),
    ]:
        g = chosen[chosen.method.eq(method)].set_index("scenario").reindex(order)
        axes[0].errorbar(
            100 * g.bias,
            np.arange(len(order)) + offset,
            xerr=196 * g.empirical_sd / np.sqrt(g.repetitions),
            fmt="o",
            markersize=4,
            label=method,
            color=color,
            capsize=2,
        )
        coverage_error = np.vstack(
            [
                g.covered_rate - g.covered_mc_wilson_low,
                g.covered_mc_wilson_high - g.covered_rate,
            ]
        )
        axes[1].errorbar(
            g.covered_rate,
            np.arange(len(order)) + offset,
            xerr=np.maximum(coverage_error, 0),
            fmt="o",
            markersize=4,
            label=method,
            color=color,
            capsize=2,
        )
    axes[0].axvline(0, color="grey", lw=1)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_xlabel("Bias (percentage points)")
    axes[1].axvline(0.95, color="grey", ls="--")
    axes[1].set_yticks(range(len(order)), [])
    axes[1].set_xlabel("95% interval coverage")
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    axes[1].set_xlim(-0.025, 1.03)
    axes[1].legend(loc="lower right", fontsize=7)
    fig.suptitle(
        "SEMI-SYNTHETIC: adjustment cannot repair hidden intent or missing overlap",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "semi_synthetic_causal_bias_coverage.png", dpi=180)
    plt.close(fig)
    rct = pd.read_csv(out / "semi_synthetic_rct_summary.csv")
    chosen = rct[rct.method.eq("difference_in_means")]
    fig, ax = plt.subplots(figsize=(10, 5))
    ys = np.arange(len(chosen))
    ax.barh(ys, chosen.reject_zero_rate, color="#397c91")
    ax.set_yticks(ys, chosen.scenario)
    ax.set_xlim(0, 1)
    ax.axvline(0.05, ls="--", color="grey")
    ax.set_xlabel("Probability of rejecting zero effect (not all are valid successes)")
    ax.set_title("SEMI-SYNTHETIC: power and measurement failure scenarios")
    fig.tight_layout()
    fig.savefig(out / "semi_synthetic_rct_detection.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--aa-reps", type=int, default=500)
    ap.add_argument("--rct-reps", type=int, default=500)
    ap.add_argument("--causal-reps", type=int, default=150)
    ap.add_argument("--sim-n", type=int, default=6000)
    ap.add_argument("--effect", type=float)
    args = ap.parse_args()
    start = time.time()
    b = read_table(args.baseline).sort_values("user_key").reset_index(drop=True)
    if not b.user_key.is_unique:
        raise ValueError("Baseline must contain one row per user")
    if len(b) < 300:
        raise ValueError(
            "Insufficient real user-level baseline; aggregates cannot substitute for users"
        )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(args.seed)
    aa_seed, rct_seed, causal_seed, power_seed = seed_sequence.spawn(4)
    score, evaluate, training = fixed_preperiod_score(b)
    effect = (
        args.effect
        if args.effect is not None
        else float(np.clip(0.15 * b.y7.mean(), 0.003, 0.05))
    )
    config = dict(
        evidence_label="historical A/A replay plus semi-synthetic method validation; no observed intervention effect",
        baseline_path=args.baseline,
        baseline_n=len(b),
        baseline_purchase_rate=float(b.y7.mean()),
        seed=args.seed,
        aa_repetitions=args.aa_reps,
        rct_repetitions=args.rct_reps,
        causal_repetitions=args.causal_reps,
        simulation_n=args.sim_n,
        synthetic_probability_difference=effect,
        training=training,
        python=platform.python_version(),
        package_versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]
        },
        ci_notes="OR and IPW are point estimates only. AIPW uses cross-fitted influence-score normal CIs. Coverage is evaluated, not guaranteed.",
        inference_notes="Synthetic repetitions resample observed baseline covariates with replacement and generate new independent outcomes. Population truth is the fixed empirical baseline mixture, including the simulator-only latent U.",
        aa_notes="A/A repeatedly randomizes unique existing evaluation users. It does not create independent real trials or validate a production assignment system.",
    )
    (out / "validation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )
    raw, summary = run_aa(
        b, score, evaluate, np.random.default_rng(aa_seed), args.aa_reps
    )
    save(raw, out / "historical_aa_replay_runs.csv")
    save(summary, out / "historical_aa_replay_summary.csv")
    raw, summary, gates = run_rct(
        b, score, np.random.default_rng(rct_seed), args.rct_reps, args.sim_n, effect
    )
    save(raw, out / "semi_synthetic_rct_runs.csv")
    save(summary, out / "semi_synthetic_rct_summary.csv")
    save(gates, out / "semi_synthetic_guardrail_decisions.csv")
    raw, summary, diagnostics, population = run_causal(
        b, np.random.default_rng(causal_seed), args.causal_reps, args.sim_n, effect
    )
    save(raw, out / "semi_synthetic_causal_runs.csv")
    save(summary, out / "semi_synthetic_causal_summary.csv")
    save(diagnostics, out / "semi_synthetic_overlap_diagnostics.csv")
    save(population, out / "semi_synthetic_population_truth.csv")
    design_table = power_table(b, effect, extra_absolute_effects=(0.025, 0.075))
    save(design_table, out / "experiment_sample_size_scenarios.csv")
    save(
        validate_sample_size(design_table, np.random.default_rng(power_seed)),
        out / "synthetic_sample_size_validation.csv",
    )
    make_figures(out)
    config.update(elapsed_seconds=time.time() - start, status="completed")
    (out / "validation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2)
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
