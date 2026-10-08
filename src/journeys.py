"""Real session journey, checkout follow-up, and device standardization.

Session diagnostics and one-index-per-user experimental baseline are kept as
separate populations. No observed behavior is treated as randomized exposure.
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cohort import read_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/analysis"
OUT.mkdir(parents=True, exist_ok=True)
SEED, REPS = (20261007, 600)
SOURCE = ROOT / "data/processed/session_facts.csv"
TX = ROOT / "data/processed/transaction_events.csv"
BASE = ROOT / "data/processed/validation/first_checkout_positive_merchandise.csv"
s = pd.read_csv(SOURCE, dtype={"user_pseudo_id": "string", "ga_session_id": "string"})
t = pd.read_csv(TX, dtype={"user_pseudo_id": "string", "ga_session_id": "string"})
flags = pd.read_csv(OUT / "transaction_event_quality.csv")
t = t.merge(
    flags[["event_id", "positive_merchandise_event", "candidate_repeat_excess"]],
    on="event_id",
    validate="one_to_one",
)
assert not s.duplicated(["user_pseudo_id", "ga_session_id"]).any()
assert s.purchase_events.sum() == len(t)
s["date"] = pd.to_datetime(s.session_date)
s["week"] = (s.date - pd.to_timedelta(s.date.dt.dayofweek, unit="D")).dt.strftime(
    "%Y-%m-%d"
)
s["prior_observed_session"] = s.session_start_ts.gt(
    s.groupby("user_pseudo_id").session_start_ts.transform("min")
)
for col, source_col in [
    ("has_view", "view_events"),
    ("has_checkout", "checkout_events"),
    ("has_search", "search_events"),
    ("has_cart", "cart_events"),
    ("has_shipping", "shipping_events"),
    ("has_payment", "payment_events"),
    ("has_raw_purchase", "purchase_events"),
]:
    s[col] = s[source_col].gt(0)
s["has_core_checkout"] = s.checkout_after_view_ts.notna()
s["has_payment_after_checkout"] = s.payment_after_checkout_ts.notna()
s["view_mean_item_rows"] = s.viewed_item_rows.div(s.view_events.replace(0, np.nan))
key = ["user_pseudo_id", "ga_session_id"]
p = t[t.positive_merchandise_event].merge(
    s[
        key
        + [
            "first_view_ts",
            "first_checkout_ts",
            "checkout_after_view_ts",
            "payment_after_checkout_ts",
            "checkout_after_view_cart_ts",
        ]
    ],
    on=key,
    validate="many_to_one",
)
for out, threshold in [
    ("positive_any_ts", None),
    ("positive_after_view_ts", "first_view_ts"),
    ("positive_after_checkout_ts", "first_checkout_ts"),
    ("positive_after_core_checkout_ts", "checkout_after_view_ts"),
    ("positive_after_payment_ts", "payment_after_checkout_ts"),
    ("positive_after_view_cart_checkout_ts", "checkout_after_view_cart_ts"),
]:
    use = p if threshold is None else p[p.event_timestamp > p[threshold]]
    val = use.groupby(key).event_timestamp.min().rename(out)
    s = s.merge(val, left_on=key, right_index=True, how="left", validate="one_to_one")
    s["has_" + out.removesuffix("_ts")] = s[out].notna()
assert (
    s.has_positive_any.sum()
    == t[t.positive_merchandise_event][key].drop_duplicates().shape[0]
)
assert (s.has_positive_after_core_checkout <= s.has_core_checkout).all()
assert (s.has_positive_after_payment <= s.has_payment_after_checkout).all()
scopes = {
    "full_92d": s,
    "stable_57d": s[s.session_date.between("2020-11-30", "2021-01-25")],
}
count_cols = [
    "has_view",
    "has_checkout",
    "has_core_checkout",
    "has_search",
    "has_cart",
    "has_shipping",
    "has_payment",
    "has_payment_after_checkout",
    "has_raw_purchase",
    "has_positive_any",
    "has_positive_after_view",
    "has_positive_after_checkout",
    "has_positive_after_core_checkout",
    "has_positive_after_payment",
]


def session_metrics(x):
    out = {
        "sessions": len(x),
        "users": x.user_pseudo_id.nunique(),
        **{col: int(x[col].sum()) for col in count_cols},
    }
    for name, num, den in [
        ("view_to_core_checkout", "has_core_checkout", "has_view"),
        ("view_to_positive_after_view", "has_positive_after_view", "has_view"),
        ("view_to_core_positive", "has_positive_after_core_checkout", "has_view"),
        ("checkout_to_positive", "has_positive_after_checkout", "has_checkout"),
        ("checkout_to_payment", "has_payment_after_checkout", "has_checkout"),
        (
            "payment_to_positive",
            "has_positive_after_payment",
            "has_payment_after_checkout",
        ),
    ]:
        out[name] = out[num] / out[den] if out[den] else None
    out["all_session_positive_rate"] = (
        out["has_positive_any"] / len(x) if len(x) else None
    )
    return out


group_rows, timing_rows, summary = ([], [], {})
for scope, x in scopes.items():
    summary[scope] = session_metrics(x)
    for dim in [
        "device_category",
        "country",
        "first_user_source",
        "week",
        "prior_observed_session",
        "has_search",
    ]:
        for val, group in x.groupby(dim, dropna=False):
            group_rows.append(
                {
                    "scope": scope,
                    "dimension": dim,
                    "value": str(val),
                    **session_metrics(group),
                }
            )
    for (dev, week), group in x.groupby(["device_category", "week"]):
        group_rows.append(
            {
                "scope": scope,
                "dimension": "device_week",
                "value": dev + "|" + week,
                **session_metrics(group),
            }
        )
    co = x[x.has_checkout]
    shipping_delta = (co.first_shipping_ts - co.first_checkout_ts) / 1000000.0
    payment_delta = (co.payment_after_checkout_ts - co.first_checkout_ts) / 1000000.0
    purchase_delta = (co.positive_after_checkout_ts - co.first_checkout_ts) / 1000000.0
    for stage, series in [
        ("first_shipping_minus_checkout", shipping_delta),
        ("first_payment_after_checkout", payment_delta),
        ("first_positive_after_checkout", purchase_delta),
    ]:
        vals = series.dropna()
        timing_rows.append(
            {
                "scope": scope,
                "stage": stage,
                "observed_sessions": len(vals),
                "before_checkout": int(vals.lt(0).sum()),
                "within_10ms": int(vals.abs().le(0.01).sum()),
                "within_1s": int(vals.abs().le(1).sum()),
                **{
                    f"q{q:g}_seconds": vals.quantile(q)
                    for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
                },
            }
        )
    summary[scope]["checkout_without_payment_sessions"] = int(
        (co.has_checkout & ~co.has_payment_after_checkout).sum()
    )
    summary[scope]["positive_without_payment_sessions"] = int(
        (co.has_positive_after_checkout & ~co.has_positive_after_payment).sum()
    )
    summary[scope]["shipping_coverage_sessions"] = int(
        co.first_shipping_ts.notna().sum()
    )
    summary[scope]["shipping_within_one_second_sessions"] = int(
        shipping_delta.abs().le(1).sum()
    )
    summary[scope]["shipping_first_before_checkout_sessions"] = int(
        shipping_delta.lt(0).sum()
    )
pd.DataFrame(group_rows).to_csv(OUT / "session_segment_metrics.csv", index=False)
pd.DataFrame(timing_rows).to_csv(OUT / "session_checkout_timing.csv", index=False)
std_rows, std_cells = ([], [])
country_top = s.country.value_counts().head(4).index.tolist()
s["country_group"] = s.country.where(s.country.isin(country_top), "Other country")
s["country_us"] = s.country.eq("United States").map({True: "US", False: "non-US"})
s["source_group"] = s.first_user_source.fillna("Missing")
scopes = {
    "full_92d": s,
    "stable_57d": s[s.session_date.between("2020-11-30", "2021-01-25")],
}


def device_bootstrap(x, outcome, strata, name, scope, eligible_name):
    x = x[x.device_category.isin(["mobile", "desktop"])].copy()
    x["stratum"] = "all" if not strata else x[strata].astype(str).agg("|".join, axis=1)
    counts = (
        x.groupby(["stratum", "device_category"])
        .agg(
            n=(outcome, "size"), y=(outcome, "sum"), users=("user_pseudo_id", "nunique")
        )
        .reset_index()
    )
    pivot = counts.pivot(
        index="stratum", columns="device_category", values=["n", "users"]
    ).fillna(0)
    supported = pivot.index[
        (pivot["n"].min(axis=1) >= 10) & (pivot["users"].min(axis=1) >= 8)
    ]
    common = x[x.stratum.isin(supported)].copy()
    if common.empty:
        return
    counts = counts[counts.stratum.isin(supported)].copy()
    for r in counts.to_dict("records"):
        std_cells.append(
            {"scope": scope, "eligible": eligible_name, "specification": name, **r}
        )
    cells = pd.Categorical(common.stratum, categories=supported).codes.astype(np.int64)
    dev = common.device_category.eq("mobile").to_numpy().astype(int)
    group_code = cells * 2 + dev
    n_cells = len(supported)
    uid, users = pd.factorize(common.user_pseudo_id, sort=False)
    y = common[outcome].to_numpy().astype(float)
    base_n = np.bincount(group_code, minlength=n_cells * 2).reshape(-1, 2)
    base_y = np.bincount(group_code, weights=y, minlength=n_cells * 2).reshape(-1, 2)
    weights = base_n.sum(axis=1) / base_n.sum()
    rates = base_y / base_n
    standardized = (weights[:, None] * rates).sum(axis=0)
    crude = base_y.sum(axis=0) / base_n.sum(axis=0)
    all_mobile = x[x.device_category.eq("mobile")][outcome].mean()
    all_desktop = x[x.device_category.eq("desktop")][outcome].mean()
    rng = np.random.default_rng(SEED)
    estimates, crude_estimates = ([], [])
    for _ in range(REPS):
        user_weight = np.bincount(
            rng.integers(0, len(users), len(users)), minlength=len(users)
        )
        w = user_weight[uid]
        n = np.bincount(group_code, weights=w, minlength=n_cells * 2).reshape(-1, 2)
        yy = np.bincount(group_code, weights=w * y, minlength=n_cells * 2).reshape(
            -1, 2
        )
        if (n == 0).any():
            continue
        pp = (weights[:, None] * (yy / n)).sum(axis=0)
        estimates.append(pp[1] - pp[0])
        cc = yy.sum(axis=0) / n.sum(axis=0)
        crude_estimates.append(cc[1] - cc[0])
    row = {
        "scope": scope,
        "eligible": eligible_name,
        "outcome": outcome,
        "specification": name,
        "all_eligible_desktop_mobile_sessions": len(x),
        "common_support_sessions": len(common),
        "common_support_share": len(common) / len(x),
        "supported_strata": n_cells,
        "desktop_n_common": int(base_n[:, 0].sum()),
        "mobile_n_common": int(base_n[:, 1].sum()),
        "desktop_rate_all_eligible": all_desktop,
        "mobile_rate_all_eligible": all_mobile,
        "crude_gap_all_eligible": all_mobile - all_desktop,
        "desktop_rate_common": crude[0],
        "mobile_rate_common": crude[1],
        "crude_gap_common": crude[1] - crude[0],
        "desktop_standardized_rate": standardized[0],
        "mobile_standardized_rate": standardized[1],
        "standardized_gap": standardized[1] - standardized[0],
        "standardized_gap_ci_low": np.quantile(estimates, 0.025),
        "standardized_gap_ci_high": np.quantile(estimates, 0.975),
        "crude_common_gap_ci_low": np.quantile(crude_estimates, 0.025),
        "crude_common_gap_ci_high": np.quantile(crude_estimates, 0.975),
        "bootstrap_repetitions": REPS,
        "valid_repetitions": len(estimates),
        "seed": SEED,
        "method": "User-cluster percentile bootstrap; fixed common-support cells and fixed pooled target weights; exploratory nominal 95% interval",
    }
    std_rows.append(row)


for scope, x in scopes.items():
    for eligible, outcome, label in [
        ("has_checkout", "has_positive_after_checkout", "checkout_sessions"),
        ("has_view", "has_positive_after_core_checkout", "view_event_sessions"),
    ]:
        use = x[x[eligible]]
        device_bootstrap(use, outcome, [], "unadjusted", scope, label)
        if scope == "stable_57d":
            device_bootstrap(
                use,
                outcome,
                ["country_group", "source_group"],
                "country_source",
                scope,
                label,
            )
            device_bootstrap(
                use,
                outcome,
                ["country_us", "source_group", "week"],
                "country_source_week",
                scope,
                label,
            )
            device_bootstrap(
                use,
                outcome,
                ["country_us", "source_group", "prior_observed_session"],
                "country_source_prior_history",
                scope,
                label,
            )
pd.DataFrame(std_rows).to_csv(OUT / "session_device_standardization.csv", index=False)
pd.DataFrame(std_cells).to_csv(
    OUT / "session_device_standardization_cells.csv", index=False
)
b = read_table(BASE)
co_sessions = s[s.first_checkout_ts.notna()]
assert not co_sessions.duplicated(["user_pseudo_id", "first_checkout_ts"]).any()
cohort = b.merge(
    co_sessions,
    left_on=["user_key", "index_ts"],
    right_on=["user_pseudo_id", "first_checkout_ts"],
    suffixes=("_baseline", ""),
    validate="one_to_one",
)
assert len(cohort) == len(b)
cohort["index_session_payment"] = cohort.has_payment_after_checkout
cohort["index_session_positive"] = cohort.has_positive_after_checkout
cohort["index_to_payment_seconds"] = (
    cohort.payment_after_checkout_ts - cohort.index_ts
) / 1000000.0
cohort["payment_to_positive_seconds"] = (
    cohort.positive_after_payment_ts - cohort.payment_after_checkout_ts
) / 1000000.0
payment_candidates = (
    pd.concat(
        [
            s[key + [c]].rename(columns={c: "payment_ts"})
            for c in ["first_payment_ts", "payment_after_checkout_ts"]
        ]
    )
    .dropna(subset=["payment_ts"])
    .drop_duplicates()
)
pay = cohort[["user_key", "index_ts"]].merge(
    payment_candidates, left_on="user_key", right_on="user_pseudo_id", how="left"
)
pay = pay[
    (pay.payment_ts > pay.index_ts)
    & (pay.payment_ts < pay.index_ts + 7 * 86400 * 1000000.0)
]
first_pay_7d = pay.groupby("user_key").payment_ts.min()
cohort["first_observed_payment_7d_ts"] = cohort.user_key.map(first_pay_7d)
cohort["observed_payment_any_session_7d"] = cohort.first_observed_payment_7d_ts.notna()
cohort["observed_later_payment_after_no_index_payment"] = (
    ~cohort.index_session_payment & cohort.observed_payment_any_session_7d
)
assert cohort.loc[cohort.index_session_positive, "y7"].eq(1).all()


def wilson(y, n):
    if not n:
        return [None, None]
    z, pp = (1.959963984540054, y / n)
    denom = 1 + z * z / n
    center = (pp + z * z / (2 * n)) / denom
    half = z * math.sqrt(pp * (1 - pp) / n + z * z / (4 * n * n)) / denom
    return [center - half, center + half]


import math

cohort_rows = []
for dim in [
    "index_session_payment",
    "device_category",
    "country",
    "first_user_source",
    "pre_history_present",
    "index_date",
]:
    for value, x in cohort.groupby(dim, dropna=False):
        ci = wilson(int(x.y7.sum()), len(x))
        cohort_rows.append(
            {
                "dimension": dim,
                "value": str(value),
                "eligible_users": len(x),
                "purchase_7d_users": int(x.y7.sum()),
                "purchase_24h_users": int(x.y24.sum()),
                "index_session_positive_users": int(x.index_session_positive.sum()),
                "index_session_payment_users": int(x.index_session_payment.sum()),
                "observed_payment_7d_users": int(
                    x.observed_payment_any_session_7d.sum()
                ),
                "purchase_7d_rate": x.y7.mean(),
                "purchase_7d_wilson_low": ci[0],
                "purchase_7d_wilson_high": ci[1],
                "nonbuyers_7d": int((1 - x.y7).sum()),
            }
        )
pd.DataFrame(cohort_rows).to_csv(
    OUT / "checkout_user_cohort_diagnostics.csv", index=False
)
cohort[
    [
        "user_key",
        "index_ts",
        "index_session_payment",
        "index_session_positive",
        "index_to_payment_seconds",
        "payment_to_positive_seconds",
        "observed_payment_any_session_7d",
        "observed_later_payment_after_no_index_payment",
        "y7",
        "y24",
        "first_purchase_delay_hours",
    ]
].to_csv(OUT / "checkout_user_journey.csv", index=False)
no_pay = cohort[~cohort.index_session_payment]
yes_pay = cohort[cohort.index_session_payment]
joint_history = cohort.groupby(["pre_history_present", "index_session_payment"]).agg(
    eligible_users=("y7", "size"), purchase_users_7d=("y7", "sum")
)
joint_history.to_csv(OUT / "checkout_history_payment_joint.csv")
scenarios = []
for relative_lift in [0.05, 0.1, 0.15, 0.2]:
    absolute_lift = cohort.y7.mean() * relative_lift
    extra = len(cohort) * absolute_lift
    scenarios.append(
        {
            "evidence_type": "assumption_scenario_not_estimated_effect",
            "relative_purchase_rate_lift": relative_lift,
            "absolute_purchase_rate_lift": absolute_lift,
            "eligible_users": len(cohort),
            "incremental_purchase_users": extra,
            "required_recovery_fraction_if_all_gain_from_no_payment_nonbuyers": extra
            / (len(no_pay) - no_pay.y7.sum()),
        }
    )
pd.DataFrame(scenarios).to_csv(OUT / "checkout_opportunity_scenarios.csv", index=False)
summary["checkout_exact_validation_cohort"] = {
    "eligible_users": len(cohort),
    "purchase_7d_users": int(cohort.y7.sum()),
    "purchase_24h_users": int(cohort.y24.sum()),
    "index_session_purchase_users": int(cohort.index_session_positive.sum()),
    "no_index_payment_users": len(no_pay),
    "no_index_payment_7d_purchase_users": int(no_pay.y7.sum()),
    "no_index_payment_later_observed_payment_users": int(
        no_pay.observed_payment_any_session_7d.sum()
    ),
    "with_index_payment_users": len(yes_pay),
    "with_index_payment_7d_purchase_users": int(yes_pay.y7.sum()),
    "no_index_payment_share_of_7d_nonbuyers": float(
        (len(no_pay) - no_pay.y7.sum()) / (len(cohort) - cohort.y7.sum())
    ),
    "purchase_delay_hours_quantiles_among_7d_purchasers": {
        str(q): cohort.loc[cohort.y7.eq(1), "first_purchase_delay_hours"].quantile(q)
        for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    },
    "index_to_payment_seconds_quantiles": {
        str(q): cohort.index_to_payment_seconds.quantile(q)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    },
    "payment_to_positive_seconds_quantiles": {
        str(q): cohort.payment_to_positive_seconds.quantile(q)
        for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    },
}
view = s[s.has_view]
stable_search_checkout = scopes["stable_57d"].loc[
    lambda z: z.has_search & z.has_checkout
]
summary["search_timing_warning"] = {
    "stable_search_and_checkout_sessions": len(stable_search_checkout),
    "first_search_before_first_checkout": int(
        (
            stable_search_checkout.first_search_ts
            < stable_search_checkout.first_checkout_ts
        ).sum()
    ),
    "first_search_at_or_after_first_checkout": int(
        (
            stable_search_checkout.first_search_ts
            >= stable_search_checkout.first_checkout_ts
        ).sum()
    ),
}
summary["view_item_semantic_warning"] = {
    "sessions_with_view_event": len(view),
    "sessions_mean_item_rows_gt_1": int(view.view_mean_item_rows.gt(1).sum()),
    "weighted_item_rows_per_view_event": float(
        view.viewed_item_rows.sum() / view.view_events.sum()
    ),
    "session_mean_item_rows_median": float(view.view_mean_item_rows.median()),
    "max_session_mean_item_rows": float(view.view_mean_item_rows.max()),
    "note": "Session aggregates show many multi-item view_item arrays; this alone does not prove page type, so do not label the trigger a verified single-product detail page.",
}
summary["interpretation_limits"] = [
    "Session conditional rates are not the one-index-per-user experimental endpoint; the validation cohort is reported separately.",
    "Device standardization is conditional association, not an effect of changing device; unobserved purchase intent remains.",
    "Search and whether payment was later reached are post-index self-selected behaviors. Their differences localize journeys, not treatment effects.",
    "Prior observed session/history does not establish real new-versus-existing customer status because observation is left-truncated.",
    "Shipping and checkout timestamps are almost simultaneous; strict arbitrary order between them is not a valid standalone friction diagnosis.",
    "Later payment follow-up scans retained first and post-checkout payment timestamps, not every source payment event; absent later payment is absent in these retained observations.",
    "Intervals are exploratory nominal 95% intervals, not multiplicity-adjusted confirmation. Stable-window and support choices were selected after observing quality data.",
]


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


(OUT / "session_research_summary.json").write_text(
    json.dumps(clean(summary), indent=2, ensure_ascii=False) + "\n"
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "savefig.dpi": 170,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.subplots_adjust(left=0.085, right=0.965, top=0.85, bottom=0.22, wspace=0.32)
xs = np.arange(2)
denom = np.array([len(no_pay), len(yes_pay)])
y = np.array([no_pay.y7.sum(), yes_pay.y7.sum()])
axes[0].bar(xs, denom - y, color="#DAB5A1", label="No economic purchase within 7d")
axes[0].bar(
    xs, y, bottom=denom - y, color="#167D9A", label="Economic purchase proxy within 7d"
)
axes[0].set_xticks(
    xs,
    [
        "No observed payment info\nin index session",
        "Observed payment info\nin index session",
    ],
)
axes[0].set_ylabel("Eligible checkout users")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
for k in range(2):
    axes[0].text(
        k,
        denom[k] + 40,
        f"{y[k]:,}/{denom[k]:,} = {y[k] / denom[k]:.1%}",
        ha="center",
        fontsize=10,
    )
axes[0].set_ylim(0, max(denom) * 1.15)
delays = np.sort(cohort.loc[cohort.y7.eq(1), "first_purchase_delay_hours"].to_numpy())
axes[1].plot(delays, np.arange(1, len(delays) + 1) / len(delays), color="#263B56", lw=2)
axes[1].set(
    xscale="log",
    xlabel="Hours from checkout index (log scale)",
    ylabel="Cumulative share of observed 7d purchasers",
    ylim=(0, 1.02),
)
axes[1].axvline(24, color="#D28133", ls="--")
axes[1].annotate(
    "96.57% purchase within 24h",
    (24, 0.9657),
    xytext=(0.04, 0.78),
    arrowprops={"arrowstyle": "-", "color": "#555"},
    fontsize=9,
)
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
fig.suptitle(
    "Most observed checkout loss occurs before payment information",
    fontsize=15,
    x=0.085,
    y=0.97,
    ha="left",
)
fig.text(
    0.085,
    0.065,
    "Observed public data. Same 4,872-user cohort as validation; one index per user, mature 7d outcome.\nPayment progression is a post-index behavior, not a randomized treatment. 'No payment' means no retained payment-info event.",
    fontsize=9,
    color="#555",
)
fig.savefig(OUT / "checkout_payment_opportunity.png")
plt.close(fig)
std = pd.DataFrame(std_rows)
plot = std[(std.scope == "stable_57d") & (std.eligible == "checkout_sessions")].copy()
fig, ax = plt.subplots(figsize=(10.5, 4.8))
fig.subplots_adjust(left=0.32, right=0.96, top=0.84, bottom=0.25)
names = {
    "unadjusted": "Unadjusted",
    "country_source": "Country + first-user source",
    "country_source_week": "US/non-US + source + week",
    "country_source_prior_history": "US/non-US + source + prior history",
}
v = plot.standardized_gap.to_numpy() * 100
lo, hi = (
    plot.standardized_gap_ci_low.to_numpy() * 100,
    plot.standardized_gap_ci_high.to_numpy() * 100,
)
ax.errorbar(
    v,
    np.arange(len(plot)),
    xerr=np.vstack([v - lo, hi - v]),
    fmt="o",
    color="#167D9A",
    capsize=4,
)
ax.set_yticks(np.arange(len(plot)), [names[x] for x in plot.specification])
ax.axvline(0, color="#777", ls="--")
ax.set_xlabel("Mobile minus desktop checkout-to-purchase rate (percentage points)")
ax.grid(axis="x", alpha=0.2)
fig.suptitle(
    "The data do not establish a mobile checkout deficit",
    x=0.06,
    y=0.965,
    ha="left",
    fontsize=14,
)
fig.text(
    0.06,
    0.065,
    "Observed session association, 2020-11-30 to 2021-01-25. 95% user-cluster bootstrap intervals, 600 resamples.\nAdjusted estimates use fixed pooled weights and common support; detailed coverage is reported in the accompanying CSV.",
    fontsize=9,
    color="#555",
)
fig.savefig(OUT / "checkout_device_standardization.png")
plt.close(fig)
print(
    json.dumps(
        clean(
            {
                "session_metrics": {k: summary[k] for k in ["full_92d", "stable_57d"]},
                "checkout_cohort": summary["checkout_exact_validation_cohort"],
                "device_results": std.to_dict("records"),
            }
        ),
        ensure_ascii=False,
        indent=2,
    )
)
