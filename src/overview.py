"""Funnel, retention, device/source and daily metrics in a common denominator."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/overview"
OUT.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(
    ROOT / "data/processed/session_facts.csv",
    dtype={"user_pseudo_id": "string", "ga_session_id": "string"},
)
# A strict successor path is nested by construction; marginal counts are separate.
s["view"] = s.first_view_ts.notna()
s["checkout_after_view"] = s.checkout_after_view_ts.notna()
s["purchase_after_view_checkout"] = s.purchase_after_view_checkout_ts.notna()
s["checkout"] = s.first_checkout_ts.notna()
s["purchase_after_checkout"] = s.purchase_after_checkout_ts.notna()
rows = []
for window, x in [
    ("full_92d", s),
    ("november", s[s.session_date.between("2020-11-01", "2020-11-30")]),
    ("stable_57d", s[s.session_date.between("2020-11-30", "2021-01-25")]),
]:
    counts = [
        len(x),
        int(x.view.sum()),
        int(x.checkout_after_view.sum()),
        int(x.purchase_after_view_checkout.sum()),
    ]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    for j, (stage, n) in enumerate(
        zip(
            ["All sessions", "View", "Checkout after view", "Purchase after checkout"],
            counts,
        )
    ):
        rows.append(
            dict(
                window=window,
                stage=stage,
                sessions=n,
                previous_stage=counts[max(0, j - 1)],
                step_rate=n / counts[max(0, j - 1)],
                rate_of_all_sessions=n / len(x),
            )
        )
pd.DataFrame(rows).to_csv(OUT / "session_funnel.csv", index=False)
# Device/source denominator is checkout sessions, revenue is restricted to those sessions.
segment = s.groupby(["device_category", "first_user_source"], dropna=False).agg(
    sessions=("user_pseudo_id", "size"),
    view_sessions=("view", "sum"),
    checkout_sessions=("checkout", "sum"),
    ordered_purchases=("purchase_after_checkout", "sum"),
)
checkout_revenue = (
    s[s.checkout]
    .groupby(["device_category", "first_user_source"], dropna=False)
    .raw_purchase_revenue_usd.sum()
)
segment["checkout_purchase_rate"] = (
    segment.ordered_purchases / segment.checkout_sessions.replace(0, np.nan)
)
segment["checkout_session_revenue_usd"] = checkout_revenue
segment["revenue_per_checkout_session"] = (
    segment.checkout_session_revenue_usd / segment.checkout_sessions.replace(0, np.nan)
)
segment.to_csv(OUT / "device_source_metrics.csv")
daily = s.groupby("session_date").agg(
    sessions=("user_pseudo_id", "size"),
    observed_ids=("user_pseudo_id", "nunique"),
    view_sessions=("view", "sum"),
    checkout_after_view=("checkout_after_view", "sum"),
    purchase_after_view_checkout=("purchase_after_view_checkout", "sum"),
    checkout_sessions=("checkout", "sum"),
    purchase_after_checkout=("purchase_after_checkout", "sum"),
    purchase_events=("purchase_events", "sum"),
    raw_revenue_usd=("raw_purchase_revenue_usd", "sum"),
)
daily["checkout_purchase_rate"] = (
    daily.purchase_after_checkout / daily.checkout_sessions.replace(0, np.nan)
)
daily["revenue_per_session"] = daily.raw_revenue_usd / daily.sessions
# July's saved 92-day export is reconciled at shared count and revenue grains.
prior = pd.read_csv(ROOT / "data/processed/daily_metrics.csv").set_index("session_date")
comparison = daily[["sessions", "purchase_events", "raw_revenue_usd"]].join(
    prior[["sessions", "transactions", "revenue_usd"]], rsuffix="_july"
)
comparison["sessions_delta"] = comparison.sessions - comparison.sessions_july
comparison["purchase_events_delta"] = (
    comparison.purchase_events - comparison.transactions
)
comparison["revenue_delta"] = comparison.raw_revenue_usd - comparison.revenue_usd
assert np.allclose(
    comparison[["sessions_delta", "purchase_events_delta", "revenue_delta"]], 0
)
comparison.to_csv(OUT / "daily_reconciliation.csv")
daily.to_csv(OUT / "daily_metrics.csv")
ret = pd.read_csv(ROOT / "data/processed/retention_cohorts.csv")
ret["retention_rate"] = ret.active_size / ret.cohort_size
ret.to_csv(OUT / "retention_cohorts.csv", index=False)
pooled = ret.groupby("delta_day").agg(
    retained=("active_size", "sum"), cohort=("cohort_size", "sum")
)
pooled["retention_rate"] = pooled.retained / pooled.cohort
pooled.to_csv(OUT / "retention_summary.csv")
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
f = pd.DataFrame(rows).query("window == 'full_92d'")
axes[0].barh(f.stage.iloc[::-1], f.sessions.iloc[::-1], color="#257d91")
axes[0].set_xlabel("Sessions with strict successor events")
for y, n in enumerate(f.sessions.iloc[::-1]):
    axes[0].text(n, y, f" {n:,}", va="center", fontsize=9)
axes[0].set_xlim(0, len(s) * 1.24)
axes[0].xaxis.set_major_locator(MaxNLocator(5))
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1000:.0f}k"))
axes[1].plot(
    pooled.index[1:], 100 * pooled.retention_rate.iloc[1:], marker="o", color="#c47831"
)
axes[1].set(
    xlabel="Exact calendar day after first_visit",
    ylabel="Retention (%)",
    title="November first_visit cohorts",
)
fig.tight_layout()
fig.savefig(OUT / "funnel_retention.png", dpi=170)
plt.close(fig)
fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
x = pd.to_datetime(daily.index)
ax[0].plot(x, daily.sessions, color="#257d91")
ax[0].set_ylabel("Sessions")
ax[1].plot(x, 100 * daily.checkout_purchase_rate, color="#c47831")
ax[1].set_ylabel("Checkout → purchase (%)")
for a in ax:
    a.axvspan(
        pd.Timestamp("2021-01-26"),
        pd.Timestamp("2021-01-31"),
        color="#b54c40",
        alpha=0.12,
    )
fig.tight_layout()
fig.savefig(OUT / "daily_trends.png", dpi=170)
plt.close(fig)
print("Funnel, retention, device/source and daily metrics complete")
