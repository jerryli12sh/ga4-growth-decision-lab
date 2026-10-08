"""Reproduce descriptive measurement diagnostics from the new 92-day inventory.

No warehouse queries, inferred users, or causal claims. This audit deliberately
does not sum daily DISTINCT user/session counts into unique full-window totals.
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

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/event_inventory.csv"
OUT = ROOT / "results/analysis"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SOURCE, dtype={"event_date": str})
df["date"] = pd.to_datetime(df.event_date, format="%Y%m%d")
assert not df.duplicated(["date", "event_name", "device_category"]).any()
assert (df.events_with_items <= df.event_count).all()
assert (df.purchase_bad_txid <= df.event_count).all()
assert (df.event_count >= 0).all()
dates = pd.date_range(df.date.min(), df.date.max(), freq="D")
assert set(df.date) == set(dates)
counts = (
    df.pivot_table(
        index="date", columns="event_name", values="event_count", aggfunc="sum"
    )
    .reindex(dates)
    .fillna(0)
)
items = (
    df.pivot_table(
        index="date", columns="event_name", values="events_with_items", aggfunc="sum"
    )
    .reindex(dates)
    .fillna(0)
)
purchase = (
    df.loc[df.event_name.eq("purchase")]
    .groupby("date")
    .sum(numeric_only=True)
    .reindex(dates)
)
daily = counts.copy()
daily.columns = ["events_" + x for x in daily.columns]
daily["total_events"] = counts.sum(axis=1)
for event in [
    "view_item",
    "begin_checkout",
    "purchase",
    "select_item",
    "view_item_list",
]:
    daily["with_items_" + event] = items[event]
    daily["item_coverage_" + event] = items[event].div(counts[event].replace(0, np.nan))
daily["purchase_bad_txid"] = purchase.purchase_bad_txid
daily["purchase_bad_txid_share"] = purchase.purchase_bad_txid.div(counts.purchase)
daily["purchase_revenue_usd"] = purchase.purchase_revenue_usd
daily["revenue_per_purchase_event_usd"] = purchase.purchase_revenue_usd.div(
    counts.purchase
)
for event in [
    "add_to_cart",
    "begin_checkout",
    "select_item",
    "view_search_results",
    "purchase",
]:
    daily[event + "_events_per_100_view_events"] = 100 * counts[event].div(
        counts.view_item
    )
daily["checkout_to_shipping_event_count_ratio"] = counts.begin_checkout.div(
    counts.add_shipping_info
)
daily.index.name = "date"
daily.to_csv(OUT / "inventory_daily_diagnostics.csv", float_format="%.10g")


def metrics(start: str, end: str) -> dict:
    a = daily.loc[start:end]
    return {
        "start": start,
        "end": end,
        "days": len(a),
        "events": int(a.total_events.sum()),
        "purchase_events": int(a.events_purchase.sum()),
        "purchase_revenue_usd": float(a.purchase_revenue_usd.sum()),
        "revenue_per_purchase_event_usd": float(
            a.purchase_revenue_usd.sum() / a.events_purchase.sum()
        ),
        "purchase_bad_txid": int(a.purchase_bad_txid.sum()),
        "purchase_bad_txid_share": float(
            a.purchase_bad_txid.sum() / a.events_purchase.sum()
        ),
        "view_events": int(a.events_view_item.sum()),
        "view_items_coverage": float(
            a.with_items_view_item.sum() / a.events_view_item.sum()
        ),
        "checkout_events": int(a.events_begin_checkout.sum()),
        "checkout_items_coverage": float(
            a.with_items_begin_checkout.sum() / a.events_begin_checkout.sum()
        ),
        "add_to_cart_events": int(a.events_add_to_cart.sum()),
        "select_item_events": int(a.events_select_item.sum()),
    }


phases = {
    "full_window": metrics("2020-11-01", "2021-01-31"),
    "november": metrics("2020-11-01", "2020-11-30"),
    "december": metrics("2020-12-01", "2020-12-31"),
    "january": metrics("2021-01-01", "2021-01-31"),
    "initial_txid_placeholder_regime": metrics("2020-11-01", "2020-11-10"),
    "before_checkout_item_enrichment": metrics("2020-11-01", "2020-11-22"),
    "checkout_item_transition": metrics("2020-11-23", "2020-11-29"),
    "candidate_rich_event_window": metrics("2020-11-30", "2021-01-25"),
    "candidate_later_view_coverage_window": metrics("2020-12-15", "2021-01-25"),
    "matched_weekday_reference": metrics("2021-01-19", "2021-01-24"),
    "late_txid_payload_degradation": metrics("2021-01-26", "2021-01-31"),
}
pd.DataFrame.from_dict(phases, orient="index").rename_axis("period").to_csv(
    OUT / "inventory_phase_summary.csv"
)
cells = (
    df.loc[df.event_name.isin(["begin_checkout", "add_shipping_info"])]
    .pivot(
        index=["date", "device_category"], columns="event_name", values="event_sessions"
    )
    .fillna(0)
)
cells["same_session_count"] = cells.begin_checkout.eq(cells.add_shipping_info)
cells.to_csv(OUT / "checkout_shipping_session_count_cells.csv")


def strongest_rate_split(success, trials, min_days=7):
    """Exploratory two-regime fit, no independence assumption or p-value."""
    k, n = (np.asarray(success, float), np.asarray(trials, float))
    best = None
    for t in range(min_days, len(n) - min_days + 1):
        ps = [k[:t].sum() / n[:t].sum(), k[t:].sum() / n[t:].sum()]
        score = sum(
            (
                (n[s] * (k[s] / n[s] - p) ** 2).sum()
                for s, p in [(slice(None, t), ps[0]), (slice(t, None), ps[1])]
            )
        )
        candidate = {
            "right_regime_start": str(dates[t].date()),
            "left_rate": ps[0],
            "right_rate": ps[1],
            "weighted_sse": float(score),
        }
        if best is None or score < best["weighted_sse"]:
            best = candidate
    return best


reference = phases["matched_weekday_reference"]
late = phases["late_txid_payload_degradation"]
summary = {
    "evidence_type": "observed_public_data_descriptive_quality_audit",
    "source": str(SOURCE.relative_to(ROOT)),
    "query_job_id": "ga4sql-502402:US.job_CVw6c7P8g9oLpuQuRlvGj-NwYw33",
    "source_rows": len(df),
    "days": len(dates),
    "zero_cart_dates": [str(d.date()) for d in dates[counts.add_to_cart.eq(0)]],
    "select_item_first_observed_date": str(dates[counts.select_item.gt(0)][0].date()),
    "checkout_items_first_observed_date": str(
        dates[items.begin_checkout.gt(0)][0].date()
    ),
    "checkout_items_complete_contiguous_tail_start": str(
        dates[daily.item_coverage_begin_checkout.ne(1)][-1].date()
        + pd.Timedelta(days=1)
    ),
    "checkout_shipping_equal_session_count_cells": int(cells.same_session_count.sum()),
    "checkout_shipping_total_cells": len(cells),
    "checkout_shipping_unequal_cells": [
        {
            "date": str(i[0].date()),
            "device": i[1],
            "checkout_sessions": int(r.begin_checkout),
            "shipping_sessions": int(r.add_shipping_info),
        }
        for i, r in cells.loc[~cells.same_session_count].iterrows()
    ],
    "monthly_phase_metrics": phases,
    "same_weekday_late_vs_prior_week": {
        "purchase_event_relative_change": late["purchase_events"]
        / reference["purchase_events"]
        - 1,
        "revenue_relative_change": late["purchase_revenue_usd"]
        / reference["purchase_revenue_usd"]
        - 1,
        "revenue_per_purchase_event_relative_change": late[
            "revenue_per_purchase_event_usd"
        ]
        / reference["revenue_per_purchase_event_usd"]
        - 1,
        "bad_txid_share_pp_change": 100
        * (late["purchase_bad_txid_share"] - reference["purchase_bad_txid_share"]),
    },
    "exploratory_single_split_fits": {
        "method": "Minimize trial-count-weighted SSE for two daily rate regimes; min 7 days each; descriptive scan, no statistical test or independent validation.",
        "checkout_item_coverage": strongest_rate_split(
            items.begin_checkout, counts.begin_checkout
        ),
        "view_item_coverage": strongest_rate_split(items.view_item, counts.view_item),
        "bad_transaction_id_share": strongest_rate_split(
            purchase.purchase_bad_txid, counts.purchase
        ),
    },
    "interpretation_limits": [
        "Counts of events are additive; distinct users/sessions are not added across dates or event types as full-window unique totals.",
        "Daily event-count ratios are logging diagnostics, not user conversion rates.",
        "Equal checkout/shipping session counts do not establish equality of actual session sets, timestamps, or a guaranteed path.",
        "Revenue field has no zero-versus-null audit here; purchase_bad_txid means NULL, blank, (not set), or <Other> under source SQL.",
        "Observed regimes may reflect implementation, obfuscation, or business changes; this audit cannot identify their cause.",
        "Suggested phase windows were selected after inspecting data and are exploratory, not untouched holdouts.",
    ],
}
(OUT / "inventory_measurement_audit.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "font.size": 10,
        "savefig.dpi": 170,
    }
)
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig.subplots_adjust(left=0.105, right=0.975, top=0.945, bottom=0.105, hspace=0.13)
axes[0].plot(
    dates,
    100 * daily.item_coverage_begin_checkout,
    label="Checkout events with items",
    color="#167D9A",
    lw=2,
)
axes[0].plot(
    dates,
    100 * daily.item_coverage_view_item,
    label="Product-view events with items",
    color="#D28133",
    lw=2,
)
axes[0].set(ylabel="Item coverage (%)", ylim=(-3, 105))
axes[0].legend(loc="lower right", ncol=2, frameon=False)
axes[1].plot(
    dates,
    daily.add_to_cart_events_per_100_view_events,
    label="Cart / product view",
    color="#167D9A",
)
axes[1].plot(
    dates,
    daily.select_item_events_per_100_view_events,
    label="Item selection / product view",
    color="#834C99",
)
axes[1].set(ylabel="Events per 100\nproduct-view events")
axes[1].legend(loc="upper left", ncol=2, frameon=False)
axes[2].plot(dates, 100 * daily.purchase_bad_txid_share, color="#BB4430", lw=2)
axes[2].set(ylabel="Invalid/placeholder\ntransaction ID (%)", ylim=(-3, 105))
axes[3].plot(dates, daily.revenue_per_purchase_event_usd, color="#263B56", lw=2)
axes[3].set(
    ylabel="Reported USD per\npurchase event", xlabel="Date (property event_date)"
)
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
    ax.axvline(pd.Timestamp("2020-11-23"), color="#666666", ls="--", lw=1)
    ax.axvspan(pd.Timestamp("2021-01-26"), dates.max(), color="#BB4430", alpha=0.1)
axes[0].annotate(
    "Nov 23: item-selection records start",
    (pd.Timestamp("2020-11-23"), 22),
    xytext=(pd.Timestamp("2020-12-01"), 30),
    arrowprops={"arrowstyle": "-", "color": "#555"},
    fontsize=9,
)
axes[3].annotate(
    "Jan 26-31: transaction payload warning",
    (pd.Timestamp("2021-01-28"), 7),
    xytext=(pd.Timestamp("2020-12-17"), 10),
    arrowprops={"arrowstyle": "-", "color": "#555"},
    fontsize=9,
)
axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.suptitle(
    "Observed public data: measurement diagnostics",
    fontsize=16,
    x=0.105,
    y=0.98,
    ha="left",
)
fig.text(
    0.05,
    0.014,
    "Source: new BigQuery event inventory, 2020-11-01 to 2021-01-31. Event ratios are not conversion rates.\nRegime changes are observed; their implementation or business causes are not identified.",
    fontsize=9,
    color="#555",
)
fig.savefig(OUT / "measurement_regimes.png")
plt.close(fig)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.subplots_adjust(left=0.105, right=0.975, top=0.91, bottom=0.17, hspace=0.13)
axes[0].plot(
    dates, daily.events_purchase, color="#167D9A", label="Raw purchase events", lw=1.8
)
axes[0].plot(
    dates,
    daily.events_purchase - daily.purchase_bad_txid,
    color="#283B54",
    label="Purchase events with non-placeholder transaction ID",
    lw=1.8,
)
axes[0].set(ylabel="Purchase event count")
fig.suptitle(
    "Purchase records persist while reported revenue and transaction IDs deteriorate",
    fontsize=12,
    x=0.105,
    y=0.97,
    ha="left",
)
axes[0].legend(frameon=False, fontsize=9)
axes[1].plot(dates, daily.purchase_revenue_usd, color="#D28133", lw=1.8)
axes[1].set(
    ylabel="Reported purchase revenue (USD)", xlabel="Date (property event_date)"
)
for ax in axes:
    ax.axvspan(pd.Timestamp("2021-01-26"), dates.max(), color="#BB4430", alpha=0.1)
    ax.grid(axis="y", alpha=0.2)
axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.text(
    0.08,
    0.014,
    "Observed public data, not backend orders. An apparently valid transaction ID does not establish uniqueness.\nThe inventory cannot separate zero values from missing values within daily revenue sums.",
    fontsize=9,
    color="#555",
)
fig.savefig(OUT / "purchase_revenue_payload.png")
plt.close(fig)
print(
    json.dumps(
        {
            "output_dir": str(OUT),
            "days": len(dates),
            "phases": phases,
            "same_weekday_change": summary["same_weekday_late_vs_prior_week"],
            "equal_session_count_cells": summary[
                "checkout_shipping_equal_session_count_cells"
            ],
            "total_session_count_cells": len(cells),
            "split_fits": summary["exploratory_single_split_fits"],
        },
        ensure_ascii=False,
        indent=2,
    )
)
