"""Transaction/purchased-item quality and opportunity diagnostics.

All source events remain intact. Candidate repeat collapse is a sensitivity
scenario, not an assertion that backend order identity has been recovered.
"""

from __future__ import annotations
import itertools
import json
import math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/transaction_events.csv"
OUT = ROOT / "results/analysis"
OUT.mkdir(parents=True, exist_ok=True)
BAD = {"", "(not set)", "<Other>"}
d = pd.read_csv(
    SOURCE,
    dtype={
        "user_pseudo_id": "string",
        "transaction_id": "string",
        "ga_session_id": "string",
    },
)
assert d.event_id.is_unique
assert d.identical_raw_row_count.eq(1).all()
assert d.event_name.eq("purchase").all()
d["date"] = pd.to_datetime(d.event_date)
d["valid_txid"] = d.transaction_id.notna() & ~d.transaction_id.isin(BAD)
d["positive_amount"] = d.purchase_revenue_usd.gt(0)
d["zero_amount"] = d.purchase_revenue_usd.eq(0)
d["items"] = d.items_json.map(json.loads)


def economic_item(item):
    return item.get("item_id") not in BAD | {None} and (item.get("quantity") or 0) > 0


def basket_signature(items):
    keys = [
        "item_id",
        "item_variant",
        "quantity",
        "price_in_usd",
        "item_revenue_in_usd",
    ]
    basket = [{k: item.get(k) for k in keys} for item in items]
    basket.sort(key=lambda z: json.dumps(z, sort_keys=True))
    return json.dumps(basket, sort_keys=True, separators=(",", ":"))


d["positive_merchandise_event"] = d.positive_amount & d["items"].map(
    lambda x: any((economic_item(z) for z in x))
)
d["basket_signature"] = d["items"].map(basket_signature)
d = d.sort_values(["event_timestamp", "event_id"]).reset_index(drop=True)
repeat_key = [
    "user_pseudo_id",
    "ga_session_id",
    "transaction_id",
    "purchase_revenue_usd",
    "basket_signature",
]
d["candidate_repeat_excess"] = d.valid_txid & d.duplicated(repeat_key, keep="first")
d["candidate_repeat_collapsed_keep"] = ~d.candidate_repeat_excess
d["event_item_revenue_delta"] = d.purchase_revenue_usd - d.item_revenue_sum_usd
core = d.date.between("2020-11-30", "2021-01-25")
scopes = {
    "all_raw_events": d,
    "positive_merchandise_events": d[d.positive_merchandise_event],
    "candidate_repeat_collapsed_all": d[d.candidate_repeat_collapsed_keep],
    "candidate_repeat_collapsed_positive": d[
        d.candidate_repeat_collapsed_keep & d.positive_merchandise_event
    ],
    "core_window_raw": d[core],
    "core_window_collapsed_positive": d[
        core & d.candidate_repeat_collapsed_keep & d.positive_merchandise_event
    ],
}
items = []
for row in d.itertuples():
    for offset, item in enumerate(row.items):
        items.append(
            {
                **item,
                "event_id": row.event_id,
                "item_offset": offset,
                "event_date": row.event_date,
                "user_key": row.user_pseudo_id,
                "positive_merchandise_event": row.positive_merchandise_event,
                "candidate_repeat_collapsed_keep": row.candidate_repeat_collapsed_keep,
            }
        )
i = pd.DataFrame(items)
for col in ["price_in_usd", "quantity", "item_revenue_in_usd"]:
    i[col] = pd.to_numeric(i[col], errors="coerce")
i["valid_item_id"] = i.item_id.notna() & ~i.item_id.isin(BAD)
i["price_quantity_revenue_delta"] = i.item_revenue_in_usd - i.price_in_usd * i.quantity
recomputed = i.groupby("event_id").agg(
    rows=("event_id", "size"),
    item_revenue=("item_revenue_in_usd", lambda x: x.sum(min_count=1)),
    quantity=("quantity", lambda x: x.sum(min_count=1)),
)
audit_join = d.set_index("event_id").join(recomputed)
assert np.allclose(audit_join.rows.fillna(0), audit_join.item_row_count)
assert np.allclose(
    audit_join.item_revenue.fillna(0), audit_join.item_revenue_sum_usd.fillna(0)
)
assert np.allclose(
    audit_join.quantity.fillna(0), audit_join.item_quantity_sum.fillna(0)
)
inventory = pd.read_csv(ROOT / "data/processed/event_inventory.csv")
anchor = inventory[inventory.event_name.eq("purchase")]
assert len(d) == anchor.event_count.sum()
assert d.purchase_revenue_usd.sum() == anchor.purchase_revenue_usd.sum()
tx_groups = (
    d[d.valid_txid]
    .groupby("transaction_id")
    .agg(
        events=("transaction_id", "size"),
        users=("user_pseudo_id", "nunique"),
        days=("event_date", "nunique"),
        amounts=("purchase_revenue_usd", "nunique"),
        min_revenue=("purchase_revenue_usd", "min"),
        max_revenue=("purchase_revenue_usd", "max"),
    )
)
collisions = tx_groups[tx_groups.users.gt(1)]
collisions.to_csv(OUT / "transaction_id_collisions.csv")
user_tx = (
    d[d.valid_txid]
    .groupby(["user_pseudo_id", "transaction_id"])
    .agg(
        events=("transaction_id", "size"),
        sessions=("ga_session_id", "nunique"),
        amounts=("purchase_revenue_usd", "nunique"),
        baskets=("basket_signature", "nunique"),
        start_ts=("event_timestamp", "min"),
        end_ts=("event_timestamp", "max"),
        revenue=("purchase_revenue_usd", "first"),
        first_date=("event_date", "min"),
    )
)
repeats = user_tx[user_tx.events.gt(1)].copy()
repeats["elapsed_seconds"] = (repeats.end_ts - repeats.start_ts) / 1000000.0
repeats.reset_index().drop(columns="user_pseudo_id").to_csv(
    OUT / "transaction_repeat_groups.csv", index=False
)
daily = d.groupby("date").agg(
    purchase_events=("event_name", "size"),
    positive_merchandise_events=("positive_merchandise_event", "sum"),
    zero_amount_events=("zero_amount", "sum"),
    valid_id_events=("valid_txid", "sum"),
    candidate_repeat_excess=("candidate_repeat_excess", "sum"),
    reported_revenue_usd=("purchase_revenue_usd", "sum"),
    purchase_users=("user_pseudo_id", "nunique"),
)
daily["positive_merchandise_users"] = (
    d[d.positive_merchandise_event].groupby("date").user_pseudo_id.nunique()
)
daily.positive_merchandise_users = daily.positive_merchandise_users.fillna(0).astype(
    int
)
daily["zero_amount_event_share"] = daily.zero_amount_events / daily.purchase_events
daily["candidate_repeat_share"] = daily.candidate_repeat_excess / daily.purchase_events
daily["collapsed_reported_revenue_usd"] = (
    d[d.candidate_repeat_collapsed_keep].groupby("date").purchase_revenue_usd.sum()
)
daily.to_csv(OUT / "transaction_daily_quality.csv", float_format="%.10g")
d[
    [
        "event_id",
        "event_date",
        "valid_txid",
        "positive_amount",
        "positive_merchandise_event",
        "zero_amount",
        "candidate_repeat_excess",
        "basket_signature",
        "event_item_revenue_delta",
    ]
].to_csv(OUT / "transaction_event_quality.csv", index=False)
distribution, concentration, scope_stats = ([], [], {})
for scope, x in scopes.items():
    revenue = x.purchase_revenue_usd
    scope_stats[scope] = {
        "events": len(x),
        "users": x.user_pseudo_id.nunique(),
        "sessions": len(x[["user_pseudo_id", "ga_session_id"]].drop_duplicates()),
        "revenue_usd": revenue.sum(),
        "item_revenue_usd": x.item_revenue_sum_usd.sum(),
        "mean_reported_revenue_per_event": revenue.mean(),
        "median_reported_revenue_per_event": revenue.median(),
    }
    for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 1.0]:
        distribution.append(
            {
                "scope": scope,
                "quantile": q,
                "revenue_usd": revenue.quantile(q),
                "item_quantity_sum_source": x.item_quantity_sum.quantile(q),
            }
        )
    for share in [0.01, 0.05, 0.1]:
        n = math.ceil(len(x) * share)
        chosen = x.nlargest(n, "purchase_revenue_usd")
        concentration.append(
            {
                "scope": scope,
                "rule": f"top_{share:.0%}_events_by_revenue",
                "selected_events": len(chosen),
                "event_share": len(chosen) / len(x),
                "revenue_usd": chosen.purchase_revenue_usd.sum(),
                "revenue_share": chosen.purchase_revenue_usd.sum() / revenue.sum(),
            }
        )
    for threshold in [10, 20, 50, 100]:
        chosen = x[x.item_quantity_sum.ge(threshold)]
        concentration.append(
            {
                "scope": scope,
                "rule": f"quantity_ge_{threshold}",
                "selected_events": len(chosen),
                "event_share": len(chosen) / len(x),
                "revenue_usd": chosen.purchase_revenue_usd.sum(),
                "revenue_share": chosen.purchase_revenue_usd.sum() / revenue.sum(),
            }
        )
pd.DataFrame(distribution).to_csv(
    OUT / "transaction_amount_distribution.csv", index=False
)
pd.DataFrame(concentration).to_csv(OUT / "transaction_concentration.csv", index=False)
category_tables, name_tables = ([], [])
for scope in [
    "all_raw_events",
    "candidate_repeat_collapsed_positive",
    "core_window_collapsed_positive",
]:
    selected = i[i.event_id.isin(scopes[scope].event_id)]
    for group, collector in [
        ("item_category", category_tables),
        ("item_name", name_tables),
    ]:
        tab = (
            selected.groupby(group, dropna=False)
            .agg(
                purchase_events=("event_id", "nunique"),
                purchasing_users=("user_key", "nunique"),
                item_quantity=("quantity", lambda x: x.sum(min_count=1)),
                item_revenue_usd=("item_revenue_in_usd", lambda x: x.sum(min_count=1)),
                sku_ids=("item_id", "nunique"),
            )
            .reset_index()
        )
        tab["scope"] = scope
        tab["item_revenue_share"] = (
            tab.item_revenue_usd / selected.item_revenue_in_usd.sum()
        )
        collector.append(tab)
pd.concat(category_tables).to_csv(
    OUT / "transaction_category_structure.csv", index=False
)
pd.concat(name_tables).to_csv(
    OUT / "transaction_product_name_structure.csv", index=False
)
id_structure = (
    i[i.valid_item_id]
    .groupby("item_id")
    .agg(
        item_name=("item_name", "first"),
        names=("item_name", "nunique"),
        category_labels=("item_category", "nunique"),
        categories=("item_category", lambda x: " / ".join(sorted(set(x)))),
        price_values=("price_in_usd", "nunique"),
        min_price=("price_in_usd", "min"),
        max_price=("price_in_usd", "max"),
        purchase_events=("event_id", "nunique"),
    )
)
id_structure.to_csv(OUT / "transaction_product_label_consistency.csv")
selected_i = i[i.event_id.isin(scopes["candidate_repeat_collapsed_positive"].event_id)]
cat_sets = selected_i.groupby("event_id").item_category.agg(set)
core_categories = [
    "Apparel",
    "Bags",
    "Drinkware",
    "Accessories",
    "Office",
    "Stationery",
]
pairs = []
for a, b in itertools.combinations(core_categories, 2):
    n_a = int(cat_sets.map(lambda s: a in s).sum())
    n_b = int(cat_sets.map(lambda s: b in s).sum())
    n_ab = int(cat_sets.map(lambda s: a in s and b in s).sum())
    pairs.append(
        {
            "category_a": a,
            "category_b": b,
            "positive_baskets": len(cat_sets),
            "baskets_a": n_a,
            "baskets_b": n_b,
            "baskets_both": n_ab,
            "b_given_a": n_ab / n_a if n_a else None,
            "cooccurrence_lift_among_purchases": (
                n_ab * len(cat_sets) / (n_a * n_b) if n_a * n_b else None
            ),
        }
    )
pd.DataFrame(pairs).to_csv(OUT / "transaction_basket_category_pairs.csv", index=False)
zero = d[d.zero_amount]
zero_i = i[i.event_id.isin(zero.event_id)]
zero_ph = d[d.zero_amount & d.item_row_count.gt(0)]
summary = {
    "source": str(SOURCE.relative_to(ROOT)),
    "evidence_type": "observed_public_data_with_explicit_repeat_sensitivity",
    "source_records": len(d),
    "raw_row_duplicates": int(d.event_id.duplicated().sum()),
    "purchase_revenue_missing": int(d.purchase_revenue_usd.isna().sum()),
    "purchase_revenue_negative": int(d.purchase_revenue_usd.lt(0).sum()),
    "zero_amount_events": len(zero),
    "zero_amount_all_invalid_txid": bool((~zero.valid_txid).all()),
    "zero_amount_with_empty_items": int(zero.item_row_count.eq(0).sum()),
    "zero_amount_with_placeholder_items": len(zero_ph),
    "zero_amount_item_rows": len(zero_i),
    "zero_amount_item_ids_all_placeholder": bool((~zero_i.valid_item_id).all()),
    "zero_amount_item_numeric_fields_all_missing": bool(
        zero_i[["quantity", "price_in_usd", "item_revenue_in_usd"]].isna().all().all()
    ),
    "all_bad_transaction_events": int((~d.valid_txid).sum()),
    "positive_amount_bad_id_events": int((d.positive_amount & ~d.valid_txid).sum()),
    "valid_transaction_id_count": len(tx_groups),
    "cross_user_transaction_id_collisions": len(collisions),
    "repeated_user_transaction_groups": len(repeats),
    "repeated_user_transaction_events": int(repeats.events.sum()),
    "repeated_user_transaction_all_same_basket_amount_session": bool(
        repeats[["sessions", "amounts", "baskets"]].eq(1).all().all()
    ),
    "repeat_group_elapsed_seconds_quantiles": {
        str(q): repeats.elapsed_seconds.quantile(q)
        for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    },
    "candidate_repeat_excess_events": int(d.candidate_repeat_excess.sum()),
    "candidate_repeat_excess_revenue": float(
        d.loc[d.candidate_repeat_excess, "purchase_revenue_usd"].sum()
    ),
    "candidate_repeat_revenue_share": float(
        d.loc[d.candidate_repeat_excess, "purchase_revenue_usd"].sum()
        / d.purchase_revenue_usd.sum()
    ),
    "item_rows": len(i),
    "nonplaceholder_sku_ids": int(i.loc[i.valid_item_id, "item_id"].nunique()),
    "item_numeric_quality": {
        col: {
            "missing": int(i[col].isna().sum()),
            "zero": int(i[col].eq(0).sum()),
            "negative": int(i[col].lt(0).sum()),
        }
        for col in ["price_in_usd", "quantity", "item_revenue_in_usd"]
    },
    "sku_ids_with_multiple_category_labels": int(
        id_structure.category_labels.gt(1).sum()
    ),
    "sku_ids_with_multiple_names": int(id_structure.names.gt(1).sum()),
    "event_item_amount_reconciliation": {
        "event_revenue_usd": float(d.purchase_revenue_usd.sum()),
        "item_revenue_usd": float(i.item_revenue_in_usd.sum()),
        "net_delta_usd": float(d.event_item_revenue_delta.sum()),
        "sum_absolute_delta_usd": float(d.event_item_revenue_delta.abs().sum()),
        "nonempty_events_with_mismatch_gt_cent": int(
            d.event_item_revenue_delta.abs().gt(0.01).sum()
        ),
        "max_absolute_event_delta_usd": float(d.event_item_revenue_delta.abs().max()),
    },
    "item_price_quantity_reconciliation": {
        "comparable_rows": int(i.price_quantity_revenue_delta.notna().sum()),
        "mismatch_rows_gt_cent": int(
            i.price_quantity_revenue_delta.abs().gt(0.01).sum()
        ),
        "sum_absolute_delta_usd": float(i.price_quantity_revenue_delta.abs().sum()),
        "max_absolute_delta_usd": float(i.price_quantity_revenue_delta.abs().max()),
    },
    "shipping_amount_nonmissing_events": int(d.shipping_value_usd.notna().sum()),
    "tax_amount_nonmissing_events": int(d.tax_value_usd.notna().sum()),
    "refund_event_count_in_export": int(d.event_name.eq("refund").sum()),
    "scope_metrics": scope_stats,
    "assumptions_and_limits": [
        "positive_merchandise_event is an observable economic-purchase proxy, not proof of backend payment success; excluded events may be incomplete records of true purchases.",
        "Candidate repeat excess uses same user, same session, non-placeholder transaction ID, same event amount and identical sorted item ID/variant/quantity/price/revenue basket. All source events remain preserved.",
        "A global transaction ID alone is unsafe for order deduplication because 15 IDs collide across users and days.",
        "Item amount reconciliation is approximate; total near-agreement hides offsetting event discrepancies. Shipping values are wholly absent and refund events are absent.",
        "Category labels include merchandising destinations and may vary within SKU. Category and product-name revenue shares are observed purchased structure, not conversion or causal opportunity rankings.",
        "Basket cooccurrence uses purchasers only and is not incremental cross-sell lift. Bulk thresholds are descriptive choices, not a classification of corporate customers.",
    ],
}


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_json(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if np.isnan(x) else float(x)
    return x


(OUT / "transaction_research_summary.json").write_text(
    json.dumps(clean_json(summary), ensure_ascii=False, indent=2) + "\n"
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
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.subplots_adjust(left=0.105, right=0.975, top=0.92, bottom=0.135, hspace=0.16)
axes[0].plot(
    daily.index, daily.purchase_events, color="#167D9A", label="All purchase events"
)
axes[0].plot(
    daily.index,
    daily.positive_merchandise_events,
    color="#263B56",
    label="Positive-merchandise purchase proxy",
)
axes[0].set_ylabel("Purchase events")
axes[0].legend(frameon=False, fontsize=9)
axes[1].plot(daily.index, 100 * daily.zero_amount_event_share, color="#BB4430")
axes[1].set_ylabel("Zero amount with\nplaceholder payload (%)")
axes[2].plot(daily.index, 100 * daily.candidate_repeat_share, color="#834C99")
axes[2].set_ylabel("Candidate repeat\nexcess events (%)")
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
    ax.axvspan(
        pd.Timestamp("2021-01-26"),
        pd.Timestamp("2021-01-31"),
        color="#BB4430",
        alpha=0.1,
    )
axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[-1].set_xlabel("Date (property event_date)")
fig.suptitle(
    "Observed purchase counts need payload and repeat sensitivity",
    fontsize=15,
    x=0.105,
    y=0.97,
    ha="left",
)
fig.text(
    0.105,
    0.035,
    "Source: all 5,692 new purchase records. Zero amounts are recorded values; placeholder numeric item fields are missing.\nRepeat collapse is a sensitivity scenario, not verified backend deduplication. The economic proxy is not a verified payment label.",
    fontsize=9,
    color="#555",
)
fig.savefig(OUT / "transaction_quality_regimes.png")
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.subplots_adjust(left=0.08, right=0.975, bottom=0.22, top=0.87, wspace=0.25)
pos = d[d.positive_merchandise_event].purchase_revenue_usd
axes[0].hist(
    pos, bins=np.geomspace(pos.min(), pos.max() + 1, 36), color="#167D9A", alpha=0.9
)
axes[0].set(
    xscale="log",
    xlabel="Reported revenue per positive purchase event (USD; log scale)",
    ylabel="Event count",
    title=f"Median ${pos.median():.0f}; upper tail is material",
)
ordered = d.purchase_revenue_usd.sort_values(ascending=False).to_numpy()
axes[1].plot(
    100 * np.arange(1, len(ordered) + 1) / len(ordered),
    100 * ordered.cumsum() / ordered.sum(),
    color="#263B56",
    lw=2,
)
axes[1].set(
    xlabel="Top share of purchase events by revenue (%)",
    ylabel="Share of reported revenue (%)",
    title="Top 1% contribute 8.91% of revenue",
    xlim=(0, 100),
    ylim=(0, 100),
)
for ax in axes:
    ax.grid(axis="y", alpha=0.2)
fig.suptitle(
    "Recorded purchase value and concentration", x=0.08, y=0.97, ha="left", fontsize=15
)
fig.text(
    0.08,
    0.07,
    "Observed public data. Histogram excludes 450 zero-amount placeholder events; concentration includes all 5,692 records.\nAll amounts are purchase-event values, not verified order AOV, profit, or causal uplift.",
    fontsize=9,
    color="#555",
)
fig.savefig(OUT / "transaction_amount_concentration.png")
plt.close(fig)
names = pd.concat(name_tables)
top = (
    names[names.scope.eq("candidate_repeat_collapsed_positive")]
    .nlargest(12, "item_revenue_usd")
    .iloc[::-1]
)
fig, ax = plt.subplots(figsize=(12, 6))
fig.subplots_adjust(left=0.36, right=0.96, top=0.87, bottom=0.18)
ax.barh(top.item_name, top.item_revenue_usd, color="#167D9A")
ax.set_xlabel("Reported purchased-item revenue (USD)")
ax.grid(axis="x", alpha=0.2)
fig.suptitle(
    "Apparel names lead purchased revenue candidates",
    x=0.06,
    y=0.97,
    ha="left",
    fontsize=15,
)
fig.text(
    0.06,
    0.045,
    "Observed public data; positive-merchandise baskets with candidate repeats collapsed. Product names aggregate size/variant SKUs.\nThis chart prioritizes matching to view data; purchased revenue alone does not show conversion opportunity or justify discounting.",
    fontsize=9,
    color="#555",
)
fig.savefig(OUT / "transaction_top_products.png")
plt.close(fig)
print(
    json.dumps(
        clean_json(
            {
                k: summary[k]
                for k in [
                    "source_records",
                    "zero_amount_events",
                    "candidate_repeat_excess_events",
                    "candidate_repeat_excess_revenue",
                    "cross_user_transaction_id_collisions",
                    "item_rows",
                    "nonplaceholder_sku_ids",
                    "sku_ids_with_multiple_category_labels",
                    "scope_metrics",
                ]
            }
        ),
        indent=2,
    )
)
