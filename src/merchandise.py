"""Audit event semantics, cross-stage item identity, and pre-outcome checkout baskets.

No item-day marginal is interpreted as a same-user conversion funnel. The
concentrated cap-basket block is retained in the primary population and removed
only in a clearly labelled sensitivity, never called bot traffic or a true bug.
"""

from __future__ import annotations
import json
import math
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
paths = {
    "item_daily": ROOT / "data/processed/item_daily.csv",
    "event_semantics": ROOT / "data/processed/event_semantics_profile.csv",
    "checkout_index": ROOT / "data/processed/checkout_index_facts.csv",
    "validation_cohort": ROOT
    / "data/processed/validation/first_checkout_positive_merchandise.csv",
    "cohort_journey": OUT / "checkout_user_journey.csv",
    "session_facts": ROOT / "data/processed/session_facts.csv",
    "checkout_local_price": ROOT / "data/processed/checkout_local_price.csv",
    "price_coverage": ROOT / "data/processed/price_coverage.csv",
}
x = pd.read_csv(paths["item_daily"], dtype={"item_id": "string"})
e = pd.read_csv(paths["event_semantics"])
i = pd.read_csv(
    paths["checkout_index"],
    dtype={"user_pseudo_id": "string", "ga_session_id": "string"},
)
b = read_table(paths["validation_cohort"])
j = read_table(paths["cohort_journey"])
s = pd.read_csv(
    paths["session_facts"],
    dtype={"user_pseudo_id": "string", "ga_session_id": "string"},
)
local_price = pd.read_csv(
    paths["checkout_local_price"],
    dtype={"user_pseudo_id": "string", "ga_session_id": "string"},
)
valid_id = ~x.item_id.isin(["", "(not set)", "<Other>"]) & x.item_id.notna()
assert len(i) == s.checkout_events.gt(0).sum()
assert i.index_rows_same_timestamp.eq(1).all() and i.index_payload_variants.eq(1).all()
assert x.purchase_item_rows.sum() == 16003
assert np.isclose(x.item_purchase_revenue_usd.sum(), 362110)


def wilson(y, n):
    if n == 0:
        return [None, None]
    z, p = (1.959963984540054, y / n)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


summary = {}
sem_rows, path_rows = ([], [])
for scope, sub in [
    ("full_92d", e),
    ("stable_57d", e[e.property_date.between("2020-11-30", "2021-01-25")]),
]:
    for name, group in sub.groupby("event_name"):
        n = int(group.event_count.sum())
        sem_rows.append(
            {
                "scope": scope,
                "event_name": name,
                "events": n,
                "empty_array_events": int(
                    group.loc[group.item_array_length.eq(0), "event_count"].sum()
                ),
                "single_item_events": int(
                    group.loc[group.item_array_length.eq(1), "event_count"].sum()
                ),
                "multi_item_events": int(
                    group.loc[group.item_array_length.gt(1), "event_count"].sum()
                ),
                "twelve_item_events": int(
                    group.loc[group.item_array_length.eq(12), "event_count"].sum()
                ),
                "with_valid_sku_events": int(group.events_with_valid_sku.sum()),
                "with_value_events": int(group.events_with_value.sum()),
                "with_currency_events": int(group.events_with_currency.sum()),
            }
        )
        for path, pg in group.groupby("normalized_page_path"):
            path_rows.append(
                {
                    "scope": scope,
                    "event_name": name,
                    "normalized_page_path": path,
                    "events": int(pg.event_count.sum()),
                    "event_share": pg.event_count.sum() / n,
                    "multi_item_events": int(
                        pg.loc[pg.item_array_length.gt(1), "event_count"].sum()
                    ),
                    "page_title_examples": " | ".join(
                        pg.observed_page_title_examples.dropna()
                        .drop_duplicates()
                        .head(4)
                    ),
                }
            )
pd.DataFrame(sem_rows).to_csv(OUT / "event_semantics_summary.csv", index=False)
pd.DataFrame(path_rows).sort_values(
    ["scope", "event_name", "events"], ascending=[True, True, False]
).to_csv(OUT / "event_semantics_paths.csv", index=False)
summary["semantics"] = sem_rows
stages = {
    "view": "view_item_rows",
    "cart": "cart_item_rows",
    "checkout": "checkout_item_rows",
    "purchase": "purchase_item_rows",
}
identity_rows, overlaps = ([], [])
for scope, xx in [
    ("full_92d", x),
    ("stable_57d", x[x.event_date.between("2020-11-30", "2021-01-25")]),
]:
    ids, names = ({}, {})
    for stage, col in stages.items():
        zz = xx[(xx[col] > 0) & valid_id.reindex(xx.index)]
        ids[stage], names[stage] = (set(zz.item_id), set(zz.item_name))
        for namespace, mask in [
            ("numeric", zz.item_id.str.fullmatch("\\d+")),
            ("GGO", zz.item_id.str.startswith("GGO")),
            (
                "other",
                ~zz.item_id.str.fullmatch("\\d+") & ~zz.item_id.str.startswith("GGO"),
            ),
        ]:
            identity_rows.append(
                {
                    "scope": scope,
                    "stage": stage,
                    "namespace": namespace,
                    "distinct_item_ids": zz.loc[mask, "item_id"].nunique(),
                    "item_rows": int(zz.loc[mask, col].sum()),
                }
            )
    for a, bb in [
        ("view", "purchase"),
        ("view", "checkout"),
        ("cart", "purchase"),
        ("checkout", "purchase"),
    ]:
        overlaps.append(
            {
                "scope": scope,
                "from_stage": a,
                "to_stage": bb,
                "from_valid_ids": len(ids[a]),
                "to_valid_ids": len(ids[bb]),
                "shared_valid_ids": len(ids[a] & ids[bb]),
                "from_names": len(names[a]),
                "to_names": len(names[bb]),
                "shared_names": len(names[a] & names[bb]),
                "to_item_rows_with_id_seen_from_stage": int(
                    xx.loc[xx.item_id.isin(ids[a]), stages[bb]].sum()
                ),
                "to_item_rows": int(
                    xx.loc[
                        ~xx.item_id.isin(["(not set)", "", "<Other>"]), stages[bb]
                    ].sum()
                ),
            }
        )
pd.DataFrame(identity_rows).to_csv(OUT / "item_identity_namespaces.csv", index=False)
pd.DataFrame(overlaps).to_csv(OUT / "item_identity_overlap.csv", index=False)
summary["item_identity_overlap"] = overlaps
name_map = (
    x[valid_id]
    .groupby("item_name")
    .agg(
        distinct_item_ids=("item_id", "nunique"),
        view_item_rows=("view_item_rows", "sum"),
        cart_item_rows=("cart_item_rows", "sum"),
        checkout_item_rows=("checkout_item_rows", "sum"),
        purchase_item_rows=("purchase_item_rows", "sum"),
        item_revenue=("item_purchase_revenue_usd", "sum"),
    )
)
name_map.to_csv(OUT / "item_name_family_audit.csv")
summary["item_price_coverage"] = {
    "rows_with_view": int(x.view_item_rows.gt(0).sum()),
    "view_slice_rows_with_nonnull_min_price_usd": int(
        (x.view_item_rows.gt(0) & x.min_view_price_usd.notna()).sum()
    ),
    "checkout_indices": len(i),
    "checkout_nonnull_event_value": int(i.event_value.notna().sum()),
    "checkout_nonnull_price_quantity_usd": int(
        i.observed_price_quantity_usd.notna().sum()
    ),
    "note": "Only extracted USD prices and event value audited; do not infer unqueried native price availability.",
}
z = b.merge(
    i,
    left_on=["user_key", "index_ts"],
    right_on=["user_pseudo_id", "checkout_index_ts"],
    suffixes=("", "_basket"),
    validate="one_to_one",
)
z = z.merge(
    j[["user_key", "index_session_payment", "index_session_positive"]],
    on="user_key",
    validate="one_to_one",
)
z = z.merge(
    s[
        [
            "user_pseudo_id",
            "ga_session_id",
            "event_count",
            "view_events",
            "cart_events",
            "checkout_events",
            "shipping_events",
            "search_events",
            "landing_page",
        ]
    ],
    on=["user_pseudo_id", "ga_session_id"],
    validate="one_to_one",
)
z = z.merge(
    local_price.drop(columns="currency"),
    on=["user_pseudo_id", "ga_session_id", "checkout_index_ts"],
    validate="one_to_one",
)
assert len(z) == len(b)
z["cap_only"] = z.valid_sku_ids_json.eq('["9188192"]')
z["cap_missing_category"] = z.cap_only & z.observed_category_count.eq(0)
z["has_valid_basket"] = z.valid_sku_count.gt(0)
z["basket_sku_bin"] = pd.cut(
    z.valid_sku_count, [-1, 0, 1, 2, 4, 100], labels=["missing", "1", "2", "3-4", "5+"]
)
z["basket_quantity_bin"] = pd.cut(
    z.observed_quantity_sum.fillna(-1),
    [-2, 0, 1, 2, 4, 9, 19, 10000],
    labels=["missing", "1", "2", "3-4", "5-9", "10-19", "20+"],
)
z["category_coverage"] = np.select(
    [
        ~z.has_valid_basket,
        z.observed_category_count.eq(0),
        z.missing_category_item_rows.gt(0),
    ],
    ["no_valid_item", "all_categories_missing", "partial_categories_missing"],
    default="categories_present",
)
z["local_price_quantity_bin"] = (
    pd.cut(
        z.local_price_quantity_sum,
        [0, 20, 40, 80, 160, np.inf],
        labels=["(0,20]", "(20,40]", "(40,80]", "(80,160]", "(160,inf)"],
    )
    .astype("string")
    .fillna("missing")
)


def metrics(q):
    n, y, pay = (len(q), int(q.y7.sum()), int(q.index_session_payment.sum()))
    no_pay_nonbuyer = int((~q.index_session_payment & q.y7.eq(0)).sum())
    low, high = wilson(y, n)
    sh_low, sh_high = wilson(no_pay_nonbuyer, n - y)
    return {
        "eligible_users": n,
        "purchase_7d_users": y,
        "purchase_7d_rate": y / n if n else None,
        "purchase_7d_wilson_low": low,
        "purchase_7d_wilson_high": high,
        "index_payment_users": pay,
        "index_payment_rate": pay / n if n else None,
        "no_index_payment_users": n - pay,
        "no_payment_nonbuyers": no_pay_nonbuyer,
        "nonbuyers_7d": n - y,
        "no_payment_share_of_nonbuyers": no_pay_nonbuyer / (n - y) if n - y else None,
        "no_payment_share_wilson_low": sh_low,
        "no_payment_share_wilson_high": sh_high,
        "pre_history_users": int(q.pre_history_present.sum()),
        "cap_missing_category_users": int(q.cap_missing_category.sum()),
    }


scopes = {
    "primary_all": z,
    "sensitivity_exclude_cap_missing_category": z[~z.cap_missing_category],
    "sensitivity_exclude_all_cap_only": z[~z.cap_only],
    "sensitivity_index_dec16_onward": z[z.index_date.ge("2020-12-16")],
}
sens = [{"scope": scope, **metrics(q)} for scope, q in scopes.items()]
pd.DataFrame(sens).to_csv(OUT / "checkout_basket_sensitivity.csv", index=False)
summary["checkout_sensitivities"] = sens
rows = []
for scope, q in scopes.items():
    for dim in [
        "basket_sku_bin",
        "basket_quantity_bin",
        "category_coverage",
        "local_price_quantity_bin",
        "contains_apparel",
        "contains_drinkware",
        "contains_bags",
        "contains_accessories",
        "contains_stationery_office",
        "contains_lifestyle",
        "pre_history_present",
        "device_category",
    ]:
        for value, sub in q.groupby(dim, observed=True, dropna=False):
            rows.append(
                {"scope": scope, "dimension": dim, "value": str(value), **metrics(sub)}
            )
pd.DataFrame(rows).to_csv(OUT / "checkout_basket_segments.csv", index=False)
top_sku = (
    z[z.valid_sku_count.eq(1)]
    .groupby("valid_sku_ids_json")
    .apply(lambda q: pd.Series(metrics(q)), include_groups=False)
    .sort_values("eligible_users", ascending=False)
)
top_sku.to_csv(OUT / "checkout_single_sku_audit.csv")
daily = z.groupby("index_date").apply(
    lambda q: pd.Series(metrics(q)), include_groups=False
)
daily.to_csv(OUT / "checkout_basket_daily.csv")
z[
    [
        "user_key",
        "index_ts",
        "cap_only",
        "cap_missing_category",
        "basket_sku_bin",
        "basket_quantity_bin",
        "category_coverage",
        "valid_sku_ids_json",
        "y7",
        "index_session_payment",
    ]
].to_csv(OUT / "checkout_basket_user_audit.csv", index=False)
a = z[z.cap_missing_category]
block_profiles = []
for dim in [
    "device_category",
    "country",
    "first_user_source",
    "landing_page",
    "index_date",
    "observed_quantity_sum",
]:
    for value, q in z.groupby(dim, observed=True, dropna=False):
        block_profiles.append(
            {
                "dimension": dim,
                "value": str(value),
                "all_eligible_users": len(q),
                "block_users": int(q.cap_missing_category.sum()),
                "block_within_group_share": q.cap_missing_category.mean(),
            }
        )
pd.DataFrame(block_profiles).sort_values(
    ["dimension", "block_users"], ascending=[True, False]
).to_csv(OUT / "checkout_concentrated_block_profile.csv", index=False)
joint = []
for scope, q in list(scopes.items())[:2]:
    for (history, size), group in q.groupby(
        ["pre_history_present", "basket_sku_bin"], observed=True
    ):
        joint.append(
            {
                "scope": scope,
                "pre_history_present": history,
                "basket_sku_bin": size,
                **metrics(group),
            }
        )
pd.DataFrame(joint).to_csv(OUT / "checkout_basket_history_joint.csv", index=False)
summary["concentrated_basket_block"] = {
    **metrics(a),
    "rule": 'first checkout valid_sku_ids_json == ["9188192"] AND observed_category_count == 0',
    "product_name_from_separate_item_dictionary": "YouTube Twill Sandwich Cap Black",
    "quantity_two_or_three_users": int(a.observed_quantity_sum.isin([2, 3]).sum()),
    "dec1_to_dec15_users": int(a.index_date.between("2020-12-01", "2020-12-15").sum()),
    "no_14d_history_users": int(a.pre_history_present.eq(0).sum()),
    "home_landing_users": int(
        a.landing_page.eq("https://shop.googlemerchandisestore.com/").sum()
    ),
    "share_of_primary_no_payment_nonbuyers": len(a)
    / metrics(z)["no_payment_nonbuyers"],
    "interpretation": "Exploratory concentration; cannot distinguish real behavior, invalid traffic, instrumentation, or obfuscation. Retained in primary population; exclusion is sensitivity only.",
}
price_rows = []
for scope, q in [
    ("primary_all", z),
    ("cap_missing_category_block", a),
    ("exclude_cap_missing_category", z[~z.cap_missing_category]),
]:
    price_rows.append(
        {
            "scope": scope,
            "eligible_users": len(q),
            "price_observed_users": int(q.local_price_quantity_sum.notna().sum()),
            "zero_local_price_item_rows": int(q.zero_local_price_rows.sum()),
            "negative_local_price_item_rows": int(q.negative_local_price_rows.sum()),
            "min_local_item_price": q.min_local_price.min(),
            "max_local_item_price": q.max_local_price.max(),
            "local_price_quantity_min": q.local_price_quantity_sum.min(),
            "local_price_quantity_median": q.local_price_quantity_sum.median(),
            "local_price_quantity_max": q.local_price_quantity_sum.max(),
            "unit_status": "Currency missing at checkout; unverified native-price units, not USD or observed customer total",
        }
    )
pd.DataFrame(price_rows).to_csv(OUT / "checkout_native_price_audit.csv", index=False)
summary["checkout_native_price_audit"] = price_rows
economics = []
for delta in [0.025, 0.05, 0.075]:
    for contribution in [10, 20, 30]:
        for spend in [0, 500, 1000]:
            inc = 1000 * delta
            gross = inc * contribution
            economics.append(
                {
                    "evidence_type": "assumption_scenario_not_estimated_effect_or_margin",
                    "exposure_scenario": "all_eligible_receive_new_version_rollout_not_50_50_trial",
                    "eligible_users": 1000,
                    "absolute_purchase_rate_lift": delta,
                    "incremental_purchase_users": inc,
                    "assumed_net_contribution_per_incremental_buyer_usd": contribution,
                    "incremental_contribution_before_intervention_cost_usd": gross,
                    "assumed_intervention_cost_per_1000_eligible_usd": spend,
                    "net_contribution_after_intervention_cost_usd": gross - spend,
                    "break_even_cost_per_eligible_usd": delta * contribution,
                    "break_even_absolute_lift_at_assumed_cost": spend
                    / (1000 * contribution),
                }
            )
pd.DataFrame(economics).to_csv(
    OUT / "checkout_unit_economics_scenarios.csv", index=False
)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 170,
    }
)
fig, ax = plt.subplots(
    2, 1, figsize=(12, 7.6), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
)
dt = pd.to_datetime(daily.index)
ax[0].bar(
    dt, daily.eligible_users, color="#c9dbe6", label="All first-checkout eligible users"
)
ax[0].bar(
    dt,
    daily.cap_missing_category_users,
    color="#c64b3f",
    label="Cap-only basket + missing category",
)
ax[0].set(
    ylabel="Eligible users",
    title="A concentrated basket block materially changes the apparent checkout opportunity",
)
ax[0].legend(loc="upper right", frameon=False)
ax[1].plot(
    dt,
    daily.purchase_7d_rate * 100,
    color="#42566b",
    label="Primary: all eligible users",
)
rest = z[~z.cap_missing_category].groupby("index_date").y7.mean().reindex(daily.index)
ax[1].plot(
    dt, rest * 100, color="#b44739", label="Sensitivity: exclude the identified block"
)
ax[1].set(ylabel="7-day purchase rate (%)", xlabel="First eligible checkout date")
ax[1].legend(loc="lower right", frameon=False)
ax[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.text(
    0.07,
    0.015,
    "646 users share the same SKU and missing category; 0 recorded payment and 0 seven-day purchase outcomes.",
    fontsize=9,
)
fig.subplots_adjust(left=0.075, right=0.98, top=0.93, bottom=0.12, hspace=0.19)
for ext in ["png"]:
    fig.savefig(OUT / f"checkout_concentrated_basket_block.{ext}")
plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
sem = pd.DataFrame(sem_rows).query("scope == 'full_92d'").set_index("event_name")
for idx, event in enumerate(["view_item", "select_item"]):
    vals = sem.loc[event]
    parts = [
        vals.empty_array_events,
        vals.single_item_events,
        vals.multi_item_events - vals.twelve_item_events,
        vals.twelve_item_events,
    ]
    labels = ["Empty", "1 item", "2-11 items", "12 items"]
    bars = ax[idx].bar(
        labels,
        np.array(parts) / vals.events * 100,
        color=["#b9c2cc", "#458b87", "#83b0ad", "#c75c48"],
    )
    for bar, value in zip(bars, parts):
        ax[idx].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:,}\n{value / vals.events:.1%}",
            ha="center",
            fontsize=9,
        )
    ax[idx].set(title=event, ylabel="Share of events (%)", ylim=(0, 100))
fig.suptitle(
    "Item-array composition clarifies the meaning of browsing events",
    fontsize=13,
)
fig.text(
    0.055,
    0.035,
    "Full 92-day export; many event arrays contain 12 products. Checkout/purchase item IDs mostly use a different namespace.",
    fontsize=9,
)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.83, wspace=0.23)
for ext in ["png"]:
    fig.savefig(OUT / f"event_item_semantics.{ext}")
plt.close(fig)
# Display the two sensitivity definitions used in the business recommendation.
review = pd.read_csv(ROOT / "results/validation/review_sensitivity_baselines.csv")
selected = review.set_index("sensitivity").loc[
    [
        "original_all_checkout_itt",
        "exclude_observed_single_cap_missing_category_sensitivity",
        "reindexed_new_enrollment_window_from_20201216",
    ]
]
labels = ["All eligible", "Exclude concentrated SKU", "New enrollment from Dec 16"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
y = np.arange(3)
for axis, column, title in [
    (ax[0], "purchase_7d_rate", "7-day purchase rate"),
    (ax[1], "no_payment_share_of_7d_nonbuyers", "Share of nonbuyers before payment"),
]:
    values = 100 * selected[column].to_numpy()
    axis.barh(y, values, color=["#719ba9", "#277e88", "#d3964d"], height=0.52)
    axis.set(
        yticks=y,
        yticklabels=labels if axis is ax[0] else [],
        xlim=(0, 82),
        xlabel="Percent",
        title=title,
    )
    axis.invert_yaxis()
    for row, value in enumerate(values):
        axis.text(value + 1, row, f"{value:.2f}%", va="center", fontsize=10)
fig.text(
    0.04,
    0.025,
    "Retrospective cohort sensitivity • 7-day outcomes • n = 4,872 / 4,226 / 1,944",
    fontsize=9,
)
fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.18, wspace=0.15)
fig.savefig(OUT / "checkout_basket_sensitivity.png")
plt.close(fig)
(OUT / "merchandise_research_summary.json").write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
        default=lambda o: o.item() if isinstance(o, np.generic) else str(o),
    )
    + "\n"
)
print(
    json.dumps(
        {
            "status": "PASS",
            "primary_users": len(z),
            "cap_missing_category_users": len(a),
            "outputs": "results/analysis/merchandise_* and related CSV/PNG files",
        },
        ensure_ascii=False,
    )
)
