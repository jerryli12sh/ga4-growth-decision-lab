"""Independent fact/endpoint review and targeted basket/time sensitivity checks.

This does not estimate a new causal model or replace the original ITT population.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from cohort import read_table, prepare, utc_us, DAY, qualifying_purchase
from validation import power_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/validation"
BASE = ROOT / "data/processed/validation"


def main():
    source = ROOT / "data/processed/session_facts.csv"
    purchase = ROOT / "data/processed/transaction_events.csv"
    basket_file = ROOT / "data/processed/checkout_index_facts.csv"
    quality = ROOT / "results/analysis/transaction_event_quality.csv"
    s = read_table(source)
    p = read_table(purchase)
    basket = read_table(basket_file)
    b = read_table(BASE / "first_checkout_positive_merchandise.csv")
    quality_flags = pd.read_csv(quality)[
        ["event_id", "positive_merchandise_event", "candidate_repeat_excess"]
    ]
    px = p.merge(quality_flags, on="event_id", validate="one_to_one")
    assert np.array_equal(
        qualifying_purchase(px, "positive_merchandise"),
        px.positive_merchandise_event.to_numpy(),
    )
    dates = pd.to_datetime(s.session_start_ts, unit="us", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    purchase_dates = pd.to_datetime(p.event_timestamp, unit="us", utc=True).dt.strftime(
        "%Y-%m-%d"
    )
    start = utc_us("2020-11-30T00:00:00Z")
    end = utc_us("2021-01-19T00:00:00Z")
    obs_end = utc_us("2021-01-26T00:00:00Z")
    eligible = s[s.first_checkout_ts.ge(start) & s.first_checkout_ts.lt(end)]
    independent_index = eligible.groupby("user_pseudo_id").first_checkout_ts.min()
    assert len(independent_index) == len(b) and np.array_equal(
        b.user_key.map(independent_index).to_numpy(), b.index_ts.to_numpy()
    )
    hits = b[["user_key", "index_ts"]].merge(
        px[px.positive_merchandise_event],
        left_on="user_key",
        right_on="user_pseudo_id",
        how="inner",
    )
    hits7 = hits[
        hits.event_timestamp.gt(hits.index_ts)
        & hits.event_timestamp.lt(hits.index_ts + 7 * DAY)
    ]
    hits24 = hits[
        hits.event_timestamp.gt(hits.index_ts)
        & hits.event_timestamp.lt(hits.index_ts + DAY)
    ]
    assert np.array_equal(
        b.user_key.isin(hits7.user_key).astype(int).to_numpy(), b.y7.to_numpy()
    )
    assert np.array_equal(
        b.user_key.isin(hits24.user_key).astype(int).to_numpy(), b.y24.to_numpy()
    )
    co = s[s.first_checkout_ts.notna()]
    index_session = b.merge(
        co,
        left_on=["user_key", "index_ts"],
        right_on=["user_pseudo_id", "first_checkout_ts"],
        validate="one_to_one",
        suffixes=("_baseline", ""),
    )
    joined = b.merge(
        basket,
        left_on=["user_key", "index_ts"],
        right_on=["user_pseudo_id", "checkout_index_ts"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    assert joined["_merge"].eq("both").all()
    block = joined.valid_sku_ids_json.map(json.loads).map(
        lambda ids: ids == ["9188192"]
    ) & joined.observed_category_count.eq(0)
    late_start = utc_us("2020-12-16T00:00:00Z")
    late_reindexed, late_audit = prepare(
        s,
        p,
        "first_checkout_ts",
        late_start,
        end,
        obs_end,
        purchase_definition="positive_merchandise",
    )
    late_audit["inputs"] = [dict(path=str(source)), dict(path=str(purchase))]
    samples = {
        "original_all_checkout_itt": b,
        "exclude_observed_single_cap_missing_category_sensitivity": b.loc[
            ~block.to_numpy()
        ].copy(),
        "original_cohort_index_on_or_after_20201216_subset": b[
            b.index_ts.ge(late_start)
        ].copy(),
        "reindexed_new_enrollment_window_from_20201216": late_reindexed,
    }
    metrics = []
    power = []
    for name, data in samples.items():
        narrow = data.merge(
            co[["user_pseudo_id", "first_checkout_ts", "payment_after_checkout_ts"]],
            left_on=["user_key", "index_ts"],
            right_on=["user_pseudo_id", "first_checkout_ts"],
            validate="one_to_one",
        )
        payment = narrow.payment_after_checkout_ts.notna()
        no_payment_nonbuyers = int((~payment & narrow.y7.eq(0)).sum())
        nonbuyers = int(narrow.y7.eq(0).sum())
        dates2 = pd.to_datetime(data.index_date)
        metric = dict(
            sensitivity=name,
            eligible_users=len(data),
            purchase_7d_users=int(data.y7.sum()),
            purchase_7d_rate=float(data.y7.mean()),
            purchase_24h_users=int(data.y24.sum()),
            index_payment_users=int(payment.sum()),
            no_index_payment_users=int((~payment).sum()),
            no_index_payment_7d_nonbuyers=no_payment_nonbuyers,
            nonbuyers_7d=nonbuyers,
            no_payment_share_of_7d_nonbuyers=no_payment_nonbuyers / nonbuyers,
            window_first_observed_index_date=str(dates2.min().date()),
            window_last_observed_index_date=str(dates2.max().date()),
            evidence="retrospective sensitivity, not production eligibility or observed treatment effect",
        )
        metrics.append(metric)
        table = power_table(data, 0.05, extra_absolute_effects=(0.025, 0.075))
        table.insert(0, "sensitivity", name)
        power.append(table)
        if name != "original_all_checkout_itt":
            target = BASE / (name + ".csv")
            data.to_csv(target, index=False)
            if name == "reindexed_new_enrollment_window_from_20201216":
                late_audit["output"] = dict(path=str(target))
                target.with_suffix(".audit.json").write_text(
                    json.dumps(late_audit, ensure_ascii=False, indent=2)
                )
    pd.DataFrame(metrics).to_csv(OUT / "review_sensitivity_baselines.csv", index=False)
    pd.concat(power, ignore_index=True).to_csv(
        OUT / "review_sensitivity_power.csv", index=False
    )
    no_payment = index_session[index_session.payment_after_checkout_ts.isna()][
        ["user_key", "index_ts"]
    ]
    overlap = no_payment.merge(s, left_on="user_key", right_on="user_pseudo_id")
    possible_missing = overlap[
        overlap.session_start_ts.lt(overlap.index_ts)
        & overlap.session_end_ts.gt(overlap.index_ts)
        & overlap.first_payment_ts.lt(overlap.index_ts)
        & overlap.payment_events.gt(1)
    ]
    stable = s[s.session_date.between("2020-11-30", "2021-01-25")]
    stable_purchases = px[px.positive_merchandise_event].merge(
        stable[["user_pseudo_id", "ga_session_id", "first_checkout_ts"]],
        on=["user_pseudo_id", "ga_session_id"],
        validate="many_to_one",
    )
    stable_purchases = stable_purchases[
        stable_purchases.event_timestamp.gt(stable_purchases.first_checkout_ts)
    ]
    first_stable_purchase = stable_purchases.groupby(
        ["user_pseudo_id", "ga_session_id"]
    ).event_timestamp.min()
    audit = dict(
        evidence="independent deterministic checks of new public-data facts",
        session_rows=len(s),
        unique_session_keys=int(
            s[["user_pseudo_id", "ga_session_id"]].drop_duplicates().shape[0]
        ),
        purchase_rows=len(p),
        all_session_date_equals_utc_start=bool(dates.eq(s.session_date).all()),
        all_purchase_event_date_equals_utc_timestamp=bool(
            purchase_dates.eq(p.event_date).all()
        ),
        independent_index_users=len(independent_index),
        independent_purchase_7d_users=hits7.user_key.nunique(),
        independent_purchase_24h_users=hits24.user_key.nunique(),
        min_cohort_index_utc=str(pd.to_datetime(b.index_ts.min(), unit="us", utc=True)),
        max_cohort_index_utc=str(pd.to_datetime(b.index_ts.max(), unit="us", utc=True)),
        max_cohort_window_end_utc=str(
            pd.to_datetime((b.index_ts + 7 * DAY).max(), unit="us", utc=True)
        ),
        observation_end_utc="2021-01-26T00:00:00Z",
        all_windows_mature=bool((b.index_ts + 7 * DAY <= obs_end).all()),
        users_index_hour_17_utc=int(
            pd.to_datetime(b.index_ts, unit="us", utc=True).dt.hour.eq(17).sum()
        ),
        user_purchase_timestamp_ties=int(b.purchase_timestamp_tie.sum()),
        index_session_payment_later_than_7d=int(
            index_session.payment_after_checkout_ts.gt(
                index_session.index_ts + 7 * DAY
            ).sum()
        ),
        index_session_end_later_than_7d=int(
            index_session.session_end_ts.gt(index_session.index_ts + 7 * DAY).sum()
        ),
        maximum_index_session_hours_after_index=float(
            (
                (index_session.session_end_ts - index_session.index_ts) / 3600000000.0
            ).max()
        ),
        potential_omitted_later_payment_overlap_users=int(
            possible_missing.user_key.nunique()
        ),
        stable_scope_basis="Session starting dates 2020-11-30 through 2021-01-25 inclusive; complete session retained",
        stable_sessions_ending_after_Jan25=int(
            stable.session_last_date.gt("2021-01-25").sum()
        ),
        stable_positive_events_after_Jan25=int(
            stable_purchases.event_timestamp.ge(obs_end).sum()
        ),
        stable_positive_sessions_lost_if_hard_event_cutoff=int(
            first_stable_purchase.ge(obs_end).sum()
        ),
        main_cohort_session_start_before_Nov30=int(
            index_session.session_date.lt("2020-11-30").sum()
        ),
        all_index_basket_payload_variants_unique=bool(
            joined.index_payload_variants.eq(1).all()
        ),
        deterministic_suspicious_basket_block_users=int(block.sum()),
        suspicious_basket_definition="exactly valid_sku_ids=[9188192] and observed_category_count=0 at first checkout timestamp",
        suspicious_block_purchase_7d_users=int(joined.loc[block, "y7"].sum()),
        interpretation="Original all-checkout ITT stays primary. Basket/time variants are retrospective sensitivity analyses; no bot or fraud identity is established.",
        sensitivity_results=metrics,
    )
    (OUT / "review_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
