"""Restore portable CSV inputs and check the grains used by the analysis."""

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "session_facts": 360129,
    "transaction_events": 5692,
    "item_daily": 241261,
    "event_semantics_profile": 28691,
    "checkout_index_facts": 11106,
    "checkout_local_price": 11106,
    "price_coverage": 3,
    "event_inventory": 4143,
    "retention_cohorts": 450,
    "daily_metrics": 92,
}


def main():
    dest = ROOT / "data/processed"
    dest.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "data/inputs").glob("*.csv.gz")):
        with gzip.open(source, "rb") as src, (dest / source.stem).open("wb") as out:
            shutil.copyfileobj(src, out)
    frames = {
        name: pd.read_csv(
            dest / f"{name}.csv",
            dtype={
                "user_pseudo_id": "string",
                "ga_session_id": "string",
                "event_id": "string",
            },
            low_memory=False,
        )
        for name in EXPECTED
    }
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    for name, count in EXPECTED.items():
        check(name + "_rows", len(frames[name]) == count, len(frames[name]))
    s, t, inv, items = [
        frames[k]
        for k in [
            "session_facts",
            "transaction_events",
            "event_inventory",
            "item_daily",
        ]
    ]
    check(
        "unique_session_key",
        not s.duplicated(["user_pseudo_id", "ga_session_id"]).any(),
        len(s),
    )
    check(
        "session_boundaries",
        s.session_end_ts.ge(s.session_start_ts).all(),
        "end >= start",
    )
    check("event_ids", t.event_id.is_unique, len(t))
    check("source_events", inv.event_count.sum() == 4295584, inv.event_count.sum())
    check(
        "session_event_reconciliation",
        s.event_count.sum() == inv.event_count.sum(),
        s.event_count.sum(),
    )
    check(
        "purchase_rows",
        t.identical_raw_row_count.sum()
        == inv.loc[inv.event_name.eq("purchase"), "event_count"].sum(),
        len(t),
    )
    check(
        "item_revenue",
        np.isclose(items.item_purchase_revenue_usd.sum(), t.item_revenue_sum_usd.sum()),
        items.item_purchase_revenue_usd.sum(),
    )
    check(
        "basket_keys",
        not frames["checkout_index_facts"]
        .duplicated(["user_pseudo_id", "ga_session_id"])
        .any(),
        11106,
    )
    r = frames["retention_cohorts"]
    check(
        "retention_bounds",
        r.active_size.between(0, r.cohort_size).all(),
        "0 <= retained <= cohort",
    )
    check(
        "retention_d0",
        r.loc[r.delta_day.eq(0), "active_size"]
        .eq(r.loc[r.delta_day.eq(0), "cohort_size"])
        .all(),
        "D0 = cohort",
    )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(checks).to_csv(out / "data_checks.csv", index=False)
    failed = [x["check"] for x in checks if not x["passed"]]
    if failed:
        raise ValueError("Data checks failed: " + ", ".join(failed))
    print(f"{len(checks)} data checks passed")


if __name__ == "__main__":
    main()
