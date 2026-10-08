"""Construct one row per eligible user, with pre-index features and mature outcomes."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

DAY = 86400000000


def read_table(path):
    path = Path(path)
    if path.suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "user_pseudo_id": "string",
            "user_key": "string",
            "ga_session_id": "string",
            "event_id": "string",
        },
    )


def utc_us(s):
    return (
        int(pd.Timestamp(s, tz="UTC").value // 1000)
        if pd.Timestamp(s).tz is None
        else int(pd.Timestamp(s).value // 1000)
    )


def qualifying_purchase(p, definition):
    if definition == "raw":
        return np.ones(len(p), dtype=bool)
    if "purchase_revenue_usd" not in p or "items_json" not in p:
        raise ValueError(
            "Positive-merchandise purchase requires purchase_revenue_usd and items_json"
        )
    revenue = pd.to_numeric(p.purchase_revenue_usd, errors="coerce")
    invalid_ids = {"", "(not set)", "<other>", "none", "null", "nan"}

    def valid_item(payload):
        items = json.loads(payload) if isinstance(payload, str) else payload
        if items is None:
            return False
        for item in items:
            item_id = str(item.get("item_id", "")).strip().lower()
            quantity = pd.to_numeric(item.get("quantity"), errors="coerce")
            if item_id not in invalid_ids and pd.notna(quantity) and (quantity > 0):
                return True
        return False

    return (revenue.gt(0) & p.items_json.map(valid_item)).to_numpy()


def prepare(
    s,
    p,
    trigger,
    start,
    end,
    observation_end,
    history_days=14,
    outcome_days=7,
    purchase_definition="raw",
    eligible_device=None,
):
    required = ["user_pseudo_id", "session_start_ts", "session_end_ts", trigger]
    missing = [x for x in required if x not in s]
    if missing:
        raise ValueError(f"Missing session fields: {missing}")
    if not {"user_pseudo_id", "event_timestamp"}.issubset(p):
        raise ValueError("Purchase events require user_pseudo_id,event_timestamp")
    s = s.copy()
    p = p.copy()
    for col in [
        x for x in s if x.endswith("_ts") or x in ["view_events", "ga_session_number"]
    ]:
        s[col] = pd.to_numeric(s[col], errors="coerce")
    p["event_timestamp"] = pd.to_numeric(p["event_timestamp"], errors="coerce")
    s = s[s.user_pseudo_id.notna() & s.user_pseudo_id.ne("")]
    p = p[p.user_pseudo_id.notna() & p.event_timestamp.notna()]
    if "event_name" in p:
        p = p[p.event_name.eq("purchase")]
    raw_purchase_rows = len(p)
    p = p.loc[qualifying_purchase(p, purchase_definition)]
    candidate = s[s[trigger].ge(start) & s[trigger].lt(end)].sort_values(
        [trigger, "session_start_ts"], kind="stable"
    )
    if eligible_device is not None:
        if "device_category" not in candidate:
            raise ValueError("device_category required for device eligibility")
        candidate = candidate[candidate.device_category.eq(eligible_device)]
    index = candidate.drop_duplicates("user_pseudo_id", keep="first").copy()
    immature = index[trigger] + outcome_days * DAY > observation_end
    excluded_immature = int(immature.sum())
    index = index[~immature]
    baseline = pd.DataFrame(
        dict(
            user_key=index.user_pseudo_id.astype(str).to_numpy(),
            index_ts=index[trigger].astype("int64").to_numpy(),
        )
    )
    baseline["index_date"] = pd.to_datetime(
        baseline.index_ts, unit="us", utc=True
    ).dt.strftime("%Y-%m-%d")
    for col in ["device_category", "first_user_source", "first_user_medium", "country"]:
        baseline[col] = (
            index[col].fillna("unknown").astype(str).to_numpy()
            if col in index
            else "unknown"
        )
    baseline["ga_session_number"] = (
        index.ga_session_number.to_numpy() if "ga_session_number" in index else np.nan
    )
    link = baseline[["user_key", "index_ts"]].rename(
        columns={"user_key": "user_pseudo_id"}
    )
    history = s[["user_pseudo_id", "session_start_ts", "session_end_ts"]].copy()
    history["view_events"] = (
        s.view_events.fillna(0).to_numpy() if "view_events" in s else 0.0
    )
    history["user_pseudo_id"] = history.user_pseudo_id.astype(str)
    history = history.merge(
        link, on="user_pseudo_id", how="inner", validate="many_to_one"
    )
    prior = history[
        history.session_end_ts.lt(history.index_ts)
        & history.session_start_ts.ge(history.index_ts - history_days * DAY)
    ]
    pre_session_count = prior.groupby("user_pseudo_id").size()
    pre_views = prior.groupby("user_pseudo_id").view_events.sum()
    baseline["pre_sessions_14d"] = (
        baseline.user_key.map(pre_session_count).fillna(0).astype(int)
    )
    baseline["pre_views_14d"] = baseline.user_key.map(pre_views).fillna(0).astype(float)
    baseline["pre_history_present"] = baseline.pre_sessions_14d.gt(0).astype(int)
    future_sessions = history[
        history.session_start_ts.ge(history.index_ts)
        & history.session_start_ts.lt(history.index_ts + outcome_days * DAY)
    ]
    baseline["post_index_session_starts_7d"] = (
        baseline.user_key.map(future_sessions.groupby("user_pseudo_id").size())
        .fillna(0)
        .astype(int)
    )
    pp = p[["user_pseudo_id", "event_timestamp"]].copy()
    pp["user_pseudo_id"] = pp.user_pseudo_id.astype(str)
    pp = pp.merge(link, on="user_pseudo_id", how="inner", validate="many_to_one")
    after = pp.event_timestamp.gt(pp.index_ts)
    within7 = pp[after & pp.event_timestamp.lt(pp.index_ts + outcome_days * DAY)]
    within1 = pp[after & pp.event_timestamp.lt(pp.index_ts + DAY)]
    before = pp[
        pp.event_timestamp.lt(pp.index_ts)
        & pp.event_timestamp.ge(pp.index_ts - history_days * DAY)
    ]
    ties = pp[pp.event_timestamp.eq(pp.index_ts)]
    for col, frame in [
        ("purchases_7d", within7),
        ("pre_purchases_14d", before),
        ("purchase_timestamp_tie", ties),
    ]:
        baseline[col] = (
            baseline.user_key.map(frame.groupby("user_pseudo_id").size())
            .fillna(0)
            .astype(int)
        )
    baseline["purchase_timestamp_tie"] = baseline.purchase_timestamp_tie.gt(0).astype(
        int
    )
    baseline["y7"] = baseline.purchases_7d.gt(0).astype(int)
    baseline["y24"] = baseline.user_key.isin(within1.user_pseudo_id).astype(int)
    first_purchase = baseline.user_key.map(
        within7.groupby("user_pseudo_id").event_timestamp.min()
    )
    baseline["first_purchase_delay_hours"] = (
        first_purchase - baseline.index_ts
    ) / 3600000000
    if len(baseline) == 0:
        raise ValueError(
            "No mature eligible users; check bounds and trigger timestamps."
        )
    assert baseline.user_key.is_unique
    assert baseline.y24.le(baseline.y7).all()
    assert (baseline.index_ts + outcome_days * DAY <= observation_end).all()
    audit = dict(
        status="computed_from_real_session_and_purchase_events",
        trigger=trigger,
        target="Unique anonymous browser IDs at first eligible trigger during enrollment window",
        outcome=f"At least one recorded {purchase_definition} purchase timestamp strictly after index and before index+{outcome_days}d",
        purchase_definition=purchase_definition,
        eligible_device=eligible_device,
        enrollment_start_us=start,
        enrollment_end_exclusive_us=end,
        observation_end_exclusive_us=observation_end,
        history_days=history_days,
        outcome_days=outcome_days,
        session_rows=len(s),
        purchase_event_rows=len(p),
        raw_purchase_event_rows=raw_purchase_rows,
        excluded_purchase_event_rows=raw_purchase_rows - len(p),
        users=len(baseline),
        excluded_immature_users=excluded_immature,
        purchase_7d_users=int(baseline.y7.sum()),
        purchase_7d_rate=float(baseline.y7.mean()),
        purchase_24h_rate=float(baseline.y24.mean()),
        purchase_tie_users=int(baseline.purchase_timestamp_tie.sum()),
        pre_history_coverage=float(baseline.pre_history_present.mean()),
        notes=[
            "purchase outcome counts events without relying on potentially obfuscated transaction IDs",
            "7d counts are NOT number of orders; duplicate purchase events do not change binary y7",
            "pre-session features require session_end_ts<index and session_start_ts>=index-14d",
            "zero prior observed history does not prove that the person is a new customer",
            "observed pseudo-ID outcomes do not identify cross-device customers or capture missing GA4 purchases",
        ],
    )
    return (baseline, audit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--purchases", required=True)
    ap.add_argument("--trigger", default="first_view_ts")
    ap.add_argument("--start", default="2020-11-30T00:00:00Z")
    ap.add_argument("--end", default="2021-01-19T00:00:00Z")
    ap.add_argument("--observation-end", default="2021-01-26T00:00:00Z")
    ap.add_argument(
        "--purchase-definition",
        choices=["raw", "positive_merchandise"],
        default="positive_merchandise",
    )
    ap.add_argument("--device")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    baseline, audit = prepare(
        read_table(args.sessions),
        read_table(args.purchases),
        args.trigger,
        utc_us(args.start),
        utc_us(args.end),
        utc_us(args.observation_end),
        purchase_definition=args.purchase_definition,
        eligible_device=args.device,
    )
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(dest, index=False)
    audit["inputs"] = [dict(path=args.sessions), dict(path=args.purchases)]
    audit["output"] = dict(path=str(dest))
    dest.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
