"""Compare trigger populations on identical dates, before selecting the main experiment."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from cohort import read_table, prepare, utc_us
from validation import power_table, validate_sample_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--purchases", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--quality-flags")
    ap.add_argument("--expected-sessions", type=int)
    ap.add_argument("--start", default="2020-11-30T00:00:00Z")
    ap.add_argument("--end", default="2021-01-19T00:00:00Z")
    ap.add_argument("--observation-end", default="2021-01-26T00:00:00Z")
    args = ap.parse_args()
    s = read_table(args.sessions)
    p = read_table(args.purchases)
    if args.expected_sessions is not None and len(s) != args.expected_sessions:
        raise ValueError("Session export is incomplete")
    if s.duplicated(["user_pseudo_id", "ga_session_id"]).any():
        raise ValueError("Duplicate full-session keys")
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    rep = Path(args.reports)
    rep.mkdir(parents=True, exist_ok=True)
    sources = [dict(path=args.sessions), dict(path=args.purchases)]
    candidate_collapsed = None
    if args.quality_flags:
        flags = pd.read_csv(args.quality_flags)[["event_id", "candidate_repeat_excess"]]
        linked = p.merge(flags, on="event_id", validate="one_to_one", how="left")
        if linked.candidate_repeat_excess.isna().any():
            raise ValueError("Missing transaction quality flags")
        candidate_collapsed = linked[~linked.candidate_repeat_excess.astype(bool)].drop(
            columns=["candidate_repeat_excess"]
        )
        sources.append(dict(path=args.quality_flags))
    summaries = []
    weekly = []
    power_frames = []
    powerchecks = []
    for trigger in ["first_view_ts", "first_checkout_ts"]:
        saved = {}
        for definition in ["positive_merchandise", "raw"]:
            b, a = prepare(
                s,
                p,
                trigger,
                utc_us(args.start),
                utc_us(args.end),
                utc_us(args.observation_end),
                purchase_definition=definition,
            )
            name = trigger.removesuffix("_ts") + "_" + definition
            file = dest / (name + ".csv")
            b.to_csv(file, index=False)
            a.update(inputs=sources, output=dict(path=str(file)))
            file.with_suffix(".audit.json").write_text(
                json.dumps(a, ensure_ascii=False, indent=2)
            )
            saved[definition] = b
            summaries.append(
                dict(
                    trigger=trigger,
                    purchase_definition=definition,
                    users=len(b),
                    purchase_7d_users=int(b.y7.sum()),
                    purchase_7d_rate=float(b.y7.mean()),
                    purchase_24h_rate=float(b.y24.mean()),
                    pre_history_coverage=float(b.pre_history_present.mean()),
                    median_delay_hours=float(b.first_purchase_delay_hours.median()),
                    same_timestamp_purchase_users=int(b.purchase_timestamp_tie.sum()),
                    baseline_path=str(file),
                )
            )
            if definition == "positive_merchandise":
                delta = float(np.clip(0.15 * b.y7.mean(), 0.003, 0.05))
                table = power_table(b, delta)
                table.insert(0, "trigger", trigger)
                power_frames.append(table)
                pc = validate_sample_size(table, np.random.default_rng(20261006))
                pc.insert(0, "trigger", trigger)
                powerchecks.append(pc)
                dates = pd.to_datetime(b.index_date)
                weeks = dates.dt.to_period("W-SUN")
                for week, count in b.groupby(weeks).size().items():
                    complete = week.start_time >= pd.Timestamp(args.start).tz_localize(
                        None
                    ).normalize() and week.end_time < pd.Timestamp(
                        args.end
                    ).tz_localize(
                        None
                    )
                    weekly.append(
                        dict(
                            trigger=trigger,
                            week_start=str(week.start_time.date()),
                            week_end=str(week.end_time.date()),
                            first_eligible_users=int(count),
                            complete_week=bool(complete),
                            daily_users=float(count / 7) if complete else np.nan,
                        )
                    )
        pair = saved["positive_merchandise"][["user_key", "y7"]].merge(
            saved["raw"][["user_key", "y7"]],
            on="user_key",
            suffixes=("_proxy", "_raw"),
            validate="one_to_one",
        )
        assert len(pair) == len(saved["raw"]) == len(saved["positive_merchandise"])
        assert pair.y7_raw.ge(pair.y7_proxy).all()
        if candidate_collapsed is not None:
            collapsed, a = prepare(
                s,
                candidate_collapsed,
                trigger,
                utc_us(args.start),
                utc_us(args.end),
                utc_us(args.observation_end),
                purchase_definition="positive_merchandise",
            )
            check = saved["positive_merchandise"][["user_key", "y7"]].merge(
                collapsed[["user_key", "y7"]],
                on="user_key",
                suffixes=("_full", "_collapsed"),
                validate="one_to_one",
            )
            assert len(check) == len(collapsed)
            summaries[-2]["candidate_repeat_collapse_changed_y7_users"] = int(
                check.y7_full.ne(check.y7_collapsed).sum()
            )
            summaries[-2]["candidate_repeat_collapse_y7_rate"] = float(
                collapsed.y7.mean()
            )
    pd.DataFrame(summaries).to_csv(rep / "baseline_comparison.csv", index=False)
    pd.DataFrame(weekly).to_csv(rep / "baseline_weekly_traffic.csv", index=False)
    pd.concat(power_frames, ignore_index=True).to_csv(
        rep / "baseline_power_comparison.csv", index=False
    )
    pd.concat(powerchecks, ignore_index=True).to_csv(
        rep / "baseline_binomial_power_validation.csv", index=False
    )
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
