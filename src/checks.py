"""Deterministic integrity checks; fixtures are entirely synthetic, not GA4 evidence."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from cohort import prepare, utc_us, DAY, qualifying_purchase
from methods import (
    difference,
    weighted_cluster_difference,
    causal_estimates,
    crossfit_nuisance,
)


def check_time_and_leakage():
    t = utc_us("2020-12-01T00:00:00Z")
    sessions = pd.DataFrame(
        [
            dict(
                user_pseudo_id="u",
                session_start_ts=t - DAY,
                session_end_ts=t - DAY + 10,
                first_view_ts=t - DAY + 1,
                view_events=3,
            ),
            dict(
                user_pseudo_id="u",
                session_start_ts=t - 10,
                session_end_ts=t + DAY,
                first_view_ts=t,
                view_events=99,
            ),
            dict(
                user_pseudo_id="u",
                session_start_ts=t + DAY,
                session_end_ts=t + DAY + 10,
                first_view_ts=t + DAY + 1,
                view_events=88,
            ),
            dict(
                user_pseudo_id="v",
                session_start_ts=t,
                session_end_ts=t + 20,
                first_view_ts=t + 1,
                view_events=1,
            ),
        ]
    )
    purchase = pd.DataFrame(
        [
            dict(user_pseudo_id="u", event_timestamp=t),
            dict(user_pseudo_id="u", event_timestamp=t + DAY),
            dict(user_pseudo_id="u", event_timestamp=t + 7 * DAY),
            dict(user_pseudo_id="v", event_timestamp=t + 1 + 7 * DAY),
        ]
    )
    b, a = prepare(sessions, purchase, "first_view_ts", t, t + 2 * DAY, t + 9 * DAY)
    u = b.set_index("user_key").loc["u"]
    v = b.set_index("user_key").loc["v"]
    assert (
        u.y24 == 0
        and u.y7 == 1
        and (u.purchases_7d == 1)
        and (u.purchase_timestamp_tie == 1)
    )
    assert (
        u.pre_sessions_14d == 1 and u.pre_views_14d == 3
    ), "Index/future session leaked into baseline features"
    assert v.y7 == 0 and b.user_key.is_unique


def check_estimation_units():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    a = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    est = difference(y, a)
    wrong, cluster = weighted_cluster_difference(y, a, np.full(len(y), 5))
    assert np.isclose(est.estimate, wrong.estimate) and np.isclose(est.se, cluster.se)
    assert (
        wrong.se < cluster.se / 2
    ), "Repeating user rows must not create independent information"


def check_economic_proxy():
    p = pd.DataFrame(
        dict(
            purchase_revenue_usd=[10, 0, 10, 10, 10],
            items_json=[
                '[{"item_id":"sku-1","quantity":1}]',
                '[{"item_id":"sku-1","quantity":1}]',
                '[{"item_id":null,"quantity":1}]',
                '[{"item_id":"<Other>","quantity":1}]',
                '[{"item_id":"sku-1","quantity":0},{"item_id":null,"quantity":1}]',
            ],
        )
    )
    assert qualifying_purchase(p, "positive_merchandise").tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert qualifying_purchase(p, "raw").all()


def check_orthogonal_score():
    rng = np.random.default_rng(18)
    n = 30000
    x = rng.normal(size=(n, 2))
    e = 1 / (1 + np.exp(-0.8 * x[:, 0]))
    p0 = 0.1 + 0.1 / (1 + np.exp(-x[:, 0]))
    p1 = p0 + 0.04
    a = rng.binomial(1, e)
    y = rng.binomial(1, np.where(a, p1, p0))
    est = causal_estimates(a, y, e, p0, p1)["aipw"]
    assert (
        abs(est.estimate - 0.04) < 4 * est.se
    ), "Oracle AIPW failed known-truth sanity check"
    assert np.isfinite(est.se) and est.se > 0
    ep, m0, m1 = crossfit_nuisance(x[:1000], x[:1000], a[:1000], y[:1000], 42)
    assert (
        np.all((ep > 0) & (ep < 1))
        and np.all((m0 >= 0) & (m0 <= 1))
        and np.all((m1 >= 0) & (m1 <= 1))
    )


if __name__ == "__main__":
    check_time_and_leakage()
    check_estimation_units()
    check_economic_proxy()
    check_orthogonal_score()
    print(
        "PASS: strict window boundaries, pre-index features, unique-user clustering, and known-truth AIPW sanity checks (synthetic fixtures only)."
    )
