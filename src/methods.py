"""Small estimators with explicit estimands; no treatment effects are inferred from GA4."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.special import expit
from scipy.stats import norm, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


@dataclass
class Estimate:
    estimate: float
    se: float = float("nan")

    def as_dict(self, truth: float = 0.0) -> dict:
        lo, hi = (self.estimate - 1.96 * self.se, self.estimate + 1.96 * self.se)
        return dict(
            estimate=self.estimate,
            se=self.se,
            truth=truth,
            bias=self.estimate - truth,
            ci_low=lo,
            ci_high=hi,
            covered=float(lo <= truth <= hi) if np.isfinite(self.se) else np.nan,
            reject_zero=(
                float(abs(self.estimate) > 1.96 * self.se)
                if np.isfinite(self.se)
                else np.nan
            ),
            p_value=(
                float(2 * norm.sf(abs(self.estimate / self.se)))
                if self.se > 0
                else np.nan
            ),
        )


def difference(y, a) -> Estimate:
    y, a = (np.asarray(y, float), np.asarray(a, bool))
    y1, y0 = (y[a], y[~a])
    if min(len(y1), len(y0)) < 2:
        return Estimate(np.nan)
    return Estimate(
        float(y1.mean() - y0.mean()),
        float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))),
    )


def srm(a, probability=0.5) -> float:
    a = np.asarray(a)
    return float(binomtest(int(a.sum()), len(a), probability).pvalue)


def weighted_cluster_difference(y, a, repetitions) -> tuple[Estimate, Estimate]:
    """One response repeated w times: row-IID SE versus user-cluster ratio SE.

    This estimates a repetition-weighted effect. Used only at the sharp null, where
    its target and the unweighted user target are both zero.
    """
    y, a, w = (np.asarray(y, float), np.asarray(a, bool), np.asarray(repetitions, int))
    means = []
    row_variance = []
    cluster_variance = []
    for group in [a, ~a]:
        yy, ww = (y[group], w[group])
        n = float(ww.sum())
        m = float(np.dot(ww, yy) / n)
        means.append(m)
        row_variance.append(float(np.dot(ww, (yy - m) ** 2) / (n - 1) / n))
        cluster_variance.append(
            float(np.sum((ww * (yy - m)) ** 2) / n**2 * len(yy) / (len(yy) - 1))
        )
    effect = means[0] - means[1]
    return (
        Estimate(effect, np.sqrt(sum(row_variance))),
        Estimate(effect, np.sqrt(sum(cluster_variance))),
    )


def fit_logistic(x, y):
    x, y = (np.asarray(x, float), np.asarray(y, int))
    if len(np.unique(y)) < 2:
        p = float((y.sum() + 0.5) / (len(y) + 1))
        return lambda z: np.full(len(z), p)
    model = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=300, tol=1e-07)
    model.fit(x, y)
    return lambda z: model.predict_proba(z)[:, 1]


def crossfit_nuisance(x_prop, x_outcome, a, y, seed):
    """Every predicted row is excluded from both nuisance training datasets."""
    n = len(a)
    e = np.empty(n)
    m0 = np.empty(n)
    m1 = np.empty(n)
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=int(seed))
    for tr, te in splitter.split(x_prop, a):
        e[te] = fit_logistic(x_prop[tr], a[tr])(x_prop[te])
        for treatment, m in [(0, m0), (1, m1)]:
            arm = tr[a[tr] == treatment]
            m[te] = fit_logistic(x_outcome[arm], y[arm])(x_outcome[te])
    return (e, m0, m1)


def causal_estimates(a, y, e, m0, m1, clip=0.01):
    """Only AIPW has an influence-score CI; OR/IPW are point-estimate benchmarks.

    AIPW CI uses the usual sample SD of the cross-fitted score / sqrt(n).
    Coverage is evaluated by simulation; it is NOT guaranteed under arbitrary
    misspecification, clipping, missing confounders, or positivity violations.
    """
    a, y = (np.asarray(a, float), np.asarray(y, float))
    ec = np.clip(e, clip, 1 - clip)
    ipw = a * y / ec - (1 - a) * y / (1 - ec)
    score = m1 - m0 + a * (y - m1) / ec - (1 - a) * (y - m0) / (1 - ec)
    return {
        "naive": difference(y, a),
        "outcome_regression": Estimate(float(np.mean(m1 - m0))),
        "ipw": Estimate(float(ipw.mean())),
        "aipw": Estimate(
            float(score.mean()), float(score.std(ddof=1) / np.sqrt(len(score)))
        ),
    }


def overlap_diagnostics(a, e, x, clip=0.01):
    ec = np.clip(e, clip, 1 - clip)
    w = np.where(a == 1, 1 / ec, 1 / (1 - ec))
    out = {
        "propensity_min": float(np.min(e)),
        "propensity_max": float(np.max(e)),
        "clip_fraction": float(np.mean((e < clip) | (e > 1 - clip))),
    }
    smds = []
    for t in (0, 1):
        wt = w[a == t]
        out[f"ess_{t}"] = float(wt.sum() ** 2 / np.dot(wt, wt))
        out[f"ess_fraction_{t}"] = out[f"ess_{t}"] / int((a == t).sum())
    for j in range(x.shape[1]):
        s = np.std(x[:, j], ddof=1)
        if s > 1e-08:
            m = [np.average(x[a == t, j], weights=w[a == t]) for t in (0, 1)]
            smds.append(abs(m[1] - m[0]) / s)
    out["max_weighted_abs_smd"] = float(max(smds, default=0))
    return out


def summarize_runs(rows, keys):
    import pandas as pd

    df = pd.DataFrame(rows)
    summaries = []
    for key, g in df.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(keys, key))
        r.update(
            repetitions=len(g),
            truth=float(g.truth.mean()),
            mean_estimate=float(g.estimate.mean()),
            bias=float(g.bias.mean()),
            rmse=float(np.sqrt(np.mean(g.bias**2))),
            empirical_sd=float(g.estimate.std(ddof=1)),
            mean_se=float(g.se.mean()),
        )
        for col in ["covered", "reject_zero"]:
            vals = g[col].dropna()
            rate = float(vals.mean()) if len(vals) else np.nan
            r[col + "_rate"] = rate
            r[col + "_mcse"] = (
                float(np.sqrt(rate * (1 - rate) / len(vals))) if len(vals) else np.nan
            )
            if len(vals):
                size = len(vals)
                denom = 1 + 1.96**2 / size
                center = (rate + 1.96**2 / (2 * size)) / denom
                half = (
                    1.96
                    * np.sqrt(rate * (1 - rate) / size + 1.96**2 / (4 * size**2))
                    / denom
                )
                r[col + "_mc_wilson_low"] = float(center - half)
                r[col + "_mc_wilson_high"] = float(center + half)
            else:
                r[col + "_mc_wilson_low"] = r[col + "_mc_wilson_high"] = np.nan
        for col in ["srm_assignment", "srm_observed"]:
            if col in g:
                v = g[col].dropna()
                r[col + "_flag_rate"] = float((v < 0.001).mean()) if len(v) else np.nan
        summaries.append(r)
    return (df, pd.DataFrame(summaries))
