"""Paired subject-level inference for full-versus-reduced input comparisons.

Definitions
-----------
For independent subjects i = 1..n let d_i = acc_full_i - acc_reduced_i, in
accuracy units (fractions). Positive d means the reduced input performed
worse. With margin delta > 0, SE = sd(d)/sqrt(n) and nu = n - 1:

    p_NI       = F_t((mean - delta)/SE; nu)       H1: E[d] <  delta (non-inferior)
    p_TO_lower = 1 - F_t((mean + delta)/SE; nu)   H1: E[d] > -delta
    p_TOST     = max(p_NI, p_TO_lower)            H1: |E[d]| < delta (equivalent)
    p_harm     = 1 - F_t((mean - delta)/SE; nu)   H1: E[d] >  delta (loss beyond margin)
    p_better_d = F_t((mean + delta)/SE; nu)       H1: E[d] < -delta (reduced better by > delta)
    p_better_0 = F_t(mean/SE; nu)                 H1: E[d] <  0     (reduced better)
    p_worse_0  = 1 - F_t(mean/SE; nu)             H1: E[d] >  0     (reduced worse)

Holm adjustment is applied to the hypothesis p-values within a declared
family, and decisions are derived from the adjusted p-values. The unadjusted
TOST decision at alpha corresponds to the two-sided (1 - 2 alpha) t interval
lying inside (-delta, delta).

Degenerate cases: non-finite values are removed and counted; n < 2 yields
no inference (all p = nan, status "insufficient_n"); zero variance is resolved
deterministically from the sign of (mean -/+ delta) and flagged
"zero_variance". A mean exactly on a boundary is not evidence for either side
(p = 0.5 for the corresponding one-sided test when SE > 0; p = 1 when SE = 0).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy import stats

HYPOTHESES = ("p_NI", "p_TOST", "p_harm", "p_better_d", "p_better_0", "p_worse_0")


TOL = 1e-12   # numerical tolerance for zero variance and exact boundaries


def _zero_var_p(stat_num: float, lower_tail: bool) -> float:
    """Degenerate one-sided p-value when SE = 0: 0 if the sign strictly
    supports H1 (beyond TOL), otherwise 1."""
    if lower_tail:
        return 0.0 if stat_num < -TOL else 1.0
    return 0.0 if stat_num > TOL else 1.0


def paired_tests(d: Iterable[float], delta: float, alpha: float = 0.05,
                 n_boot: int = 10000, boot_seed: int = 0) -> Dict:
    """All one-sided/TOST p-values, t intervals and effect size for paired deltas."""
    arr = np.asarray(list(d), dtype=float)
    n_missing = int(np.sum(~np.isfinite(arr)))
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    out: Dict = dict(n=n, n_missing=n_missing, delta=float(delta), alpha=float(alpha))
    if n < 2:
        out.update(status="insufficient_n", mean=float(arr.mean()) if n else np.nan)
        for h in HYPOTHESES + ("p_TO_lower",):
            out[h] = np.nan
        return out
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    nu = n - 1
    out.update(mean=mean, sd=sd, se=se, df=nu, median=float(np.median(arr)),
               n_reduced_better=int(np.sum(arr < 0)), n_reduced_worse=int(np.sum(arr > 0)),
               n_tied=int(np.sum(arr == 0)))
    if sd > TOL:
        F = lambda x: float(stats.t.cdf(x, nu))  # noqa: E731
        S = lambda x: float(stats.t.sf(x, nu))   # noqa: E731
        out["p_NI"] = F((mean - delta) / se)
        out["p_TO_lower"] = S((mean + delta) / se)
        out["p_harm"] = S((mean - delta) / se)
        out["p_better_d"] = F((mean + delta) / se)
        out["p_better_0"] = F(mean / se)
        out["p_worse_0"] = S(mean / se)
        out["status"] = "ok"
        tq90, tq95 = stats.t.ppf(0.95, nu), stats.t.ppf(0.975, nu)
        out["ci90_lo"], out["ci90_hi"] = mean - tq90 * se, mean + tq90 * se
        out["ci95_lo"], out["ci95_hi"] = mean - tq95 * se, mean + tq95 * se
        out["upper95_one_sided"] = mean + tq90 * se
        out["dz"] = mean / sd
    else:
        out["p_NI"] = _zero_var_p(mean - delta, lower_tail=True)
        out["p_TO_lower"] = _zero_var_p(mean + delta, lower_tail=False)
        out["p_harm"] = _zero_var_p(mean - delta, lower_tail=False)
        out["p_better_d"] = _zero_var_p(mean + delta, lower_tail=True)
        out["p_better_0"] = _zero_var_p(mean, lower_tail=True)
        out["p_worse_0"] = _zero_var_p(mean, lower_tail=False)
        out["status"] = "zero_variance"
        for k in ("ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi", "upper95_one_sided"):
            out[k] = mean
        out["dz"] = np.nan
    out["p_TOST"] = max(out["p_NI"], out["p_TO_lower"])
    if n_boot:
        rng = np.random.default_rng(boot_seed)
        bm = arr[rng.integers(0, n, size=(n_boot, n))].mean(1)
        out["boot95_lo"], out["boot95_hi"] = (float(np.percentile(bm, 2.5)),
                                              float(np.percentile(bm, 97.5)))
        # subject-resampling estimate of Pr(mean < delta) as a robustness summary
        out["boot_frac_below_delta"] = float(np.mean(bm < delta))
    return out


def holm(pvals: Iterable[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (nan entries are ignored and kept nan)."""
    p = np.asarray(list(pvals), dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m == 0:
        return adj
    order = ok[np.argsort(p[ok], kind="mergesort")]
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def decisions(row: Dict, alpha: float = 0.05, suffix: str = "") -> Dict[str, bool]:
    """Boolean decisions from (optionally adjusted) p-values; suffix='_holm'."""
    g = lambda k: row.get(k + suffix, np.nan)  # noqa: E731
    lt = lambda v: bool(np.isfinite(v) and v < alpha)  # noqa: E731
    return dict(equivalent=lt(g("p_TOST")), noninferior=lt(g("p_NI")),
                harm=lt(g("p_harm")), better_by_delta=lt(g("p_better_d")),
                better_than_zero=lt(g("p_better_0")), worse_than_zero=lt(g("p_worse_0")))


def category(dec: Dict[str, bool]) -> str:
    """Mutually exclusive display category derived from the decisions.

    Non-inferiority includes the equivalent and 'better by > delta' cases;
    the display splits them so that each comparison has one label.
    """
    if dec["harm"]:
        return "Loss > margin"
    if dec["noninferior"] and dec["better_by_delta"]:
        return "Reduced better by > margin"
    if dec["equivalent"]:
        return "Equivalent"
    if dec["noninferior"]:
        return "Non-inferior"
    return "Inconclusive"


def add_holm(rows: List[Dict], family_keys: Iterable[str],
             hypotheses: Iterable[str] = HYPOTHESES) -> List[Dict]:
    """Holm-adjust each hypothesis p-value within families defined by
    identical values of family_keys (e.g. ('rule',))."""
    fam: Dict = {}
    for i, r in enumerate(rows):
        fam.setdefault(tuple(r[k] for k in family_keys), []).append(i)
    for idxs in fam.values():
        for h in hypotheses:
            adj = holm([rows[i][h] for i in idxs])
            for j, i in enumerate(idxs):
                rows[i][h + "_holm"] = float(adj[j])
                rows[i]["holm_family_size"] = len(idxs)
    return rows


def subject_level(values: Dict, agg: str = "mean") -> Dict:
    """Collapse repeated measurements per subject.

    values: {subject_id: sequence of per-seed (or per-fold) deltas}. Returns
    {subject_id: aggregated delta}. Repetitions of the same subject never
    increase the inferential n.
    """
    f = np.mean if agg == "mean" else np.median
    out = {}
    for s, v in values.items():
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        out[s] = float(f(a)) if a.size else np.nan
    return out
