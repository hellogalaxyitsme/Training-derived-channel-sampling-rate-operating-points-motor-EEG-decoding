"""Tests for the paired non-inferiority / equivalence helpers."""
import sys

import numpy as np
import pytest

from eegop.stats import add_holm, category, decisions, holm, paired_tests, subject_level

D = 0.03


def _dec(d, delta=D, **kw):
    r = paired_tests(d, delta, n_boot=0, **kw)
    return r, decisions(r)


def _sample(mu, sd, n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    x = (x - x.mean()) / x.std(ddof=1)
    return mu + sd * x


def test_near_zero_is_equivalent_and_noninferior():
    r, d = _dec(_sample(0.002, 0.02, 50))
    assert d["equivalent"] and d["noninferior"] and not d["harm"]
    assert category(d) == "Equivalent"


def test_well_below_minus_delta_is_noninferior_and_better_not_equivalent():
    r, d = _dec(_sample(-0.08, 0.03, 40))
    assert d["noninferior"] and d["better_by_delta"] and not d["equivalent"]
    assert category(d) == "Reduced better by > margin"


def test_well_above_delta_is_harm():
    r, d = _dec(_sample(0.09, 0.03, 40))
    assert d["harm"] and not d["noninferior"]
    assert category(d) == "Loss > margin"


def test_mean_just_above_delta_with_wide_uncertainty_is_inconclusive():
    r, d = _dec(_sample(0.035, 0.08, 14))
    assert not d["harm"] and not d["noninferior"]
    assert category(d) == "Inconclusive"


def test_mean_inside_band_with_wide_uncertainty_is_not_equivalent():
    r, d = _dec(_sample(0.005, 0.10, 10))
    assert not d["equivalent"]
    assert category(d) in ("Inconclusive", "Non-inferior")


def test_exact_boundary_is_not_evidence():
    r, d = _dec(_sample(D, 0.02, 30))
    assert abs(r["p_NI"] - 0.5) < 1e-12 and abs(r["p_harm"] - 0.5) < 1e-12
    assert not d["noninferior"] and not d["harm"]


def test_zero_variance_and_all_equal():
    r, d = _dec([0.0] * 10)
    assert r["status"] == "zero_variance"
    assert d["equivalent"] and d["noninferior"] and not d["harm"]
    r, d = _dec([0.05] * 10)
    assert d["harm"] and not d["noninferior"]
    r, d = _dec([D] * 10)          # exactly on the margin: neither side
    assert not d["noninferior"] and not d["harm"]


def test_missing_values_and_small_n():
    r = paired_tests([0.01, np.nan, 0.02, np.inf, 0.0], D, n_boot=0)
    assert r["n"] == 3 and r["n_missing"] == 2
    r = paired_tests([0.01], D, n_boot=0)
    assert r["status"] == "insufficient_n" and np.isnan(r["p_NI"])


def test_harm_and_noninferiority_are_mutually_exclusive():
    rng = np.random.default_rng(1)
    for _ in range(300):
        x = rng.normal(rng.uniform(-0.1, 0.1), rng.uniform(0.001, 0.1), rng.integers(3, 60))
        r, d = _dec(x)
        assert not (d["harm"] and d["noninferior"])
        # equivalence implies non-inferiority (same hypotheses, same alpha)
        if d["equivalent"]:
            assert d["noninferior"]


def test_unit_conversion_invariance():
    x = _sample(0.012, 0.03, 25)
    a = paired_tests(x, D, n_boot=0)
    b = paired_tests(100 * x, 100 * D, n_boot=0)
    for k in ("p_NI", "p_TOST", "p_harm", "p_better_d", "p_better_0"):
        assert abs(a[k] - b[k]) < 1e-12


def test_holm_is_never_more_permissive_and_matches_reference():
    p = [0.01, 0.04, 0.03, 0.005, np.nan]
    adj = holm(p)
    assert np.isnan(adj[-1])
    assert np.all(adj[:4] >= np.asarray(p[:4]))
    # reference values: sorted .005,.01,.03,.04 -> 4*.005=.02, 3*.01=.03, 2*.03=.06, 1*.04=.06
    assert np.allclose(adj[:4], [0.03, 0.06, 0.06, 0.02])


def test_holm_family_and_equivalence_implies_noninferiority_after_adjustment():
    rows = []
    for i, mu in enumerate([0.0, 0.01, -0.005, 0.02]):
        r = paired_tests(_sample(mu, 0.02, 40, seed=i), D, n_boot=0)
        r["rule"] = "T2"
        rows.append(r)
    add_holm(rows, ["rule"])
    for r in rows:
        assert r["holm_family_size"] == 4
        dh = decisions(r, suffix="_holm")
        if dh["equivalent"]:
            assert dh["noninferior"]
        for h in ("p_NI", "p_TOST", "p_harm"):
            assert r[h + "_holm"] >= r[h] - 1e-15


def test_tost_matches_90pct_interval_rule():
    rng = np.random.default_rng(3)
    for _ in range(200):
        x = rng.normal(rng.uniform(-0.04, 0.04), 0.03, 20)
        r = paired_tests(x, D, n_boot=0)
        inside = (r["ci90_lo"] > -D) and (r["ci90_hi"] < D)
        assert inside == (r["p_TOST"] < 0.05)


def test_seed_duplication_does_not_change_subject_level_inference():
    rng = np.random.default_rng(4)
    per_subject = {s: [v] for s, v in enumerate(rng.normal(0.01, 0.03, 20))}
    dup = {s: v * 3 for s, v in per_subject.items()}       # identical seed repeated
    a = paired_tests(list(subject_level(per_subject).values()), D, n_boot=0)
    b = paired_tests(list(subject_level(dup).values()), D, n_boot=0)
    assert a["n"] == b["n"] == 20
    assert abs(a["p_NI"] - b["p_NI"]) < 1e-12 and abs(a["se"] - b["se"]) < 1e-15


def test_significant_loss_within_margin_uncertainty_is_inconclusive():
    # Mean above the margin, significantly worse than zero but not
    # significantly worse than the margin: the comparison is inconclusive.
    x = _sample(0.035, 0.03, 20)
    r, d = _dec(x)
    assert r["p_worse_0"] < 0.05 and r["p_harm"] > 0.05
    assert category(d) == "Inconclusive"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
