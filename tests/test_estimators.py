"""Synthetic checks of the spatial and temporal operating-point estimators."""
import numpy as np
import pytest

from eegop.estimators import (_direction_scores, estimate_spatial_operating_point,
                              estimate_temporal_operating_point)


def _rand_spd(C, rng, cond=100.):
    Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
    ev = np.logspace(0, np.log10(cond), C)[::-1]
    return (Q * ev) @ Q.T


def _simulate(covs, n_trials, T, rng):
    """Zero-mean Gaussian trials with one covariance per class."""
    X, y = [], []
    for k, S in enumerate(covs):
        L = np.linalg.cholesky(S)
        for _ in range(n_trials):
            X.append(L @ rng.standard_normal((S.shape[0], T)))
            y.append(k)
    return np.asarray(X), np.asarray(y)


def _contrast_data(seed=0, C=12, n_trials=60, T=300):
    """Two classes whose covariances differ on channels 0-3 only."""
    rng = np.random.default_rng(seed)
    Sa = _rand_spd(C, rng, cond=1e2)
    Sb = Sa.copy()
    Sb[:4, :4] *= 1.6
    return _simulate([Sa, Sb], n_trials, T, rng)


# ---------------------------------------------------------------------------
# (1) equal class covariances
# ---------------------------------------------------------------------------
def test_equal_population_covariances():
    rng = np.random.default_rng(0)
    C = 10
    S = _rand_spd(C, rng, cond=1e3)
    classes = np.arange(2)
    priors = np.array([0.5, 0.5])
    reg = 1e-4 * np.trace(S) / C
    m_jg, _ = _direction_scores({0: S, 1: S}, priors, classes, S, C, reg, score="jensen")
    m_std, _ = _direction_scores({0: S, 1: S}, priors, classes, S, C, reg, score="standard")
    assert np.max(np.abs(m_jg)) < 1e-10
    lam = np.linalg.eigvalsh(S)
    pred = 0.5 * np.log2(1. + reg / lam)
    np.testing.assert_allclose(np.sort(m_std), np.sort(pred), rtol=1e-6, atol=1e-10)
    assert np.all(m_std > 0)


def test_identical_class_trials_give_zero_jensen_score():
    rng = np.random.default_rng(1)
    C, n, T = 8, 30, 250
    X0, _ = _simulate([_rand_spd(C, rng)], n, T, rng)
    X = np.concatenate([X0, X0])
    y = np.repeat([0, 1], n)
    out = estimate_spatial_operating_point(X, y, sfreq=250., n_perm=0, score="jensen")
    assert np.max(np.abs(out["score_per_direction"])) < 1e-9
    std = estimate_spatial_operating_point(X, y, sfreq=250., n_perm=0, score="standard")
    assert np.all(std["score_per_direction"] > 0)


# ---------------------------------------------------------------------------
# (2) global amplitude scaling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("score", ["standard", "jensen"])
def test_global_scaling_invariance(score):
    X, y = _contrast_data(seed=2)
    base = estimate_spatial_operating_point(X, y, sfreq=250., n_perm=5, score=score)
    scaled = estimate_spatial_operating_point(7.3 * X, y, sfreq=250., n_perm=5, score=score)
    for k in ("r_star", "r_star_PA", "r_star_rel", "r_star_PR", "r_star_effective"):
        assert base[k] == scaled[k], k
    np.testing.assert_array_equal(base["channel_ranking"], scaled["channel_ranking"])
    np.testing.assert_allclose(base["score_per_direction"], scaled["score_per_direction"],
                               rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# (3) channel permutation equivariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("score", ["standard", "jensen"])
def test_channel_permutation_equivariance(score):
    X, y = _contrast_data(seed=3)
    C = X.shape[1]
    perm = np.random.default_rng(4).permutation(C)
    base = estimate_spatial_operating_point(X, y, sfreq=250., n_perm=0, score=score)
    permd = estimate_spatial_operating_point(X[:, perm], y, sfreq=250., n_perm=0, score=score)
    r = base["r_star_effective"]
    assert permd["r_star_effective"] == r
    selected_base = set(base["channel_ranking"][:r].tolist())
    selected_perm = set(perm[permd["channel_ranking"][:r]].tolist())
    assert selected_base == selected_perm


# ---------------------------------------------------------------------------
# (4) sampling-rate rule
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("C,T_epoch,fs", [(8, 2., 250.), (16, 1., 250.), (64, 1., 250.),
                                          (4, 4., 500.)])
def test_temporal_rule(C, T_epoch, fs):
    rng = np.random.default_rng(5)
    n = int(T_epoch * fs)
    X = rng.standard_normal((40, C, n))
    y = np.repeat([0, 1], 20)
    out = estimate_temporal_operating_point(X, y, fs, T_epoch=T_epoch)
    nyq = 2. * (out["B_task"] + 2. / T_epoch)
    assert out["fs_nyquist"] == pytest.approx(nyq)
    assert out["fs_cov"] == pytest.approx(7. * C / T_epoch)
    assert out["fs_star"] == pytest.approx(min(max(nyq, 7. * C / T_epoch), fs))
    assert 4. <= out["B_task"] <= 35.
