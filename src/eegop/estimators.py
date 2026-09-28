"""Operating-point estimators computed from training data only.

* ``estimate_spatial_operating_point``  -- channel count r* and channel ranking
  from a per-direction spatial score of the class-conditional covariances.
* ``estimate_temporal_operating_point`` -- task bandwidth B_task from a
  discriminative spectrum and the sampling-rate operating point
  fs* = min(max(2 (B_task + 2/T), k C / T), fs) with k = 7.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import signal, stats


# -----------------------------------------------------------------------------
# Shared helper
# -----------------------------------------------------------------------------
def _bandpass(X: np.ndarray, flo: float, fhi: float,
              sfreq: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass; the filter order is reduced for very
    short epochs and the input is returned unfiltered if no valid design
    exists."""
    nyq = sfreq / 2.
    lo, hi = flo / nyq, min(fhi / nyq, 0.99)
    if lo >= hi:
        return X.astype(np.float64)
    n = X.shape[-1]; order = 4
    while order >= 1 and 6 * order >= n:
        order -= 1
    if order == 0:
        return X.astype(np.float64)
    b, a = signal.butter(order, [lo, hi], btype="band")
    return signal.filtfilt(b, a, X.astype(np.float64), axis=-1)


# =============================================================================
# Spatial operating point
# =============================================================================
def _class_cond_covs(Xf: np.ndarray, y: np.ndarray) -> Dict:
    """Class-conditional means of per-trial centred covariances."""
    _, n_ch, n_samp = Xf.shape
    covs: Dict = {}
    for c in np.unique(y):
        Xc = Xf[y == c]
        cc = [(xi - xi.mean(1, keepdims=True)) @
              (xi - xi.mean(1, keepdims=True)).T / n_samp
              for xi in Xc if np.isfinite(xi).all()]
        covs[c] = np.mean(cc, 0) if cc else np.eye(n_ch)
    return covs


def _trial_covs(Xf: np.ndarray):
    """Per-trial centred covariances, computed exactly as in _class_cond_covs."""
    n_samp = Xf.shape[2]
    finite = np.array([np.isfinite(xi).all() for xi in Xf])
    covs = np.stack([(xi - xi.mean(1, keepdims=True)) @
                     (xi - xi.mean(1, keepdims=True)).T / n_samp
                     if ok else np.zeros((Xf.shape[1], Xf.shape[1]))
                     for xi, ok in zip(Xf, finite)])
    return covs, finite


def _class_cond_covs_from_trials(covs: np.ndarray, finite: np.ndarray,
                                 y: np.ndarray, classes: np.ndarray) -> Dict:
    """Class means of precomputed trial covariances (same result as
    _class_cond_covs on the corresponding trials)."""
    out: Dict = {}
    n_ch = covs.shape[1]
    for c in classes:
        sel = (y == c) & finite
        out[c] = np.mean(covs[sel], 0) if sel.any() else np.eye(n_ch)
    return out


def _direction_scores(class_covs: Dict, priors: np.ndarray,
                      classes: np.ndarray, cov_mix: np.ndarray,
                      n_ch: int, reg: float, score: str = "standard"):
    """Per-direction score in the regularised whitening basis.

    Let W be the whitening matrix of the loaded mixture covariance
    cov_mix + reg * I and v_kj = w_j' Sigma_k w_j the projected variance of
    class k along whitening direction j.

    score="standard": m_j = -1/2 sum_k pi_k log2 v_kj (equation (1) of the
        paper). Because the whitening uses the loaded mixture,
        sum_k pi_k v_kj = lambda_j / (lambda_j + eps) < 1, so this score
        contains the additive loading term 1/2 log2(1 + eps/lambda_j), which
        is present even when all classes share one covariance.
    score="jensen": m_j = 1/2 [log2(sum_k pi_k v_kj) - sum_k pi_k log2 v_kj],
        the contrast-only (Jensen-gap) score, which is zero when the
        projected class variances agree.

    Returns the unsorted score vector (one entry per direction) and W.
    """
    C_reg = cov_mix + reg * np.eye(n_ch)
    ev, evc = np.linalg.eigh(C_reg)
    W = evc * (1. / np.sqrt(np.maximum(ev, 1e-12)))         # whitening columns
    m = np.zeros(n_ch)
    for j in range(n_ch):
        wj = W[:, j]
        v = [max(wj @ class_covs[c] @ wj, 1e-12) for c in classes]
        m[j] = -0.5 * sum(priors[i] * np.log2(v[i])
                          for i in range(len(classes)))
        if score == "jensen":
            m[j] += 0.5 * np.log2(sum(priors[i] * v[i]
                                      for i in range(len(classes))))
    return m, W


def estimate_spatial_operating_point(X: np.ndarray, y: np.ndarray,
                                     fmin: float = 8., fmax: float = 30.,
                                     sfreq: float = 250., n_perm: int = 200,
                                     alpha: float = 0.05,
                                     score: str = "standard",
                                     fast: bool = True) -> Dict:
    """Spatial operating-point estimator (channel count and ranking; T1 in the paper).

    Parameters
    ----------
    X : array (n_trials, n_channels, n_samples)
        Training trials only.
    y : array (n_trials,)
        Class labels.
    fmin, fmax : float
        Band-pass applied before covariance estimation (default 8-30 Hz).
    sfreq : float
        Sampling rate of X in Hz.
    n_perm : int
        Number of label permutations for the parallel-analysis count r*_PA
        (0 disables it; r*_PA is reported but not used by the final rule).
    alpha : float
        Level of the parallel-analysis threshold.
    score : {"standard", "jensen"}
        Per-direction spatial score, see ``_direction_scores``.
    fast : bool
        Reuse per-trial covariances across label permutations; results are
        identical to recomputing the class covariances per permutation.

    Returns
    -------
    dict with keys
        r_star            parallel-analysis count, floored at max(K-1, 1)
        r_star_PA         raw parallel-analysis count
        r_star_rel        smallest count reaching 95 % of the cumulative score
        r_star_PR         participation ratio of the positive scores
        r_star_effective  final channel count (r*_rel for K >= 4, r*_PR
                          otherwise, floored at K-1)
        score_per_direction  per-direction spatial scores, sorted descending
        pa_threshold      parallel-analysis threshold per rank
        channel_ranking   channels sorted by their summed squared whitening
                          loadings over the top r_star_effective directions
    """
    Xf   = _bandpass(X, fmin, fmax, sfreq)
    _, n_ch, _ = Xf.shape
    classes    = np.unique(y)
    priors     = np.array([np.mean(y == c) for c in classes])

    if fast:
        tcovs, finite = _trial_covs(Xf)
        class_covs = _class_cond_covs_from_trials(tcovs, finite, y, classes)
    else:
        class_covs = _class_cond_covs(Xf, y)
    cov_mix    = sum(priors[i] * class_covs[c]
                     for i, c in enumerate(classes))
    reg        = 1e-4 * np.trace(cov_mix) / n_ch

    m, W_white = _direction_scores(class_covs, priors, classes,
                                   cov_mix, n_ch, reg, score=score)
    sort_idx    = np.argsort(m)[::-1]
    m_sorted    = m[sort_idx]

    # Parallel-analysis null distribution from label permutations
    rng         = np.random.RandomState(42)
    null_sorted = np.zeros((n_perm, n_ch))
    for p in range(n_perm):
        yp = rng.permutation(y)
        if fast:
            cc = _class_cond_covs_from_trials(tcovs, finite, yp, classes)
        else:
            cc = _class_cond_covs(Xf, yp)
        cm = sum(priors[i] * cc[c] for i, c in enumerate(classes))
        m_p, _ = _direction_scores(cc, priors, classes, cm, n_ch, reg,
                                   score=score)
        null_sorted[p] = np.sort(m_p)[::-1]

    if n_perm > 0:
        pa_thresh = np.percentile(null_sorted, 100 * (1 - alpha), axis=0)
    else:   # parallel analysis disabled
        pa_thresh = np.full(n_ch, np.inf)
    r_PA      = int(np.sum(m_sorted > pa_thresh))

    # Relative cumulative-score criterion (95 % of the total)
    cum   = np.cumsum(m_sorted)
    tot   = cum[-1]
    r_rel = (int(np.argmax(cum / tot >= 0.95)) + 1) if tot > 1e-10 else n_ch

    r_star = max(max(r_PA, len(classes) - 1), 1)
    r_star = min(r_star, n_ch)

    # Participation ratio of the positive scores:
    # PR = (sum_j m_j)^2 / sum_j m_j^2, an effective number of directions
    # that does not require a binary cutoff.
    m_pos     = m_sorted[m_sorted > 0]
    r_pr      = (int(np.round(m_pos.sum() ** 2 / (m_pos ** 2).sum()))
                 if len(m_pos) > 0 and (m_pos ** 2).sum() > 1e-20
                 else n_ch)
    r_pr      = max(min(r_pr, n_ch), len(classes) - 1)

    # Class-dependent channel-count rule
    # K >= 4: r*_rel (95 % cumulative score).
    # K < 4 : r*_PR (participation ratio).
    # Floor: at least K-1 directions.
    K = len(classes)
    if K >= 4:
        r_star_effective = r_rel
    else:
        r_star_effective = r_pr
    r_star_effective = max(r_star_effective, K - 1)

    # Channel ranking used for selection (top r*_effective directions)
    W_disc  = W_white[:, sort_idx[:max(r_star_effective, 1)]]
    ch_rank = np.argsort(np.sum(W_disc ** 2, 1))[::-1]

    return dict(r_star=r_star, r_star_PA=r_PA, r_star_rel=r_rel,
                r_star_PR=r_pr, r_star_effective=r_star_effective,
                score_per_direction=m_sorted, pa_threshold=pa_thresh,
                channel_ranking=ch_rank)


# =============================================================================
# Temporal operating point
# =============================================================================
def estimate_temporal_operating_point(X: np.ndarray, y: np.ndarray,
                                      sfreq: float,
                                      fmax_analysis: float = 45.,
                                      btask_fmin: float = 4.,
                                      btask_fmax: float = 35.,
                                      T_epoch: float = 2.) -> Dict:
    """Sampling-rate operating point (T2 in the paper).

    Per-trial Welch PSDs are averaged over channels. The discriminative
    spectrum is D(f) = Var_k[P_k(f)] / (mean_k P_k(f) + noise)^2, where P_k is
    the class-mean PSD and the noise floor is the median PSD in 35-45 Hz.
    B_task is the larger of (i) the frequency at which the cumulative D(f)
    within [btask_fmin, btask_fmax] reaches 99 % of its total and (ii) the
    highest frequency in that range with a Bonferroni-significant one-way
    F-test across classes, clipped to [btask_fmin, btask_fmax].

        fs_nyquist = 2 (B_task + 2 / T_epoch)
        fs_cov     = 7 C / T_epoch
        fs_star    = min(max(fs_nyquist, fs_cov), sfreq)
    """
    n_trials, n_ch, n_samp = X.shape
    nperseg  = min(256, n_samp)
    noverlap = nperseg // 2
    classes  = np.unique(y)

    trial_psds = []
    for xi in X:
        f, p = signal.welch(xi, fs=sfreq, nperseg=nperseg,
                            noverlap=noverlap, axis=-1)
        trial_psds.append(p.mean(0))
    trial_psds = np.array(trial_psds)                       # (N, n_freqs)

    freq_mask = f <= fmax_analysis
    freqs_r   = f[freq_mask]
    psds_r    = trial_psds[:, freq_mask]

    # Noise-floor estimate from 35-45 Hz
    noise_mask = (freqs_r >= 35.) & (freqs_r <= 45.)
    noise_var  = float(np.median(psds_r[:, noise_mask])
                       if noise_mask.sum() >= 3
                       else np.percentile(psds_r, 5))
    noise_var  = max(noise_var, 1e-15)

    # Class-mean PSDs -> discriminative spectrum D(f)
    cls_psds = np.array([psds_r[y == c].mean(0) for c in classes])
    mean_psd = cls_psds.mean(0)
    D_tilde  = cls_psds.var(0) / np.maximum((mean_psd + noise_var) ** 2,
                                             1e-30)
    df       = float(freqs_r[1] - freqs_r[0]) if len(freqs_r) > 1 else 1.

    # Primary B_task: 99th percentile of cumulative D energy in
    # [btask_fmin, btask_fmax]
    band_mask = (freqs_r >= btask_fmin) & (freqs_r <= btask_fmax)
    D_band    = D_tilde.copy()
    D_band[~band_mask] = 0.
    total = np.sum(D_band) * df
    if total > 1e-12:
        cum_D = np.cumsum(D_band) * df / total
        B_task_energy = float(freqs_r[min(np.searchsorted(cum_D, 0.99),
                                          len(freqs_r) - 1)])
    else:
        B_task_energy = btask_fmin

    # Secondary: Bonferroni-corrected per-frequency F-test
    n_bins  = max(1, band_mask.sum())
    alpha_b = 0.05 / n_bins
    f_stats = np.zeros(len(freqs_r))
    p_vals  = np.ones(len(freqs_r))
    for fi in range(len(freqs_r)):
        groups = [psds_r[y == c, fi] for c in classes]
        if all(len(g) > 1 for g in groups):
            try:
                fv, pv      = stats.f_oneway(*groups)
                f_stats[fi] = fv if np.isfinite(fv) else 0.
                p_vals[fi]  = pv if np.isfinite(pv) else 1.
            except Exception:
                pass
    sig_band = (p_vals < alpha_b) & band_mask
    B_ftest  = (float(freqs_r[np.where(sig_band)[0][-1]])
                if sig_band.any() else 0.)

    B_task      = float(np.clip(max(B_task_energy, B_ftest),
                                btask_fmin, btask_fmax))
    B_effective = B_task + 2. / T_epoch
    fs_nyquist  = 2. * B_effective

    # Covariance-sample bound: covariance-based decoders estimate C x C
    # matrices from the samples of each epoch. The rate k C / T_epoch with
    # k = 7 keeps the per-epoch sample-to-dimension ratio at k.
    cov_factor  = 7.0
    fs_cov      = cov_factor * n_ch / T_epoch

    fs_star = min(max(fs_nyquist, fs_cov), sfreq)

    return dict(freqs=freqs_r,
                D_tilde=D_tilde / max(D_tilde.max(), 1e-12),
                B_task=B_task, B_task_ftest=B_ftest,
                B_effective=B_effective,
                fs_nyquist=fs_nyquist, fs_cov=fs_cov,
                fs_star=fs_star)
