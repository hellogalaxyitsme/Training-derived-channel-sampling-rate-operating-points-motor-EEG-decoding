"""Real-data diagnostics of the spatial score on the first outer-training fold.

For the first N subjects of each cached dataset: covariance rank/conditioning of
the 8-30 Hz mixture covariance, the share of the score of Eq. (1)
(score="standard") explained by the loading term 1/2 log2(1 + eps/lambda_j),
and the overlap between the channels selected with that score and with the
contrast-only (Jensen-gap) score.

Usage: python scripts/real_data_estimator_diag.py [N_SUBJECTS]   (default 6)
Reads the caches in $EEGOP_CACHE (default ~/mne_data/eegop_cache) and writes
tables/real_data_estimator_diag.csv.
"""
import csv
import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eegop.estimators import (_bandpass, _class_cond_covs_from_trials,  # noqa: E402
                              _trial_covs, estimate_spatial_operating_point)

CACHE = os.environ.get("EEGOP_CACHE", os.path.expanduser("~/mne_data/eegop_cache"))
SPEC = {"cho2017": 250., "lee2019_mi": 1000., "schirrmeister2017": 500.}


def main():
    n_subj = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = []
    for ds, fs in SPEC.items():
        with open(os.path.join(CACHE, ds, f"{ds}_pooled.pkl"), "rb") as f:
            data = pickle.load(f)
        for subj in list(data)[:n_subj]:
            X, y = data[subj]["X"], np.asarray(data[subj]["y"])
            tr, _ = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
            Xt, yt = X[tr], y[tr]
            Xf = _bandpass(Xt, 8., 30., fs)
            tc, fin = _trial_covs(Xf)
            classes = np.unique(yt)
            pri = np.array([np.mean(yt == c) for c in classes])
            cc = _class_cond_covs_from_trials(tc, fin, yt, classes)
            mix = sum(pri[i] * cc[c] for i, c in enumerate(classes))
            lam = np.linalg.eigvalsh(mix)
            C = mix.shape[0]
            eps = 1e-4 * np.trace(mix) / C
            loading = 0.5 * np.log2(1 + eps / np.maximum(lam, 1e-300))
            std = estimate_spatial_operating_point(Xt, yt, sfreq=fs, n_perm=0, score="standard")
            jg = estimate_spatial_operating_point(Xt, yt, sfreq=fs, n_perm=0, score="jensen")
            r_l, r_j = std["r_star_effective"], jg["r_star_effective"]
            s_l = set(std["channel_ranking"][:r_l].tolist())
            s_j = set(jg["channel_ranking"][:r_j].tolist())
            top_l = set(std["channel_ranking"][:min(r_l, r_j)].tolist())
            top_j = set(jg["channel_ranking"][:min(r_l, r_j)].tolist())
            rows.append(dict(
                dataset=ds, subject=subj, C=C, n_train=len(yt),
                cond_number=float(lam.max() / max(lam.min(), 1e-300)),
                min_eig_over_mean=float(lam.min() / lam.mean()),
                numerical_rank=int(np.sum(lam > lam.max() * 1e-10)),
                eps_over_mean_eig=float(eps / lam.mean()),
                standard_score_total=float(np.sum(std["score_per_direction"])),
                loading_total=float(loading.sum()),
                loading_share=float(loading.sum() / np.sum(std["score_per_direction"])),
                r_standard=r_l, r_jensen=r_j,
                jaccard_selected=len(s_l & s_j) / len(s_l | s_j),
                overlap_equal_size=len(top_l & top_j) / max(1, min(r_l, r_j))))
            print(rows[-1], flush=True)
        del data
    out = os.path.join(ROOT, "tables", "real_data_estimator_diag.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
