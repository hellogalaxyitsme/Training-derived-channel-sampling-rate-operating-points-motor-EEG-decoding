"""Evaluation protocol: input construction, classical and deep decoders.

* Reduced inputs are formed by selecting channels and Fourier-resampling each
  trial to ``int(T * fs_target / fs_full)`` samples.
* Classical decoders (CSP, FBCSP, Riemannian MDM, tangent-space logistic
  regression) are fitted on the outer training partition and scored once on
  the outer test partition.
* Deep decoders (EEGNet, ATCNet-S) select their early-stopping checkpoint on
  an inner validation split drawn from the outer training partition only
  (stratified 80/20). Every configuration of a decoder within one outer fold
  uses the same inner split, seed, epoch cap and patience. Inputs are
  z-scored per channel with statistics of the inner-training part. Training
  uses AdamW (learning rate 1e-3, weight decay 1e-4) with a cosine schedule,
  mini-batches of 32, at most 100 epochs, and stops after 20 epochs without
  improvement of the inner-validation cross-entropy; the checkpoint with the
  lowest inner-validation loss is scored once on the outer test partition.
* Deep training uses fp32 arithmetic with TF32 matrix units; the long
  temporal convolution is evaluated with an FFT when the input has at least
  ``FFT_MIN_SAMPLES`` (1500) samples, which is numerically equivalent to
  direct convolution.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit

from .models import CSPDecoder, FBCSPDecoder, RiemannianMDM, get_model

FFT_MIN_SAMPLES = 1500
DEEP = ("eegnet", "atcnet", "eegnet_bp")
CLASSICAL = ("csp", "fbcsp", "riemann", "tslr")


def idx_hash(idx: np.ndarray) -> str:
    """Short SHA-1 digest of an index array (records the exact splits)."""
    return hashlib.sha1(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]


def bandpass(X: np.ndarray, flo: float, fhi: float, sfreq: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth band-pass (same design as the decoders)."""
    nyq = sfreq / 2.
    b, a = signal.butter(4, [flo / nyq, min(fhi / nyq, 0.99)], btype="band")
    return signal.filtfilt(b, a, X.astype(np.float64), axis=-1).astype(np.float32)


def resample_to(X: np.ndarray, fs_target: float, fs_full: float) -> np.ndarray:
    """Fourier resampling to int(T * fs_target / fs_full) samples."""
    if fs_target is None or abs(fs_target - fs_full) < 1e-9:
        return X
    new_T = max(4, int(X.shape[2] * fs_target / fs_full))
    return signal.resample(X, new_T, axis=-1).astype(np.float32)


def make_input(X: np.ndarray, channels: Sequence[int], fs_target: float,
               fs_full: float, band: Optional[tuple] = None) -> np.ndarray:
    """Select channels, optionally band-pass at the full rate, then resample."""
    Xs = X[:, np.asarray(channels), :]
    if band is not None:
        Xs = bandpass(Xs, band[0], band[1], fs_full)
    return resample_to(Xs, fs_target, fs_full)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "y_pred": np.asarray(y_pred, dtype=np.int8).tolist(),
    }


# ---------------------------------------------------------------------------
# Classical decoders
# ---------------------------------------------------------------------------
class TangentSpaceLR:
    """8-30 Hz band-pass, OAS covariances, Riemannian tangent space, and
    logistic regression with the inverse regularisation strength chosen by
    5-fold cross-validation on the training data only."""

    def __init__(self, sfreq: float, fmin: float = 8., fmax: float = 30.):
        self.sfreq, self.fmin, self.fmax = sfreq, fmin, fmax

    def _covs(self, X):
        from pyriemann.estimation import Covariances
        fmax = min(self.fmax, self.sfreq / 2. - 1.)
        Xf = bandpass(X, self.fmin, fmax, self.sfreq).astype(np.float64)
        return Covariances(estimator="oas").transform(Xf)

    def fit(self, X, y):
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.ts_ = TangentSpace(metric="riemann")
        F = self.ts_.fit_transform(self._covs(X))
        self.clf_ = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=5, max_iter=3000,
                                 n_jobs=1))
        self.clf_.fit(F, y)
        return self

    def predict(self, X):
        return self.clf_.predict(self.ts_.transform(self._covs(X)))


def eval_classical(decoder: str, X_tr, y_tr, X_te, y_te, sfreq: float,
                   n_classes: int) -> Dict:
    """Fit a classical decoder on (X_tr, y_tr) and score (X_te, y_te) once."""
    t0 = time.perf_counter()
    n_ch = X_tr.shape[1]
    fmax_u = min(30., sfreq / 2. - 1.)
    fmin_u = min(8., max(2., fmax_u - 22.))
    status = "ok"
    try:
        if decoder == "csp":
            m = CSPDecoder(n_components=min(3, max(1, n_ch // 2)),
                           fmin=fmin_u, fmax=fmax_u, sfreq=sfreq).fit(X_tr, y_tr)
        elif decoder == "fbcsp":
            m = FBCSPDecoder(n_components=min(2, max(1, n_ch // 2)),
                             sfreq=sfreq).fit(X_tr, y_tr)
        elif decoder == "riemann":
            m = RiemannianMDM(fmin=fmin_u, fmax=fmax_u, sfreq=sfreq).fit(X_tr, y_tr)
        elif decoder == "tslr":
            m = TangentSpaceLR(sfreq=sfreq).fit(X_tr, y_tr)
        else:
            raise ValueError(decoder)
        failed = bool(getattr(m, "_failed", False))
        y_pred = m.predict(X_te) if not failed else None
    except Exception as exc:  # recorded in the output, never silently dropped
        failed, y_pred, status = True, None, f"error: {exc!r}"[:200]
    if failed:
        # Fallback when the decoder cannot be fitted: predict the first
        # training class. The status field records that this happened.
        y_pred = np.full_like(y_te, fill_value=np.unique(y_tr)[0])
        status = status if status != "ok" else "decoder_failed_fallback"
    out = _metrics(y_te, y_pred)
    out.update(status=status, fit_s=float(time.perf_counter() - t0),
               n_components=int(min(3, max(1, n_ch // 2))) if decoder == "csp" else None)
    return out


# ---------------------------------------------------------------------------
# Deep decoders with inner-validation early stopping
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inner_split(y_tr: np.ndarray, seed: int, val_frac: float = 0.2):
    """Stratified inner train/validation split of the outer training partition."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    return next(sss.split(np.zeros(len(y_tr)), y_tr))


def train_eval_deep(arch: str, X_in, y_in, X_val, y_val, X_te, y_te,
                    sfreq: float, n_classes: int, seed: int,
                    max_epochs: int, patience: int, batch_size: int = 32,
                    lr: float = 1e-3, weight_decay: float = 1e-4,
                    device: str = "cuda") -> Dict:
    """Train on X_in, early-stop on (X_val, y_val), then score X_te once."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.perf_counter()
    set_seed(seed)
    mu = X_in.mean(axis=(0, 2), keepdims=True)
    sd = X_in.std(axis=(0, 2), keepdims=True) + 1e-8
    norm = lambda A: ((A - mu) / sd).astype(np.float32)  # noqa: E731
    Xi, Xv, Xt = norm(X_in), norm(X_val), norm(X_te)
    n_ch, n_T = Xi.shape[1], Xi.shape[2]
    use_fft = n_T >= FFT_MIN_SAMPLES
    model = get_model(arch, n_ch, n_T, n_classes, sfreq,
                      fft_temporal=use_fft).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator(); g.manual_seed(seed)
    Xi_t = torch.from_numpy(Xi); yi_t = torch.from_numpy(np.asarray(y_in, dtype=np.int64))
    Xv_t = torch.from_numpy(Xv).to(device); yv_t = torch.from_numpy(np.asarray(y_val, dtype=np.int64)).to(device)

    def _predict(Xa: torch.Tensor):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, Xa.shape[0], 32):          # evaluation batches of 32 trials
                outs.append(model(Xa[i:i + 32].to(device)).float())
        return torch.cat(outs)

    best = (-1., np.inf)      # (val acc, val loss); selection by lowest val loss
    best_state, best_epoch, bad, epochs_run = None, -1, 0, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(Xi_t.shape[0], generator=g)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            xb, yb = Xi_t[b].to(device, non_blocking=True), yi_t[b].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        sched.step()
        epochs_run = ep + 1
        logits = _predict(Xv_t)
        v_acc = (logits.argmax(1) == yv_t).float().mean().item()
        v_loss = crit(logits, yv_t).item()
        if v_loss < best[1] or (v_loss == best[1] and v_acc > best[0]):
            best, best_epoch, bad = (v_acc, v_loss), ep + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    y_pred = _predict(torch.from_numpy(Xt)).argmax(1).cpu().numpy()
    out = _metrics(y_te, y_pred)
    out.update(status="ok", best_epoch=int(best_epoch), epochs_run=int(epochs_run),
               val_acc=float(best[0]), val_loss=float(best[1]), n_params=n_params,
               fft_temporal=bool(use_fft), fit_s=float(time.perf_counter() - t0))
    del model, opt, Xv_t, yv_t
    torch.cuda.empty_cache()
    return out
