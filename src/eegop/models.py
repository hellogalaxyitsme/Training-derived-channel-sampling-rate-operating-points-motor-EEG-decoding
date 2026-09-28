"""Decoders used in the evaluation.

Deep decoders (EEGNet, ATCNet-S) accept ``(B, C, T)`` tensors
(batch x channels x samples). Classical decoders (CSP, FBCSP, Riemannian MDM)
follow the scikit-learn ``fit``/``predict`` convention on ``(N, C, T)`` arrays.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


class FFTTemporalConv(nn.Module):
    """Temporal convolution computed as an FFT cross-correlation.

    Numerically equivalent (up to floating-point rounding) to
    ``nn.Conv2d(1, n_filters, (1, kern), padding=(0, kern // 2), bias=False)``
    applied to a ``(B, 1, C, T)`` tensor with odd ``kern``. The weight tensor
    has the same shape as the Conv2d weight, so initialisation and parameter
    counts are unchanged. The FFT path is used because the temporal kernel of
    the EEGNet/ATCNet variants spans half of the epoch, which makes direct
    convolution the dominant training cost at high sampling rates.
    """

    def __init__(self, n_filters: int, kern: int):
        super().__init__()
        if kern % 2 != 1:
            raise ValueError("kern must be odd")
        self.kern = kern
        self.weight = nn.Parameter(torch.empty(n_filters, 1, 1, kern))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T) -> (B, F, C, T)
        pad = self.kern // 2
        T = x.shape[-1]
        xp = nn.functional.pad(x.float(), (pad, pad))
        L = xp.shape[-1]
        n_fft = 1 << (L - 1).bit_length()
        Xf = torch.fft.rfft(xp, n=n_fft)                       # (B,1,C,Nf)
        Wf = torch.fft.rfft(self.weight.float()[:, 0, 0, :], n=n_fft)  # (F,Nf)
        Yf = Xf * torch.conj(Wf)[None, :, None, :]             # (B,F,C,Nf)
        return torch.fft.irfft(Yf, n=n_fft)[..., :T]


def _temporal_conv(n_filters: int, kern: int, pad: int, fft: bool) -> nn.Module:
    if fft:
        return FFTTemporalConv(n_filters, kern)
    return nn.Conv2d(1, n_filters, (1, kern), padding=(0, pad), bias=False)


# -----------------------------------------------------------------------------
# EEGNet  (Lawhern et al., J. Neural Eng., 2018)
# -----------------------------------------------------------------------------
class EEGNet(nn.Module):
    """EEGNet with a temporal kernel spanning half of the epoch.

    Temporal convolution -> depthwise spatial convolution -> separable
    convolution -> linear classifier. Kernel and pooling sizes are derived
    from the number of input samples so that the same architecture can be
    applied at every sampling rate.
    """

    def __init__(self, n_ch: int, n_samp: int, n_cls: int,
                 F1: int = 8, D: int = 2, F2: int = 16, dropout: float = 0.5,
                 fft_temporal: bool = False):
        super().__init__()
        kern = _odd(min(max(n_samp // 2, 3), n_samp - 1))
        pad  = kern // 2
        p1   = min(4, max(1, n_samp // 8))
        T1   = max(1, n_samp // p1)
        p2   = min(8, max(1, T1 // 2))
        sk   = _odd(max(T1 // 4, 3))
        sp   = sk // 2

        self.b1 = nn.Sequential(
            _temporal_conv(F1, kern, pad, fft_temporal),
            nn.BatchNorm2d(F1))
        self.dw = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, p1)), nn.Dropout(dropout))
        self.sep = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, sk), padding=(0, sp), bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, p2)), nn.Dropout(dropout))

        with torch.no_grad():
            n_flat = self.sep(self.dw(self.b1(
                torch.zeros(1, 1, n_ch, n_samp)))).numel()
        self.fc = nn.Linear(n_flat, n_cls)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, FFTTemporalConv)):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.sep(self.dw(self.b1(x.unsqueeze(1)))).flatten(1))


# -----------------------------------------------------------------------------
# ATCNet-S  (simplified; after Altaheri et al., IEEE Trans. Ind. Informat., 2023)
# -----------------------------------------------------------------------------
class ATCNet(nn.Module):
    """ATCNet-S: a simplified attention-temporal-convolution network
    following Altaheri et al. (2023).

    EEGNet-style convolutional stem -> one multi-head self-attention block
    with residual connection and layer normalisation -> dilated
    depthwise-separable temporal convolution stack with a 1x1 residual
    projection -> global average pooling -> linear classifier. Unlike the
    original ATCNet, it does not use the sliding-window ensemble.
    """

    def __init__(self, n_ch: int, n_samp: int, n_cls: int,
                 F1: int = 16, D: int = 2, n_heads: int = 2,
                 tcn_ch: int = 32, tcn_kern: int = 5,
                 tcn_depth: int = 2, dropout: float = 0.3,
                 fft_temporal: bool = False):
        super().__init__()
        F2   = F1 * D
        kern = _odd(min(max(n_samp // 2, 3), n_samp - 1))
        pad  = kern // 2
        p1   = min(8, max(1, n_samp // 8))

        # EEGNet-style convolutional stem
        self.stem = nn.Sequential(
            _temporal_conv(F1, kern, pad, fft_temporal),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, p1)), nn.Dropout(dropout))

        # Multi-head self-attention
        nh = n_heads
        while F2 % nh != 0 and nh > 1:
            nh -= 1
        self.attn      = nn.MultiheadAttention(F2, nh, dropout=dropout,
                                               batch_first=True)
        self.attn_norm = nn.LayerNorm(F2)
        self.attn_drop = nn.Dropout(dropout)

        # Temporal convolution stack (depthwise-separable, dilated)
        tcn_layers = []
        for d in [2 ** i for i in range(tcn_depth)]:
            tcn_layers += [
                nn.Conv1d(F2, F2, tcn_kern, dilation=d,
                          padding="same", groups=F2, bias=False),
                nn.Conv1d(F2, tcn_ch, 1, bias=False),
                nn.BatchNorm1d(tcn_ch), nn.ELU(), nn.Dropout(dropout)]
            F2 = tcn_ch
        self.tcn     = nn.Sequential(*tcn_layers)
        self.tcn_res = nn.Conv1d(F1 * D, tcn_ch, 1, bias=False)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(tcn_ch, n_cls))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x.unsqueeze(1)).squeeze(2)            # (B, F2, T')
        ht   = h.permute(0, 2, 1)
        a, _ = self.attn(ht, ht, ht)
        h    = self.attn_norm(ht + self.attn_drop(a)).permute(0, 2, 1)
        h    = self.tcn(h) + self.tcn_res(h)                # (B, tcn_ch, T')
        return self.head(h)


# -----------------------------------------------------------------------------
# CSP + shrinkage LDA  (multi-class via per-class log-variance features)
# -----------------------------------------------------------------------------
class CSPDecoder:
    """Common spatial patterns with shrinkage LDA.

    For each class, filters are obtained from the eigendecomposition of the
    class covariance in the whitened space of the pooled covariance (one
    versus rest); the first and last ``n_components`` filters per class are
    kept and log-variance features are classified with shrinkage LDA.
    """

    def __init__(self, n_components: int = 3, fmin: float = 8.,
                 fmax: float = 30., sfreq: float = 250.):
        self.n_components = n_components
        self.fmin, self.fmax, self.sfreq = fmin, fmax, sfreq
        self.W_ = self.scaler_ = self.lda_ = None
        self._failed = False

    def _bp(self, X: np.ndarray) -> np.ndarray:
        nyq = self.sfreq / 2.
        lo, hi = self.fmin / nyq, min(self.fmax / nyq, 0.99)
        if lo >= hi:
            return X.astype(np.float64)
        n = X.shape[-1]; order = 4
        while order >= 1 and 6 * order >= n:
            order -= 1
        if order == 0:
            return X.astype(np.float64)
        b, a = signal.butter(order, [lo, hi], btype="band")
        return signal.filtfilt(b, a, X.astype(np.float64), axis=-1)

    def _cov(self, X: np.ndarray) -> Optional[np.ndarray]:
        covs = []
        for xi in X:
            if not np.isfinite(xi).all():
                continue
            c  = xi @ xi.T; tr = np.trace(c)
            if tr < 1e-20:
                continue
            covs.append(c / tr)
        return np.mean(covs, 0) if covs else None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPDecoder":
        self._failed = False
        Xf    = self._bp(X)
        C_all = self._cov(Xf)
        if C_all is None:
            self._failed = True
            return self
        n_ch  = Xf.shape[1]
        reg   = 1e-4
        ev, evc = np.linalg.eigh(C_all + reg * np.eye(n_ch))
        W_w   = evc * (1. / np.sqrt(np.maximum(ev, 1e-12)))
        filt  = []
        for c in np.unique(y):
            Cc = self._cov(Xf[y == c])
            Cc = C_all if Cc is None else Cc
            Cw = W_w.T @ Cc @ W_w
            Cw = (Cw + Cw.T) / 2
            try:
                ev_c, ev_v = np.linalg.eigh(Cw)
            except Exception:
                continue
            V = W_w @ ev_v[:, np.argsort(ev_c)[::-1]]
            m = min(self.n_components, max(1, n_ch // 2))
            filt += [V[:, :m].T, V[:, -m:].T]
        if not filt:
            self._failed = True
            return self
        self.W_      = np.vstack(filt)
        feats        = self._feats(Xf)
        self.scaler_ = StandardScaler()
        self.lda_    = LinearDiscriminantAnalysis(solver="lsqr",
                                                  shrinkage="auto")
        try:
            self.lda_.fit(self.scaler_.fit_transform(feats), y)
        except Exception:
            self._failed = True
        return self

    def _feats(self, Xf: np.ndarray) -> np.ndarray:
        proj = np.einsum("fc,nct->nft", self.W_, Xf)
        return np.log(np.clip(np.var(proj, -1), 1e-10, None))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lda_.predict(
            self.scaler_.transform(self._feats(self._bp(X))))


# -----------------------------------------------------------------------------
# Filter-bank CSP + shrinkage LDA
# -----------------------------------------------------------------------------
class FBCSPDecoder:
    """Filter-bank CSP using fold-local sub-band CSP features.

    The implementation follows the classical motor-imagery FBCSP pattern:
    band-pass each trial into several mu/beta sub-bands, fit CSP in each band
    on the training split only, concatenate log-variance CSP features, then
    classify with shrinkage LDA.
    """

    def __init__(self, n_components: int = 2,
                 bands: Optional[list[tuple[float, float]]] = None,
                 sfreq: float = 250.):
        self.n_components = n_components
        self.bands = bands or [
            (8., 12.), (12., 16.), (16., 20.), (20., 24.), (24., 30.)
        ]
        self.sfreq = sfreq
        self.csp_bank_: list[CSPDecoder] = []
        self.scaler_ = None
        self.lda_ = None
        self._failed = False

    def _valid_bands(self) -> list[tuple[float, float]]:
        nyq = self.sfreq / 2.
        valid = []
        for flo, fhi in self.bands:
            hi = min(float(fhi), nyq - 1.)
            lo = float(flo)
            if hi > lo and hi > 2.:
                valid.append((lo, hi))
        return valid

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FBCSPDecoder":
        self._failed = False
        self.csp_bank_ = []
        feats = []
        n_ch = X.shape[1]
        m = min(self.n_components, max(1, n_ch // 2))
        for flo, fhi in self._valid_bands():
            csp = CSPDecoder(
                n_components=m, fmin=flo, fmax=fhi, sfreq=self.sfreq)
            csp.fit(X, y)
            if csp._failed or csp.W_ is None:
                continue
            try:
                feats.append(csp._feats(csp._bp(X)))
            except Exception:
                continue
            self.csp_bank_.append(csp)
        if not feats:
            self._failed = True
            return self
        X_feat = np.concatenate(feats, axis=1)
        if not np.isfinite(X_feat).all():
            self._failed = True
            return self
        self.scaler_ = StandardScaler()
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr",
                                               shrinkage="auto")
        try:
            self.lda_.fit(self.scaler_.fit_transform(X_feat), y)
        except Exception:
            self._failed = True
        return self

    def _feats(self, X: np.ndarray) -> np.ndarray:
        feats = [csp._feats(csp._bp(X)) for csp in self.csp_bank_]
        return np.concatenate(feats, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.lda_.predict(self.scaler_.transform(self._feats(X)))


# -----------------------------------------------------------------------------
# Riemannian MDM  (Barachant et al., IEEE Trans. Biomed. Eng., 2012)
# -----------------------------------------------------------------------------
class RiemannianMDM:
    """Minimum Distance to Mean on the SPD manifold of EEG covariances.

    Uses pyriemann for the Riemannian geometry operations and follows the
    scikit-learn fit/predict convention. Trials are band-passed to 8-30 Hz
    (mu and beta) before the covariance matrices are computed.
    """

    def __init__(self, fmin: float = 8., fmax: float = 30.,
                 sfreq: float = 250.):
        self.fmin, self.fmax, self.sfreq = fmin, fmax, sfreq
        self._failed = False

    def _bp(self, X: np.ndarray) -> np.ndarray:
        nyq = self.sfreq / 2.
        lo, hi = self.fmin / nyq, min(self.fmax / nyq, 0.99)
        if lo >= hi:
            return X.astype(np.float64)
        n = X.shape[-1]; order = 4
        while order >= 1 and 6 * order >= n:
            order -= 1
        if order == 0:
            return X.astype(np.float64)
        b, a = signal.butter(order, [lo, hi], btype="band")
        return signal.filtfilt(b, a, X.astype(np.float64), axis=-1)

    def _covs(self, X: np.ndarray) -> np.ndarray:
        """Regularised sample covariance per trial, shape (N, C, C)."""
        Xf = self._bp(X)
        N, C, T = Xf.shape
        covs = np.zeros((N, C, C), dtype=np.float64)
        for i in range(N):
            xi = Xf[i] - Xf[i].mean(1, keepdims=True)
            c = xi @ xi.T / T
            c += 1e-6 * np.trace(c) / C * np.eye(C)
            covs[i] = c
        return covs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannianMDM":
        self._failed = False
        try:
            from pyriemann.classification import MDM
            covs = self._covs(X)
            self.mdm_ = MDM(metric="riemann")
            self.mdm_.fit(covs, y)
        except Exception:
            self._failed = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mdm_.predict(self._covs(X))


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------
def get_model(name: str, n_ch: int, n_samp: int,
              n_cls: int, sfreq: float,
              fft_temporal: bool = False) -> nn.Module:
    """Build a deep decoder by name ("eegnet" or "atcnet").

    ``sfreq`` is accepted for interface compatibility; both architectures
    derive their kernel sizes from ``n_samp``.
    """
    if name == "eegnet":
        return EEGNet(n_ch, n_samp, n_cls, fft_temporal=fft_temporal)
    if name == "atcnet":
        return ATCNet(n_ch, n_samp, n_cls, fft_temporal=fft_temporal)
    raise ValueError(f"Unknown deep decoder: {name}")
