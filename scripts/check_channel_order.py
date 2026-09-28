"""Scale-invariant check that cached arrays share MOABB's channel order.

For subject 1, each cached channel is correlated with every channel of a fresh
MOABB epochs load; the channel order is confirmed if the best match of every
cached channel is the channel with the same index.

Usage: python scripts/check_channel_order.py {cho2017|lee2019_mi|schirrmeister2017}
Reads the cache in $EEGOP_CACHE (default ~/mne_data/eegop_cache).
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import mne  # noqa: E402
from moabb.paradigms import MotorImagery  # noqa: E402

mne.set_log_level("ERROR")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
CACHE = os.environ.get("EEGOP_CACHE", os.path.expanduser("~/mne_data/eegop_cache"))


def main():
    name = sys.argv[1]
    if name == "cho2017":
        from moabb.datasets import Cho2017
        ds = Cho2017()
        kw = dict(n_classes=2, tmin=0., tmax=3., resample=250.)
    elif name == "lee2019_mi":
        from eegop.lee_sessions import Lee2019MISessions
        ds = Lee2019MISessions()
        kw = dict(n_classes=2, tmin=0., tmax=4., resample=1000.)
    elif name == "schirrmeister2017":
        from moabb.datasets import Schirrmeister2017
        ds = Schirrmeister2017()
        kw = dict(n_classes=4, tmin=0., tmax=4., resample=500.)
    else:
        raise SystemExit(f"unknown dataset {name}")
    ep, y, meta = MotorImagery(fmin=0.5, fmax=100., **kw).get_data(dataset=ds, subjects=[1],
                                                                   return_epochs=True)
    E = ep.get_data(copy=False)
    if name == "lee2019_mi":
        # the within-session cache holds the second recording session
        E = E[meta["session"].values == "2"]
    with open(os.path.join(CACHE, name, f"{name}_pooled.pkl"), "rb") as f:
        c = pickle.load(f)[1]["X"]
    n = min(len(E), len(c))
    A = E[:n].transpose(1, 0, 2).reshape(E.shape[1], -1).astype(np.float64)
    B = c[:n].transpose(1, 0, 2).reshape(c.shape[1], -1).astype(np.float64)
    A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-12)
    B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-12)
    R = B @ A.T / A.shape[1]
    best = R.argmax(1)
    ok = bool(np.all(best == np.arange(len(best))))
    scale = float(np.median(np.std(c[:n], axis=(0, 2)) / (np.std(E[:n], axis=(0, 2)) + 1e-30)))
    print(json.dumps(dict(dataset=name, channel_order_confirmed=ok, min_diag_corr=float(np.diag(R).min()),
                          median_scale_cache_over_epochs=scale, n_channels=int(c.shape[1]),
                          ch_names_epochs=list(ep.ch_names)[:5])))


if __name__ == "__main__":
    main()
