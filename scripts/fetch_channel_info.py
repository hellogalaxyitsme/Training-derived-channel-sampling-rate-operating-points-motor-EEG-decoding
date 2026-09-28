"""Record channel names/types and epoch metadata for subject 1 of each dataset.

Writes ``channel_info/<dataset>.json`` for every requested dataset. The channel
names are used by ``scripts/run_evaluation.py`` to form the sensorimotor
montage configuration (SM). If the within-session cache for a dataset exists,
the first trials of the cache are also compared with a fresh MOABB load to
confirm that both share the channel order returned by the MOABB MotorImagery
paradigm.

Usage: python scripts/fetch_channel_info.py [dataset ...]   (default: all four)
"""
import json
import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import mne  # noqa: E402
import moabb  # noqa: E402
from moabb.datasets import BNCI2014_001, Cho2017, Lee2019_MI, Schirrmeister2017  # noqa: E402
from moabb.paradigms import MotorImagery  # noqa: E402

mne.set_log_level("ERROR")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("EEGOP_CACHE", os.path.expanduser("~/mne_data/eegop_cache"))
OUT_DIR = os.path.join(ROOT, "channel_info")
SPECS = {
    "cho2017": (Cho2017, dict(n_classes=2, tmin=0., tmax=3., resample=250.), "cho2017/cho2017_pooled.pkl"),
    "lee2019_mi": (Lee2019_MI, dict(n_classes=2, tmin=0., tmax=4., resample=1000.),
                   "lee2019_mi/lee2019_mi_pooled.pkl"),
    "schirrmeister2017": (Schirrmeister2017, dict(n_classes=4, tmin=0., tmax=4., resample=500.),
                          "schirrmeister2017/schirrmeister2017_pooled.pkl"),
    "bnci2014_001": (BNCI2014_001, dict(n_classes=4, tmin=0., tmax=4., resample=250.), None),
}


def main():
    names = sys.argv[1:] or list(SPECS)
    os.makedirs(OUT_DIR, exist_ok=True)
    for name in names:
        if name not in SPECS:
            raise SystemExit(f"unknown dataset {name}; choose from {list(SPECS)}")
        cls, kw, cache_rel = SPECS[name]
        ds = cls()
        par = MotorImagery(fmin=0.5, fmax=100., **kw)
        epochs, y, meta = par.get_data(dataset=ds, subjects=[1], return_epochs=True)
        X = epochs.get_data(copy=False)
        info = {
            "n_epochs": int(len(y)), "shape": list(X.shape), "sfreq": float(epochs.info["sfreq"]),
            "ch_names": list(epochs.ch_names),
            "ch_types": sorted(set(epochs.get_channel_types())),
            "sessions": {str(k): int(v) for k, v in meta["session"].value_counts().items()},
            "runs": {str(k): int(v) for k, v in meta["run"].value_counts().items()},
            "event_id": {k: int(v) for k, v in epochs.event_id.items()},
            "interval": list(getattr(ds, "interval", [])),
            "paradigm_kwargs": kw,
        }
        cache_path = os.path.join(CACHE, cache_rel) if cache_rel else None
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                c = pickle.load(f)[1]
            n = min(len(c["y"]), X.shape[0])
            info["cache_shape"] = list(c["X"].shape)
            info["cache_first_trials_max_abs_diff"] = float(
                np.max(np.abs(X[:n, : c["X"].shape[1]].astype(np.float32) - c["X"][:n])))
            del c
        out = {"moabb_version": moabb.__version__, "mne_version": mne.__version__,
               "datasets": {name: info}}
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        print(name, info["shape"], info["ch_types"], info["sessions"],
              info.get("cache_first_trials_max_abs_diff"), "->", path, flush=True)
        del epochs, X


if __name__ == "__main__":
    main()
