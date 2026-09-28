"""Build the per-dataset epoch caches used by ``scripts/run_evaluation.py``.

Usage
-----
python scripts/build_cache.py                       # all datasets
python scripts/build_cache.py cho2017 lee2019_mi    # selected datasets

Each within-session dataset is written to
``$EEGOP_CACHE/<dataset>/<dataset>_pooled.pkl`` (default cache directory
``~/mne_data/eegop_cache``) as a dictionary ``{subject: {"X": X, "y": y}}``
with ``X`` of shape (n_trials, C, n_samples) in float32 and integer labels
``y``. BNCI2014_001 is written to ``$EEGOP_CACHE/bnci2014_001_sessions.pkl``
as ``{subject: {"X", "y", "session", "run"}}``; this is the same file that
``run_evaluation.py`` creates on first use.

MOABB downloads the raw recordings automatically on first use (to the MNE
data directory, ``~/mne_data`` by default). All datasets are epoched with the
MOABB ``MotorImagery`` paradigm with fmin = 0.5 Hz and fmax = 100 Hz:

==================  =======  =====  ====  =========  ==============================
dataset             classes  tmin   tmax  resample   channels kept
==================  =======  =====  ====  =========  ==============================
Cho2017             2        0      3 s   250 Hz     first 64 EEG channels
Lee2019_MI          2        0      4 s   1000 Hz    first 62 EEG channels
Schirrmeister2017   4        0      4 s   500 Hz     first 128 EEG channels
BNCI2014_001        4        0      4 s   250 Hz     all 22 EEG channels, both sessions
==================  =======  =====  ====  =========  ==============================

Lee2019_MI is loaded with the default MOABB ``Lee2019_MI`` class. With MOABB
1.5.0 this returns only the second recording session (offline run) of each
subject, because of the session-key behaviour documented in
``src/eegop/lee_sessions.py``. The within-session Lee2019_MI analysis
therefore uses session 2; the cross-session analysis loads both sessions
through ``eegop.lee_sessions.Lee2019MISessions`` at run time.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CACHE = Path(os.environ.get("EEGOP_CACHE", os.path.expanduser("~/mne_data/eegop_cache")))

POOLED = {
    # name: (MOABB class name, paradigm kwargs, label map, channels kept)
    "cho2017": ("Cho2017", dict(n_classes=2, tmin=0., tmax=3., resample=250.),
                {"left_hand": 0, "right_hand": 1}, 64),
    "lee2019_mi": ("Lee2019_MI", dict(n_classes=2, tmin=0., tmax=4., resample=1000.),
                   {"left_hand": 0, "right_hand": 1}, 62),
    "schirrmeister2017": ("Schirrmeister2017", dict(n_classes=4, tmin=0., tmax=4., resample=500.),
                          {"left_hand": 0, "right_hand": 1, "feet": 2, "rest": 3}, 128),
}


def _atomic_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(path)


def _labels(y, lmap):
    unknown = sorted({v for v in y if v not in lmap})
    if unknown:
        raise ValueError(f"unmapped labels {unknown}; known={sorted(lmap)}")
    return np.array([lmap[v] for v in y], dtype=np.int64)


def build_pooled(name: str, overwrite: bool = False) -> Path:
    import moabb.datasets as mdat
    from moabb.paradigms import MotorImagery

    cls_name, kw, lmap, n_ch = POOLED[name]
    path = CACHE / name / f"{name}_pooled.pkl"
    if path.exists() and not overwrite:
        print(f"{name}: cache exists at {path}", flush=True)
        return path
    ds = getattr(mdat, cls_name)()
    par = MotorImagery(fmin=0.5, fmax=100., **kw)
    out = {}
    for s in ds.subject_list:
        X, y, _ = par.get_data(dataset=ds, subjects=[s])
        X = X[:, :n_ch, :].astype(np.float32)
        out[s] = dict(X=X, y=_labels(y, lmap))
        print(f"{name} subject {s}: X={X.shape}", flush=True)
    _atomic_dump(out, path)
    print(f"{name}: wrote {path}", flush=True)
    return path


def build_bnci_sessions(overwrite: bool = False) -> Path:
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    path = CACHE / "bnci2014_001_sessions.pkl"
    if path.exists() and not overwrite:
        print(f"bnci2014_001: cache exists at {path}", flush=True)
        return path
    ds = BNCI2014_001()
    par = MotorImagery(n_classes=4, fmin=0.5, fmax=100., tmin=0., tmax=4., resample=250.)
    lmap = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    out = {}
    for s in ds.subject_list:
        X, y, meta = par.get_data(dataset=ds, subjects=[s])
        out[s] = dict(X=X.astype(np.float32), y=_labels(y, lmap),
                      session=meta["session"].astype(str).values, run=meta["run"].astype(str).values)
        print(f"bnci2014_001 subject {s}: X={X.shape} sessions={sorted(set(out[s]['session']))}",
              flush=True)
    _atomic_dump(out, path)
    print(f"bnci2014_001: wrote {path}", flush=True)
    return path


def main():
    names = list(POOLED) + ["bnci2014_001"]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="*", help="subset of: " + ", ".join(names))
    ap.add_argument("--overwrite", action="store_true", help="rebuild existing caches")
    args = ap.parse_args()
    selected = args.datasets or names
    bad = [n for n in selected if n not in names]
    if bad:
        ap.error(f"unknown dataset(s) {bad}; choose from {names}")

    import mne
    mne.set_log_level("ERROR")
    print(f"cache directory: {CACHE}", flush=True)
    for name in selected:
        if name == "bnci2014_001":
            build_bnci_sessions(args.overwrite)
        else:
            build_pooled(name, args.overwrite)


if __name__ == "__main__":
    main()
