"""Subject-level evaluation runner: within-session CV or cross-session transfer.

Example
-------
python scripts/run_evaluation.py --dataset cho2017 --mode within \
    --decoders eegnet,atcnet --configs full,T1,T2,T3 --out runs

For every subject and split the operating point (r*, channel ranking, fs*) is
estimated from the training partition only and applied unchanged to the
held-out partition. One JSON file is written per subject; existing files are
skipped, so an interrupted job can be restarted with the same command.

Data
----
Cho2017, Lee2019_MI and Schirrmeister2017 (within-session mode) are read from
the pickle caches written by ``scripts/build_cache.py`` under the directory
given by the environment variable ``EEGOP_CACHE`` (default
``~/mne_data/eegop_cache``). BNCI2014_001 and the two-session Lee2019_MI data
used in cross-session mode are loaded through MOABB (BNCI2014_001 is cached
as ``bnci2014_001_sessions.pkl`` in the same directory after the first load).

Channel names (needed only for the sensorimotor montage configuration ``SM``)
are read from ``channel_info/<dataset>.json`` written by
``scripts/fetch_channel_info.py``; if that file is absent, ``SM`` is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eegop.estimators import (estimate_spatial_operating_point,  # noqa: E402
                              estimate_temporal_operating_point)
from eegop.protocol import (CLASSICAL, DEEP, eval_classical, idx_hash,  # noqa: E402
                            inner_split, make_input, train_eval_deep)

CACHE = Path(os.environ.get("EEGOP_CACHE", os.path.expanduser("~/mne_data/eegop_cache")))
DATASETS = {
    "cho2017": dict(cache="cho2017/cho2017_pooled.pkl", fs=250., T_epoch=3.0, n_classes=2),
    "lee2019_mi": dict(cache="lee2019_mi/lee2019_mi_pooled.pkl", fs=1000., T_epoch=4.0, n_classes=2),
    "schirrmeister2017": dict(cache="schirrmeister2017/schirrmeister2017_pooled.pkl", fs=500.,
                              T_epoch=4.0, n_classes=4),
    "bnci2014_001": dict(cache=None, fs=250., T_epoch=4.0, n_classes=4),
}
# Prespecified 10-10 sensorimotor montage.
SENSORIMOTOR = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
                "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6"]
N_PERM = 50          # label permutations for the parallel-analysis count r*_PA
SPECTRAL = dict(fmax_analysis=45., btask_fmin=4., btask_fmax=35.)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_within(name: str):
    """Return {subject: (X, y)} for the within-session analysis."""
    spec = DATASETS[name]
    if spec["cache"] is not None:
        with open(CACHE / spec["cache"], "rb") as f:
            d = pickle.load(f)
        return {s: (v["X"], np.asarray(v["y"])) for s, v in d.items()}
    # BNCI2014_001: the within-session analysis uses the first session only.
    out = {}
    for s, v in load_sessions(name).items():
        first = sorted(set(v["session"]))[0]
        m = v["session"] == first
        out[s] = (v["X"][m], v["y"][m])
    return out


class _LazyLee(dict):
    """Loads Lee2019_MI subjects (both sessions) on access; nothing is cached
    to disk because the two-session full-rate arrays need about 11 GB."""

    def __init__(self):
        super().__init__()
        import mne
        from moabb.paradigms import MotorImagery
        from eegop.lee_sessions import Lee2019MISessions
        mne.set_log_level("ERROR")
        self.ds = Lee2019MISessions()
        self.par = MotorImagery(n_classes=2, fmin=0.5, fmax=100., tmin=0., tmax=4., resample=1000.)
        for s in self.ds.subject_list:
            dict.__setitem__(self, s, None)

    def __getitem__(self, s):
        X, y, meta = self.par.get_data(dataset=self.ds, subjects=[s])
        lmap = {"left_hand": 0, "right_hand": 1}
        return dict(X=X.astype(np.float32), y=np.array([lmap[v] for v in y], dtype=np.int64),
                    session=meta["session"].astype(str).values, run=meta["run"].astype(str).values)


def load_sessions(name: str):
    """Return {subject: dict(X, y, session)} with session labels preserved."""
    if name == "lee2019_mi":
        return _LazyLee()
    cache = CACHE / f"{name}_sessions.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    import mne
    from moabb.paradigms import MotorImagery
    mne.set_log_level("ERROR")
    if name == "bnci2014_001":
        from moabb.datasets import BNCI2014_001
        ds = BNCI2014_001()
        par = MotorImagery(n_classes=4, fmin=0.5, fmax=100., tmin=0., tmax=4., resample=250.)
        lmap = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    else:
        raise ValueError(name)
    out = {}
    for s in ds.subject_list:
        X, y, meta = par.get_data(dataset=ds, subjects=[s])
        out[s] = dict(X=X.astype(np.float32), y=np.array([lmap[v] for v in y], dtype=np.int64),
                      session=meta["session"].astype(str).values, run=meta["run"].astype(str).values)
        print(f"loaded {name} subject {s}: {X.shape} sessions={sorted(set(out[s]['session']))}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=4)
    tmp.replace(cache)
    return out


def channel_names(name: str, path: Path):
    """Channel names in cache order, or None if the channel-info file is absent."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["datasets"][name]["ch_names"]


# ---------------------------------------------------------------------------
# Operating points
# ---------------------------------------------------------------------------
def fs_rule(fs_nyq: float, C: int, T_epoch: float, fs_full: float, k: float) -> float:
    """Sampling-rate rule fs = min(max(fs_nyq, k C / T_epoch), fs_full)."""
    return float(min(max(fs_nyq, k * C / T_epoch), fs_full))


def estimate_operating_point(X_tr, y_tr, fs, T_epoch, op_file: Path):
    """Estimate (or reload) the operating point of one training partition."""
    if op_file.exists():
        with open(op_file) as f:
            return json.load(f)
    t0 = time.perf_counter()
    spatial = estimate_spatial_operating_point(X_tr, y_tr, sfreq=fs, n_perm=N_PERM)
    spatial_jg = estimate_spatial_operating_point(X_tr, y_tr, sfreq=fs, n_perm=0, score="jensen")
    temp = estimate_temporal_operating_point(X_tr, y_tr, fs, T_epoch=T_epoch, **SPECTRAL)
    b = dict(
        r_star_PA=int(spatial["r_star_PA"]), r_star_rel=int(spatial["r_star_rel"]),
        r_star_PR=int(spatial["r_star_PR"]), r_star_effective=int(spatial["r_star_effective"]),
        ranking=[int(i) for i in spatial["channel_ranking"]],
        mass_sorted=[float(v) for v in spatial["score_per_direction"]],
        jg_r_star_effective=int(spatial_jg["r_star_effective"]),
        jg_r_star_PR=int(spatial_jg["r_star_PR"]), jg_r_star_rel=int(spatial_jg["r_star_rel"]),
        jg_ranking=[int(i) for i in spatial_jg["channel_ranking"]],
        B_task=float(temp["B_task"]), B_task_ftest=float(temp["B_task_ftest"]),
        fs_nyquist=float(temp["fs_nyquist"]), fs_cov=float(temp["fs_cov"]),
        fs_star=float(min(temp["fs_star"], fs)), estimate_s=float(time.perf_counter() - t0))
    op_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = op_file.with_name(f"{op_file.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(b, f)
    tmp.replace(op_file)
    return b


def build_configs(b, C, fs_full, T_epoch, names):
    """Map configuration names to (channel indices, target sampling rate)."""
    r = b["r_star_effective"]
    rank = b["ranking"]
    allc = list(range(C))
    fs_star = b["fs_star"]
    cfg = {
        "full": (allc, fs_full),
        "T1": (rank[:r], fs_full),
        "T2": (allc, fs_star),
        "T3": (rank[:r], fs_star),
        "T2_k5": (allc, fs_rule(b["fs_nyquist"], C, T_epoch, fs_full, 5.)),
        "T2_k10": (allc, fs_rule(b["fs_nyquist"], C, T_epoch, fs_full, 10.)),
        "T1_JG": (b["jg_ranking"][:b["jg_r_star_effective"]], fs_full),
        "T3_c075": (rank[:max(2, int(round(0.75 * r)))], fs_star),
        "T3_c125": (rank[:min(C, int(round(1.25 * r)))], fs_star),
    }
    if fs_full > 128.:
        cfg["F128"] = (allc, 128.)
    if names is not None:
        sm = [i for i, n in enumerate(names) if n in SENSORIMOTOR]
        if 4 <= len(sm) < C:
            cfg["SM"] = (sm, fs_full)
    return cfg


# ---------------------------------------------------------------------------
# Evaluation of one split
# ---------------------------------------------------------------------------
def run_split(X_tr, y_tr, X_te, y_te, b, args, spec, names, seed):
    C = X_tr.shape[1]
    fs_full, T_epoch, K = spec["fs"], spec["T_epoch"], spec["n_classes"]
    configs = build_configs(b, C, fs_full, T_epoch, names)
    want = [c for c in args.configs.split(",") if c in configs]
    res, cfg_meta = {}, {}
    in_idx, val_idx = inner_split(y_tr, seed)
    for cname in want:
        ch, fs = configs[cname]
        cfg_meta[cname] = dict(channels=[int(c) for c in ch],
                               ch_names=[names[c] for c in ch] if names else None,
                               fs_requested=float(fs), n_channels=len(ch))
    for dec in args.decoders.split(","):
        res[dec] = {}
        for cname in want:
            ch, fs = configs[cname]
            band = (8., 30.) if dec == "eegnet_bp" else None
            A_tr = make_input(X_tr, ch, fs, fs_full, band)
            A_te = make_input(X_te, ch, fs, fs_full, band)
            fs_real = fs_full * A_tr.shape[2] / X_tr.shape[2]
            cfg_meta[cname]["n_samples"] = int(A_tr.shape[2])
            cfg_meta[cname]["fs_realized"] = float(fs_real)
            if dec in CLASSICAL:
                out = eval_classical(dec, A_tr, y_tr, A_te, y_te, fs_real, K)
            elif dec in DEEP:
                arch = "eegnet" if dec == "eegnet_bp" else dec
                out = train_eval_deep(arch, A_tr[in_idx], y_tr[in_idx], A_tr[val_idx], y_tr[val_idx],
                                      A_te, y_te, fs_real, K, seed=seed,
                                      max_epochs=args.max_epochs, patience=args.patience)
            else:
                raise ValueError(dec)
            res[dec][cname] = out
            print(f"    {dec:9s} {cname:8s} C={len(ch):3d} T={A_tr.shape[2]:5d} acc={out['acc']:.3f}"
                  + (f" ep={out.get('best_epoch')}/{out.get('epochs_run')}" if dec in DEEP else "")
                  + f" {out['fit_s']:.1f}s", flush=True)
    return res, cfg_meta, dict(inner_train_hash=idx_hash(in_idx), inner_val_hash=idx_hash(val_idx),
                               n_inner_train=int(len(in_idx)), n_inner_val=int(len(val_idx)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--mode", default="within", choices=["within", "cross"])
    ap.add_argument("--direction", default="AB", choices=["AB", "BA"],
                    help="cross mode: AB trains on the first session and tests on the second")
    ap.add_argument("--decoders", required=True,
                    help="comma-separated: " + ",".join(CLASSICAL + DEEP))
    ap.add_argument("--configs", default="full,T1,T2,T3")
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--out", default="runs",
                    help="output root (relative paths are resolved against the repository root)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--folds", default="all",
                    help="comma-separated outer-fold indices to evaluate (within mode)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--channel-info", default=None,
                    help="channel-name JSON (default: channel_info/<dataset>.json in the repository)")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    ch_info = (Path(args.channel_info) if args.channel_info
               else ROOT / "channel_info" / f"{args.dataset}.json")
    names = channel_names(args.dataset, ch_info)
    if names is None:
        print(f"channel info not found at {ch_info}; configuration SM is skipped", flush=True)
    tag = args.tag or f"{args.decoders.replace(',', '-')}"
    mode_dir = args.mode if args.mode == "within" else f"cross_{args.direction}"
    out_dir = ROOT / args.out / args.dataset / mode_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    op_dir = ROOT / args.out / args.dataset / mode_dir / "operating_points"
    import torch
    run_meta = dict(argv=sys.argv, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    python=platform.python_version(), torch=torch.__version__,
                    cuda=torch.version.cuda, n_perm=N_PERM, spectral=SPECTRAL,
                    fs_full=spec["fs"], T_epoch=spec["T_epoch"])
    try:
        import mne
        import moabb
        import pyriemann
        import scipy
        import sklearn
        run_meta.update(moabb=moabb.__version__, mne=mne.__version__, pyriemann=pyriemann.__version__,
                        scipy=scipy.__version__, sklearn=sklearn.__version__, numpy=np.__version__)
    except Exception:
        pass
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=1)

    if args.mode == "within":
        data = load_within(args.dataset)
    else:
        data = load_sessions(args.dataset)
    subjects = list(data) if args.subjects == "all" else [
        type(next(iter(data)))(s) for s in args.subjects.split(",")]

    from sklearn.model_selection import StratifiedKFold
    for subj in subjects:
        fout = out_dir / f"subj_{subj}.json"
        if fout.exists():
            continue
        t_subj = time.perf_counter()
        rec = dict(subject=int(subj), dataset=args.dataset, mode=mode_dir, splits=[])
        if args.mode == "within":
            X, y = data[subj]
            _, counts = np.unique(y, return_counts=True)
            n_folds = max(2, min(5, int(counts.min())))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
            splits = list(skf.split(X, y))
            if args.folds != "all":
                keep = {int(k) for k in args.folds.split(",")}
                splits = [sp if i in keep else None for i, sp in enumerate(splits)]
        else:
            d = data[subj]
            X, y, sess = d["X"], d["y"], d["session"]
            levels = sorted(set(sess))
            a, bsess = (levels[0], levels[1]) if args.direction == "AB" else (levels[1], levels[0])
            splits = [(np.where(sess == a)[0], np.where(sess == bsess)[0])]
            rec["train_session"], rec["test_session"] = a, bsess
        print(f"== {args.dataset} {mode_dir} subject {subj}: X={X.shape}", flush=True)
        for k, sp_idx in enumerate(splits):
            if sp_idx is None:
                continue
            tr, te = sp_idx
            seed = args.seed * 100000 + int(subj) * 100 + k
            b = estimate_operating_point(X[tr], y[tr], spec["fs"], spec["T_epoch"],
                                         op_dir / f"subj_{subj}_split_{k}.json")
            print(f"  split {k}: r*={b['r_star_effective']} fs*={b['fs_star']:.2f} "
                  f"(nyq {b['fs_nyquist']:.1f}, cov {b['fs_cov']:.1f}) JG r*={b['jg_r_star_effective']}",
                  flush=True)
            res, cfg_meta, inner = run_split(X[tr], y[tr], X[te], y[te], b, args, spec, names, seed)
            rec["splits"].append(dict(split=k, seed=seed, train_hash=idx_hash(tr), test_hash=idx_hash(te),
                                      n_train=int(len(tr)), n_test=int(len(te)),
                                      y_test=[int(v) for v in y[te]], operating_point_file=f"subj_{subj}_split_{k}.json",
                                      r_star=b["r_star_effective"], fs_star=b["fs_star"],
                                      configs=cfg_meta, inner=inner, results=res))
        rec["wall_s"] = float(time.perf_counter() - t_subj)
        tmp = fout.with_name(f"{fout.name}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f)
        tmp.replace(fout)
        print(f"  subject {subj} done in {rec['wall_s'] / 60:.1f} min", flush=True)
    with open(out_dir / "COMPLETE", "w") as f:
        f.write(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
