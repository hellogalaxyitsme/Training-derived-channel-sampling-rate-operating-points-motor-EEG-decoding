"""Aggregate per-subject JSON files into subject-level statistics.

Usage: python scripts/aggregate.py <runs_root> [--out tables] [--prefix eval]

Outputs (file names use the prefix; the default matches the files in results/)
  <prefix>_subject_level.csv     one row per dataset/mode/decoder/config/subject
  <prefix>_statistics.csv        paired tests (full minus reduced), Holm-adjusted
  <prefix>_operating_points.csv  fold-level r*, fs*, realised sample counts, input ratios
  <prefix>_training.csv          deep-decoder stopping epochs per configuration
  <prefix>_completeness.json     which jobs wrote their COMPLETE marker

Declared inference families
  primary_dev   : rules T1/T2/T3, datasets Cho2017/Lee2019_MI/Schirrmeister2017,
                  decoders CSP/MDM/EEGNet/ATCNet-S -> 12 tests per rule (Holm per rule)
  confirm_bnci  : rules T1/T2/T3 on BNCI2014_001 (within session 1, and trained on
                  session 1 and tested on session 2), same four decoders -> 4 tests
                  per rule and split
  secondary     : every other decoder/config/mode; Holm within (dataset, mode, config)
Margin: 3 percentage points of accuracy. Kappa is a secondary metric tested
with the margin translated for balanced classes: 0.03 * K/(K-1).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eegop.stats import add_holm, category, decisions, paired_tests  # noqa: E402

DELTA = 0.03
PRIMARY_DEC = ("csp", "riemann", "eegnet", "atcnet")
DEV = ("cho2017", "lee2019_mi", "schirrmeister2017")
RULES = ("T1", "T2", "T3")
N_CLASSES = {"cho2017": 2, "lee2019_mi": 2, "schirrmeister2017": 4, "bnci2014_001": 4}
C_FULL = {"cho2017": 64, "lee2019_mi": 62, "schirrmeister2017": 128, "bnci2014_001": 22}
FS_FULL = {"cho2017": 250., "lee2019_mi": 1000., "schirrmeister2017": 500., "bnci2014_001": 250.}


def family(ds, mode, dec, cfg):
    if cfg in RULES and dec in PRIMARY_DEC:
        if ds in DEV and mode == "within":
            return f"primary_dev_{cfg}"
        if ds == "bnci2014_001" and mode in ("within", "cross_AB"):
            return f"confirm_bnci_{mode}_{cfg}"
    return f"secondary_{ds}_{mode}_{cfg}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_root")
    ap.add_argument("--out", default=os.path.join(ROOT, "tables"))
    ap.add_argument("--prefix", default="eval",
                    help="file-name prefix of the output tables (default matches results/)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # acc[(ds, mode, dec, cfg)][subject] = dict(acc, kappa, bacc)
    acc = defaultdict(dict)
    op_rows, train_rows = [], []
    seen_ops = set()
    per = defaultdict(dict)   # (dataset, mode, subject) -> {(decoder, config, split): metrics}
    complete = {}
    for f in sorted(glob.glob(os.path.join(args.runs_root, "*", "*", "*", "subj_*.json"))):
        ds, mode, tag = f.split(os.sep)[-4:-1]
        if tag == "operating_points":
            continue
        complete[(ds, mode, tag)] = os.path.exists(os.path.join(os.path.dirname(f), "COMPLETE"))
        with open(f) as fh:
            rec = json.load(fh)
        subj = rec["subject"]
        for sp in rec["splits"]:
            key_op = (ds, mode, subj, sp["split"])
            if key_op not in seen_ops:
                seen_ops.add(key_op)
                cf = sp["configs"]
                if "T3" in cf and "full" in cf:
                    nf = cf["full"].get("n_samples")
                    n3 = cf["T3"].get("n_samples")
                    op_rows.append(dict(
                        dataset=ds, mode=mode, subject=subj, split=sp["split"],
                        n_train=sp["n_train"], n_test=sp["n_test"], r_star=sp["r_star"],
                        fs_star=sp["fs_star"], C=C_FULL[ds], fs_full=FS_FULL[ds],
                        n_samples_full=nf, n_samples_T3=n3,
                        input_ratio_nominal=sp["r_star"] * sp["fs_star"] / (C_FULL[ds] * FS_FULL[ds]),
                        input_ratio_realised=(sp["r_star"] * n3 / (C_FULL[ds] * nf)) if (nf and n3) else None))
            for dec, cfgs in sp["results"].items():
                for cfg, m in cfgs.items():
                    # the folds of one subject may be stored in more than one job directory
                    if (dec, cfg, sp["split"]) not in per[(ds, mode, subj)]:
                        per[(ds, mode, subj)][(dec, cfg, sp["split"])] = {k: m[k] for k in ("acc", "kappa", "bacc")}
                    if "best_epoch" in m:
                        train_rows.append(dict(dataset=ds, mode=mode, decoder=dec, config=cfg,
                                               subject=subj, split=sp["split"],
                                               best_epoch=m["best_epoch"], epochs_run=m["epochs_run"],
                                               n_params=m.get("n_params"), fit_s=m.get("fit_s")))
                    if m.get("status", "ok") != "ok":
                        train_rows.append(dict(dataset=ds, mode=mode, decoder=dec, config=cfg,
                                               subject=subj, split=sp["split"], status=m["status"]))

    for (ds, mode, subj), folds in per.items():
        grouped = defaultdict(list)
        for (dec, cfg, _split), m in folds.items():
            grouped[(dec, cfg)].append(m)
        for (dec, cfg), ms in grouped.items():
            acc[(ds, mode, dec, cfg)][subj] = {k: float(np.mean([m[k] for m in ms])) for k in ("acc", "kappa", "bacc")}

    subj_rows, stat_rows = [], []
    for (ds, mode, dec, cfg), by_s in sorted(acc.items()):
        for s, m in sorted(by_s.items()):
            subj_rows.append(dict(dataset=ds, mode=mode, decoder=dec, config=cfg, subject=s, **m))
        if cfg == "full" or (ds, mode, dec, "full") not in acc:
            continue
        full = acc[(ds, mode, dec, "full")]
        common = sorted(set(full) & set(by_s))
        if len(common) < 2:
            continue
        K = N_CLASSES[ds]
        for metric, margin in (("acc", DELTA), ("kappa", DELTA * K / (K - 1))):
            d = [full[s][metric] - by_s[s][metric] for s in common]
            r = paired_tests(d, margin)
            r.update(dataset=ds, mode=mode, decoder=dec, config=cfg, metric=metric,
                     family=family(ds, mode, dec, cfg) + ("" if metric == "acc" else "_kappa"),
                     mean_full=float(np.mean([full[s][metric] for s in common])),
                     mean_reduced=float(np.mean([by_s[s][metric] for s in common])),
                     n_subjects_full=len(full), n_subjects_cfg=len(by_s))
            stat_rows.append(r)
    add_holm(stat_rows, ["family"])
    for r in stat_rows:
        du = decisions(r)
        dh = decisions(r, suffix="_holm")
        r["category"] = category(du)
        r["category_holm"] = category(dh)
        for k, v in du.items():
            r["dec_" + k] = v
        for k, v in dh.items():
            r["dec_holm_" + k] = v

    def write(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(os.path.join(args.out, f"{args.prefix}_{name}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

    write("subject_level", subj_rows)
    write("statistics", stat_rows)
    write("operating_points", op_rows)
    write("training", train_rows)
    with open(os.path.join(args.out, f"{args.prefix}_completeness.json"), "w") as fh:
        json.dump({"/".join(k): v for k, v in complete.items()}, fh, indent=1)
    n_c = sum(complete.values())
    print(f"subject rows {len(subj_rows)}, tests {len(stat_rows)}, op rows {len(op_rows)}, "
          f"jobs complete {n_c}/{len(complete)}")


if __name__ == "__main__":
    main()
