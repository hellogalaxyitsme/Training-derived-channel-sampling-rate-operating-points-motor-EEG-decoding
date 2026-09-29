"""Model complexity (parameters, MACs) and CPU inference latency per operating point.

Configurations use the median fold-level r* and the median realised T3 sample
count of each dataset (within-session rows of the operating-point table
written by ``scripts/aggregate.py``). Datasets without rows in that table are
skipped. MACs are counted analytically with forward hooks: Conv1d/Conv2d
(including grouped/depthwise), Linear, multi-head attention (input/output
projections plus the two attention matrix products), and the FFT temporal
layer counted as the equivalent direct convolution. Latency: batch 1, single
process, torch.set_num_threads(N), 50 warm-up passes (5 for networks with at least
10^9 MACs) and 300 timed passes, with the
first temporal convolution evaluated by FFT (numerically equivalent to the
direct convolution, whose CPU cost is very high for kernels of hundreds of taps).

Usage: python scripts/complexity_latency.py [N_THREADS] [--op-table PATH] [--out PATH]
Writes tables/complexity_latency.csv by default.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from eegop.models import FFTTemporalConv, get_model  # noqa: E402

DEFAULTS = {  # dataset: (C, n_full, K, fs_full, T_epoch)
    "cho2017": (64, 750, 2, 250., 3.0),
    "lee2019_mi": (62, 4001, 2, 1000., 4.0),
    "bnci2014_001": (22, 1001, 4, 250., 4.0),
    "schirrmeister2017": (128, 2001, 4, 500., 4.0),
}


def op_points(path):
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        for ds in DEFAULTS:
            rr = [r for r in rows if r["dataset"] == ds and r["mode"] == "within"]
            if rr:
                out[ds] = (int(np.median([int(r["r_star"]) for r in rr])),
                           int(np.median([int(float(r["n_samples_T3"])) for r in rr])))
    return out


def count_macs(model, x):
    macs = [0]

    def conv_hook(m, inp, out):
        k = int(np.prod(m.kernel_size)) * (m.in_channels // m.groups)
        macs[0] += int(out.numel() // out.shape[0]) * k

    def fft_hook(m, inp, out):
        macs[0] += int(out.numel() // out.shape[0]) * m.kern

    def lin_hook(m, inp, out):
        macs[0] += int(out.numel() // out.shape[0]) * m.in_features

    def mha_hook(m, inp, out):
        q = inp[0]
        L, E = q.shape[1], q.shape[2]
        macs[0] += 3 * L * E * E + 2 * L * L * E + L * E * E

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, FFTTemporalConv):
            hooks.append(m.register_forward_hook(fft_hook))
        elif isinstance(m, nn.Linear) and not isinstance(m, nn.modules.linear.NonDynamicallyQuantizableLinear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(mha_hook))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return macs[0]


def latency(model, x, warm=50, reps=300):
    ts = []
    with torch.inference_mode():
        for _ in range(warm):
            model(x)
        for _ in range(reps):
            t0 = time.perf_counter(); model(x); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3, float(np.percentile(ts, 95)) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("threads", nargs="?", type=int, default=1)
    ap.add_argument("--op-table", default=os.path.join(ROOT, "tables", "eval_operating_points.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tables", "complexity_latency.csv"))
    args = ap.parse_args()
    threads = args.threads
    torch.set_num_threads(threads)
    ops = op_points(args.op_table)
    rows = []
    for ds, (C, n_full, K, fs, T_ep) in DEFAULTS.items():
        r_med, n3 = ops.get(ds, (None, None))
        if r_med is None:
            continue
        cfgs = {"full": (C, n_full), "T1": (r_med, n_full), "T2": (C, n3), "T3": (r_med, n3)}
        for arch in ("eegnet", "atcnet"):
            base = None
            for name, (c, n) in cfgs.items():
                torch.manual_seed(0)
                m = get_model(arch, c, n, K, fs).eval()
                x = torch.randn(1, c, n)
                params = sum(p.numel() for p in m.parameters())
                macs = count_macs(m, x)
                # latency is measured with the FFT form of the first temporal convolution
                mf = get_model(arch, c, n, K, fs, fft_temporal=True).eval()
                mf.load_state_dict(m.state_dict())
                reps = 300
                med, p95 = latency(mf, x, warm=50 if macs < 1e9 else 5, reps=reps)
                # temporal kernel length of the first temporal convolution
                kern = [mm.kernel_size[-1] for mm in m.modules() if isinstance(mm, nn.Conv2d)][0]
                row = dict(dataset=ds, decoder=arch, config=name, channels=c, samples=n,
                           rate_hz=round(fs * n / n_full, 2), input_values=c * n,
                           params=params, macs=macs, temporal_kernel_samples=kern,
                           temporal_kernel_ms=round(1e3 * kern / (fs * n / n_full), 1),
                           latency_median_ms=round(med, 3), latency_p95_ms=round(p95, 3), timed_passes=reps,
                           threads=threads, cpu=platform.processor(), torch=torch.__version__)
                if name == "full":
                    base = row
                for k in ("input_values", "params", "macs", "latency_median_ms"):
                    row[k + "_pct_of_full"] = round(100 * row[k] / base[k], 2)
                rows.append(row)
                print(row, flush=True)
    if not rows:
        print(f"no operating points found in {args.op_table}; run scripts/aggregate.py first")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
