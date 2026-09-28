"""Tables, figures and result numbers reported in the paper, from the aggregated CSV files.

Usage:
    python scripts/make_tables_figures.py --results results --out paper_outputs [--runs runs]

Inputs  (--results): eval_statistics.csv, eval_operating_points.csv, eval_training.csv,
                     complexity_latency.csv and channel_baselines_classical.csv (optional)
Outputs (--out):     tab_ops.tex, tab_primary.tex, tab_complexity.tex,
                     figures/fig_primary_forest.{pdf,png}, figures/fig_baselines.{pdf,png},
                     result_numbers.tex (LaTeX macros for every number quoted in the text)
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import argparse  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--results", default="results", help="directory with the aggregated CSV files")
_ap.add_argument("--out", default="paper_outputs", help="output directory")
_ap.add_argument("--runs", default="runs", help="per-subject run directory (for estimation times)")
_args = _ap.parse_args()
TAB = _args.results
MAN = _args.out
FIG = os.path.join(MAN, "figures")
os.makedirs(FIG, exist_ok=True)

DS_LABEL = {"cho2017": "Cho2017", "lee2019_mi": "Lee2019\\_MI", "schirrmeister2017": "HGD",
            "bnci2014_001": "BNCI2014\\_001"}
DS_PLAIN = {k: v.replace("\\_", "_") for k, v in DS_LABEL.items()}
DEC_LABEL = {"csp": "CSP", "fbcsp": "FBCSP", "riemann": "MDM", "tslr": "TS-LR", "eegnet": "EEGNet",
             "atcnet": "ATCNet-S", "eegnet_bp": "EEGNet (8--30 Hz)"}
DEC_COLOR = {"csp": "#2a78d6", "riemann": "#eb6834", "tslr": "#1baf7a", "eegnet": "#eda100",
             "atcnet": "#e87ba4", "fbcsp": "#4a3aa7", "eegnet_bp": "#4a3aa7"}
DEC_MARK = {"csp": "o", "riemann": "s", "tslr": "D", "eegnet": "^", "atcnet": "v", "fbcsp": "P",
            "eegnet_bp": "X"}
CAT_SHORT = {"Equivalent": "E", "Non-inferior": "NI", "Reduced better by > margin": "B",
             "Inconclusive": "I", "Loss > margin": "L"}
C_FULL = {"cho2017": 64, "lee2019_mi": 62, "schirrmeister2017": 128, "bnci2014_001": 22}
FS_FULL = {"cho2017": 250., "lee2019_mi": 1000., "schirrmeister2017": 500., "bnci2014_001": 250.}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42})

stats = pd.read_csv(os.path.join(TAB, "eval_statistics.csv"))
acc = stats[stats.metric == "acc"].copy()
ops = pd.read_csv(os.path.join(TAB, "eval_operating_points.csv"))
macros = {}


def fmt_pp(x, nd=1):
    return f"{100 * x:+.{nd}f}".replace("-", "$-$")


def row(ds, mode, dec, cfg):
    r = acc[(acc.dataset == ds) & (acc["mode"] == mode) & (acc.decoder == dec) & (acc.config == cfg)]
    return None if r.empty else r.iloc[0]


def cell(r, holm=True):
    if r is None:
        return "--"
    cat = CAT_SHORT[r["category_holm" if holm else "category"]]
    return (f"{fmt_pp(r['mean'])} [{fmt_pp(r['ci90_lo'])}, {fmt_pp(r['ci90_hi'])}] {cat}")


# ---------------------------------------------------------------- operating points
op_lines = []
for ds in ("cho2017", "lee2019_mi", "bnci2014_001", "schirrmeister2017"):
    o = ops[(ops.dataset == ds) & (ops["mode"] == "within")]
    if o.empty:
        continue
    C = C_FULL[ds]
    r = o.r_star
    fs = o.fs_star
    rho = o.input_ratio_realised
    cov_active = np.mean(np.isclose(fs, np.minimum(7 * C / (4.0 if ds != "cho2017" else 3.0), FS_FULL[ds])))
    op_lines.append(
        f"{DS_LABEL[ds]} & {C} & {r.mean():.1f} $\\pm$ {r.std():.1f} ({r.min()}--{r.max()}) & "
        f"{100 * r.mean() / C:.0f}\\% & {fs.mean():.1f}"
        + (f" ({fs.min():.1f}--{fs.max():.1f})" if fs.max() - fs.min() > 0.05 else "")
        + f" & {100 * rho.mean():.1f} \\\\")
    key = {"cho2017": "Cho", "lee2019_mi": "Lee", "schirrmeister2017": "HGD", "bnci2014_001": "BNCI"}[ds]
    macros[f"Rmean{key}"] = f"{r.mean():.1f}"
    macros[f"Rsd{key}"] = f"{r.std():.1f}"
    macros[f"Rpct{key}"] = f"{100 * r.mean() / C:.0f}"
    macros[f"Fs{key}"] = f"{fs.mean():.1f}"
    macros[f"Rho{key}"] = f"{100 * rho.mean():.1f}"
    macros[f"CovActive{key}"] = f"{100 * cov_active:.0f}"
    macros[f"NFolds{key}"] = str(len(o))
with open(os.path.join(MAN, "tab_ops.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\caption{Operating points estimated in the within-session outer-training "
            "partitions (mean $\\pm$ SD over folds, range). $\\rho$: retained input ratio from realised "
            "sample counts (mean over folds). The covariance term of Eq.~(\\ref{eq:rate}) set $f_s^*$ in every fold "
            "for Cho2017, Lee2019\\_MI, and HGD, and the bandwidth term in every fold for BNCI2014\\_001.}\n"
            "\\label{tab:ops}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nDataset & $C$ & $r^*$ & $r^*/C$ & $f_s^*$ (Hz) & "
            "$\\rho$ (\\%) \\\\\n\\midrule\n" + "\n".join(op_lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------------------------------------------------------- primary table
groups = [("cho2017", "within"), ("lee2019_mi", "within"), ("schirrmeister2017", "within"),
          ("bnci2014_001", "within"), ("bnci2014_001", "cross_AB"), ("lee2019_mi", "cross_AB")]
GLAB = {("cho2017", "within"): "Cho2017", ("lee2019_mi", "within"): "Lee2019\\_MI",
        ("schirrmeister2017", "within"): "HGD", ("bnci2014_001", "within"): "BNCI, session 1",
        ("bnci2014_001", "cross_AB"): "BNCI, S1$\\rightarrow$S2",
        ("lee2019_mi", "cross_AB"): "Lee, S1$\\rightarrow$S2"}
lines = []
for ds, mode in groups:
    first = True
    decs = ["csp", "riemann", "tslr", "eegnet", "atcnet"]
    present = [d for d in decs if row(ds, mode, d, "T1") is not None]
    for dec in present:
        rf = acc[(acc.dataset == ds) & (acc["mode"] == mode) & (acc.decoder == dec)]
        full = rf.iloc[0]["mean_full"] if not rf.empty else np.nan
        n = int(rf.iloc[0]["n"]) if not rf.empty else 0
        lab = (f"\\multirow{{{len(present)}}}{{*}}{{\\shortstack[l]{{{GLAB[(ds, mode)]}}}}}" if first else "")
        first = False
        lines.append(f"{lab} & {DEC_LABEL[dec]} & {n} & {100 * full:.1f} & "
                     + " & ".join(cell(row(ds, mode, dec, c)) for c in ("T1", "T2", "T3")) + " \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
_trp = os.path.join(TAB, "eval_training.csv")
_hgd_atc_folds = 5
if os.path.exists(_trp):
    _tr = pd.read_csv(_trp)
    _h = _tr[(_tr.dataset == "schirrmeister2017") & (_tr.decoder == "atcnet")]
    if len(_h):
        _hgd_atc_folds = int(_h.groupby("subject").split.nunique().min())
HGD_ATC_NOTE = "HGD ATCNet-S: first outer fold only. " if _hgd_atc_folds < 5 else ""
with open(os.path.join(MAN, "tab_primary.tex"), "w") as f:
    f.write("\\begin{table*}[t]\n\\caption{Accuracy differences, full minus reduced input (pp; positive = "
            "loss), with unadjusted 90\\% $t$ intervals and the Holm-adjusted category at the 3-pp margin: "
            "E, equivalent; NI, non-inferior but not equivalent; B, reduced better by more than the margin; "
            "I, inconclusive; L, loss larger than the margin. $n$: subjects. Full: mean full-input accuracy (\\%). "
            + HGD_ATC_NOTE + "S1$\\rightarrow$S2: trained on session 1, tested on session 2.}\n"
            "\\label{tab:primary}\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{@{}llcclll@{}}\n\\toprule\nData & Decoder & $n$ & Full & T1 (channels) & "
            "T2 (rate) & T3 (joint) \\\\\n\\midrule\n" + "\n".join(lines) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---------------------------------------------------------------- family counts
for rule in ("T1", "T2", "T3"):
    fam = acc[acc.family == f"primary_dev_{rule}"]
    macros[f"NIcount{rule}"] = str(int(fam["dec_holm_noninferior"].astype(str).eq("True").sum()))
    macros[f"EQcount{rule}"] = str(int(fam["dec_holm_equivalent"].astype(str).eq("True").sum()))
    macros[f"Lcount{rule}"] = str(int(fam["dec_holm_harm"].astype(str).eq("True").sum()))
    macros[f"Bcount{rule}"] = str(int(fam["dec_holm_better_by_delta"].astype(str).eq("True").sum()))
    macros[f"Famsize{rule}"] = str(len(fam))
    for mode in ("within", "cross_AB"):
        fb = acc[acc.family == f"confirm_bnci_{mode}_{rule}"]
        tag = "W" if mode == "within" else "X"
        macros[f"BNCINI{tag}{rule}"] = str(int(fb["dec_holm_noninferior"].astype(str).eq("True").sum()))
        macros[f"BNCIsize{tag}{rule}"] = str(len(fb))


# ---------------------------------------------------------------- figures
def forest(ax, sub, title, show_labels):
    y = 0
    ticks, labels = [], []
    for (ds, mode), g in sub:
        for dec in ["csp", "riemann", "tslr", "eegnet", "atcnet"]:
            r = g[g.decoder == dec]
            if r.empty:
                continue
            r = r.iloc[0]
            ni = str(r["dec_holm_noninferior"]) == "True"
            ax.plot([100 * r.ci90_lo, 100 * r.ci90_hi], [y, y], color=DEC_COLOR[dec], lw=1.2,
                    solid_capstyle="round")
            ax.plot(100 * r["mean"], y, DEC_MARK[dec], ms=4.5, mfc=DEC_COLOR[dec] if ni else "white",
                    mec=DEC_COLOR[dec], mew=1.0)
            ticks.append(y); labels.append(f"{GLAB[(ds, mode)].replace(chr(92), '').replace('$rightarrow$', '→')} · {DEC_LABEL[dec].replace('--', '–')}")
            y -= 1
        y -= 0.6
    ax.axvspan(-3, 3, color="#f0efec", zorder=0)
    ax.axvline(0, color="#52514e", lw=0.6)
    ax.axvline(3, color="#52514e", lw=0.6, ls="--")
    if show_labels:
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Full − reduced accuracy (pp)")
    ax.grid(axis="x", color="#e6e5e1", lw=0.5)


sel = acc[acc.config.isin(["T1", "T2", "T3"])]
fig, axes = plt.subplots(1, 3, figsize=(7.16, 6.2), sharey=True)
for ax, rule, title in zip(axes, ("T1", "T2", "T3"),
                           ("T1: reduced channels", "T2: reduced rate", "T3: joint")):
    sub = [((ds, mode), sel[(sel.dataset == ds) & (sel["mode"] == mode) & (sel.config == rule)])
           for ds, mode in groups]
    forest(ax, sub, title, ax is axes[0])
    lim = max(4.0, np.nanmax(np.abs(100 * sel[["ci90_lo", "ci90_hi"]].values)) + 0.5)
    ax.set_xlim(-lim, lim)
handles = [plt.Line2D([], [], color=DEC_COLOR[d], marker=DEC_MARK[d], lw=1, ms=4.5, label=DEC_LABEL[d])
           for d in ["csp", "riemann", "tslr", "eegnet", "atcnet"]]
handles.append(plt.Line2D([], [], color="#52514e", marker="o", mfc="white", lw=0, ms=4.5,
                          label="open: non-inferiority not established"))
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.55, -0.005))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(FIG, "fig_primary_forest.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_primary_forest.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# baselines / sensitivity figure: within-session, per dataset panel
CFG_ORDER = ["T2", "T2_k5", "T2_k10", "F128", "T1", "T1_JG", "SM", "T3_c075", "T3", "T3_c125"]
CFG_LABEL = {"T2": "T2 (k=7)", "T2_k5": "Rate k=5", "T2_k10": "Rate k=10", "F128": "Fixed 128 Hz",
             "T1": "T1", "T1_JG": "T1, contrast-only score", "SM": "Sensorimotor montage",
             "T3_c075": "T3, 0.75 r*", "T3": "T3", "T3_c125": "T3, 1.25 r*"}
dsets = [d for d in ("cho2017", "lee2019_mi", "bnci2014_001", "schirrmeister2017")
         if not acc[(acc.dataset == d) & (acc["mode"] == "within")].empty]
fig, axes = plt.subplots(1, len(dsets), figsize=(7.16, 3.6), sharey=True)
axes = np.atleast_1d(axes)
decs_b = ["csp", "riemann", "tslr", "eegnet"]
for ax, ds in zip(axes, dsets):
    for j, dec in enumerate(decs_b):
        for i, cfg in enumerate(CFG_ORDER):
            r = row(ds, "within", dec, cfg)
            if r is None:
                continue
            yy = -i + (j - 1.5) * 0.18
            ni = str(r["dec_holm_noninferior"]) == "True"
            ax.plot([100 * r.ci90_lo, 100 * r.ci90_hi], [yy, yy], color=DEC_COLOR[dec], lw=0.9)
            ax.plot(100 * r["mean"], yy, DEC_MARK[dec], ms=3.5, mfc=DEC_COLOR[dec] if ni else "white",
                    mec=DEC_COLOR[dec], mew=0.8)
    ax.axvspan(-3, 3, color="#f0efec", zorder=0)
    ax.axvline(0, color="#52514e", lw=0.6)
    ax.axvline(3, color="#52514e", lw=0.6, ls="--")
    for yline in (-3.5, -6.5):
        ax.axhline(yline, color="#d6d5d0", lw=0.5)
    ax.set_yticks([-i for i in range(len(CFG_ORDER))])
    ax.set_yticklabels([CFG_LABEL[c] for c in CFG_ORDER])
    ax.tick_params(axis="y", length=0)
    ax.set_title(DS_PLAIN[ds].replace("schirrmeister2017", "HGD"), fontsize=8)
    ax.set_xlabel("Full − reduced (pp)")
    ax.grid(axis="x", color="#e6e5e1", lw=0.5)
handles = [plt.Line2D([], [], color=DEC_COLOR[d], marker=DEC_MARK[d], lw=1, ms=4, label=DEC_LABEL[d])
           for d in decs_b]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(FIG, "fig_baselines.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_baselines.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- complexity table
cl_path = os.path.join(TAB, "complexity_latency.csv")
if os.path.exists(cl_path):
    cl = pd.read_csv(cl_path)
    lines = []
    for ds in ("cho2017", "lee2019_mi", "bnci2014_001", "schirrmeister2017"):
        for dec in ("eegnet", "atcnet"):
            s = cl[(cl.dataset == ds) & (cl.decoder == dec)].set_index("config")
            if s.empty:
                continue
            f_, t3 = s.loc["full"], s.loc["T3"]
            lines.append(f"{DS_LABEL[ds]} & {DEC_LABEL[dec]} & {int(f_.channels)}$\\times${int(f_.samples)} $\\rightarrow$ "
                         f"{int(t3.channels)}$\\times${int(t3.samples)} & {t3.input_values_pct_of_full:.1f}\\% & "
                         f"{int(f_.params):,} / {int(t3.params):,} & {f_.macs / 1e6:.0f} / {t3.macs / 1e6:.1f} & "
                         f"{f_.latency_median_ms:.2f} / {t3.latency_median_ms:.2f} \\\\")
    with open(os.path.join(MAN, "tab_complexity.tex"), "w") as f:
        f.write("\\begin{table*}[t]\n\\caption{Deep-network size and CPU inference cost at the median T3 operating "
                "point (full / T3). MACs: multiply--accumulate operations per trial (millions). Latency: median of "
                "300 single-trial forward passes, one CPU thread, after warm-up, with the first temporal convolution evaluated by FFT; excludes preprocessing and the "
                "one-off operating-point estimation.}\n\\label{tab:complexity}\n\\centering\n\\small\n"
                "\\begin{tabular}{@{}llccccc@{}}\n\\toprule\nDataset & Network & Input (ch.$\\times$samples) & "
                "Input kept & Parameters & MACs (M) & Latency (ms) \\\\\n\\midrule\n" + "\n".join(lines) +
                "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ---------------------------------------------------------------- per-comparison macros
MDS = {"cho2017": "Cho", "lee2019_mi": "Lee", "bnci2014_001": "BNCI", "schirrmeister2017": "HGD"}
MMODE = {"within": "", "cross_AB": "XAB", "cross_BA": "XBA"}
MDEC = {"csp": "Csp", "fbcsp": "Fbcsp", "riemann": "Mdm", "tslr": "Tslr", "eegnet": "Eegnet",
        "atcnet": "Atcnet", "eegnet_bp": "Eegnetbp"}
MCFG = {"T1": "TOne", "T2": "TTwo", "T3": "TThree", "T2_k5": "KFive", "T2_k10": "KTen",
        "F128": "FixedRate", "SM": "Montage", "T1_JG": "JG", "T3_c075": "CLow", "T3_c125": "CHigh"}
for _, r in acc.iterrows():
    if r.config not in MCFG or r.decoder not in MDEC:
        continue
    key = MDS[r.dataset] + MMODE[r["mode"]] + MDEC[r.decoder] + MCFG[r.config]
    macros["d" + key] = fmt_pp(r["mean"])
    macros["ci" + key] = f"[{fmt_pp(r['ci90_lo'])}, {fmt_pp(r['ci90_hi'])}]"
    macros["full" + key] = f"{100 * r.mean_full:.1f}"
    macros["red" + key] = f"{100 * r.mean_reduced:.1f}"
    macros["cat" + key] = CAT_SHORT[r.category_holm]
    macros["n" + key] = str(int(r.n))

# same-size channel subsets (classical decoders)
cb_path = os.path.join(TAB, "channel_baselines_classical.csv")
if os.path.exists(cb_path):
    cb = pd.read_csv(cb_path)
    for _, r in cb.iterrows():
        key = MDS[r.dataset] + MDEC[r.decoder] + ("Rank" if "csp_rank" in r.comparison else "Rand")
        macros["sub" + key] = f"{100 * r.mean_diff:+.1f}".replace("-", "$-$")
        macros["subci" + key] = f"[{100 * r.ci95_lo:+.1f}, {100 * r.ci95_hi:+.1f}]".replace("-", "$-$")
        macros["subp" + key] = f"{r.p_two_sided:.2g}"

# one-off estimation time per training partition (includes the 50-permutation
# parallel-analysis count and the contrast-only variant, which the rule does not need)
import glob as _glob  # noqa: E402
import json as _json  # noqa: E402
est = {}
for ds in MDS:
    ts = [_json.load(open(p))["estimate_s"]
          for p in _glob.glob(os.path.join(_args.runs, ds, "within", "operating_points", "*.json"))]
    if ts:
        est[ds] = np.median(ts)
if est:
    lo, hi = min(est.values()), max(est.values())
    macros["EstTimeSummary"] = (f"a median of {lo:.1f}--{hi:.0f}~s (depending on the dataset)")

# kappa: primary development family, Holm non-inferiority counts
kap = stats[(stats.metric == "kappa")]
for rule in ("T1", "T2", "T3"):
    fam = kap[kap.family == f"primary_dev_{rule}_kappa"]
    macros[f"KapNI{rule}"] = str(int(fam["dec_holm_noninferior"].astype(str).eq("True").sum()))
    macros[f"KapSize{rule}"] = str(len(fam))

# deep-network stopping behaviour
tr_path = os.path.join(TAB, "eval_training.csv")
if os.path.exists(tr_path):
    tr = pd.read_csv(tr_path)
    tr = tr[tr.best_epoch.notna()]
    for ds in MDS:
        for dec in ("eegnet", "atcnet"):
            t = tr[(tr.dataset == ds) & (tr.decoder == dec) & (tr["mode"] == "within")]
            if len(t):
                macros[f"EpMed{MDS[ds]}{MDEC[dec]}"] = f"{t.best_epoch.median():.0f}"
                macros[f"EpEarly{MDS[ds]}{MDEC[dec]}"] = f"{100 * np.mean(t.best_epoch <= 3):.0f}"

with open(os.path.join(MAN, "result_numbers.tex"), "w") as f:
    words = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six",
                                    "Seven", "Eight", "Nine"]))
    seen = set()
    for k, v in sorted(macros.items()):
        name = "".join(words.get(ch, ch) for ch in k if ch.isalnum())
        assert name not in seen, name
        seen.add(name)
        f.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")
print("wrote tables, figures and result_numbers.tex to", MAN)
