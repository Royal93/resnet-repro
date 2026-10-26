r"""
Summarise finished runs: tables, figures and LaTeX number macros for the write-up.

    python analyze.py                       # reads runs/, writes results/
    python analyze.py --runs runs_fresh --out results_fresh

A "variant" is a run setting other than the main one (option A shortcuts, mixed precision, no tag):
for example option B shortcuts (--shortcut B) or full precision (--no-amp --tag fp32).
Main runs and variants are never averaged together.

Writes (all into --out):
    summary.md / summary.csv   every model and variant: mean +/- std over seeds, each seed, the paper's value
    table_main.tex             LaTeX table: plain vs ResNet test error by depth, with the paper's values
    table_variants.tex         LaTeX table: ResNet-44/56 seeds and variants (FP32, option B)
    numbers.tex                LaTeX macros, e.g. \err{resnet}{56}, used in the text of the write-up
    fig6_side_by_side.png      training curves like the paper's Fig. 6 (dashed: train, bold: test)
    fig_depth.png              final test error against depth, plain vs ResNet, with the paper's values
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# Paper, Table 6 (CIFAR-10 test error, %). Plain-net values are not tabulated in the paper;
# they are read approximately from Fig. 6 (left).
PAPER_RESNET = {20: 8.75, 32: 7.51, 44: 7.17, 56: 6.97, 110: 6.43}
PAPER_TEXT = {
    ("resnet", 20): "8.75", ("resnet", 32): "7.51", ("resnet", 44): "7.17",
    ("resnet", 56): "6.97", ("resnet", 110): "6.43 (6.61 ± 0.16)",
    ("plain", 20): "≈ 9 to 10 (Fig. 6)", ("plain", 56): "≈ 13 (Fig. 6)", ("plain", 110): "> 60 (Fig. 6)",
}
# (name, first group, second group): first minus second, used as \cmpdiff{name} and \cmpt{name}.
COMPARISONS = [
    ("A-vs-B-56", ("resnet", 56, ""), ("resnet", 56, "B")),   # option A minus option B
    ("56-vs-44", ("resnet", 56, ""), ("resnet", 44, "")),     # ResNet-56 minus ResNet-44
    ("plain56-vs-20", ("plain", 56, ""), ("plain", 20, "")),  # degradation size
]
COLORS = {20: "#c9b200", 32: "#00b5d8", 44: "#22aa22", 56: "#e02020", 110: "#111111"}


def variant_of(cfg):
    """'' for the main setting; otherwise a short label such as 'B' or 'fp32'."""
    parts = []
    if cfg.get("shortcut", "A") != "A":
        parts.append(cfg["shortcut"])
    if cfg.get("tag"):
        parts.append(cfg["tag"])
    elif cfg.get("no_amp"):
        parts.append("fp32")
    return "-".join(parts)


def load_runs(runs_dir):
    groups = defaultdict(list)
    for res_path in sorted(glob.glob(os.path.join(runs_dir, "*", "result.json"))):
        d = os.path.dirname(res_path)
        res = json.load(open(res_path))
        cfg = json.load(open(os.path.join(d, "config.json")))
        if res.get("eval_set", "test10k") != "test10k" or cfg.get("iters", 64000) != 64000:
            continue  # skip validation-split and short test runs
        rows = list(csv.DictReader(open(os.path.join(d, "log.csv"))))
        res["log"] = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
        res["variant"] = variant_of(cfg)
        groups[(res["arch"], res["depth"], res["variant"])].append(res)
    return groups


def stats(runs):
    e = np.array([r["final_test_err"] for r in runs])
    return e.mean(), (e.std(ddof=1) if len(e) > 1 else None), e


def fmt(mean, std, n, tex=False):
    pm = " $\\pm$ " if tex else " ± "
    return f"{mean:.2f}{pm}{std:.2f}" if std is not None else f"{mean:.2f}" + (" (1 run)" if not tex else "$^{\\dagger}$")


def write_summary(groups, out):
    lines = ["| Model | Variant | Params | Runs | Test error % (mean ± std) | Each seed | Paper |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    for key in sorted(groups, key=lambda k: (k[2] != "", k[0], k[1], k[2])):
        arch, depth, var = key
        m, s, e = stats(groups[key])
        name = f"{'ResNet' if arch == 'resnet' else 'plain'}-{depth}"
        paper = PAPER_TEXT.get((arch, depth), "n/a") if var == "" else "not reported"
        seeds = ", ".join(f"s{r['seed']}: {r['final_test_err']:.2f}" for r in groups[key])
        lines.append(f"| {name} | {var or 'main (A, AMP)'} | {groups[key][0]['params']/1e6:.2f}M | {len(e)} | "
                     f"{fmt(m, s, len(e))} | {seeds} | {paper} |")
        rows.append([name, var or "main", groups[key][0]["params"], len(e), f"{m:.3f}",
                     "" if s is None else f"{s:.3f}", seeds, paper])
    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "params", "runs", "mean_test_err", "std_test_err", "per_seed", "paper"])
        w.writerows(rows)
    print("\n".join(lines))


def write_latex(groups, out):
    # Main table: plain vs ResNet by depth.
    depths = sorted({d for (a, d, v) in groups if v == ""})
    L = [r"\begin{tabular}{@{}rcccc@{}}", r"\toprule",
         r"Depth & Params & Plain (ours) & ResNet (ours) & ResNet (paper) \\", r"\midrule"]
    for d in depths:
        cells = []
        for a in ("plain", "resnet"):
            if (a, d, "") in groups:
                m, s, e = stats(groups[(a, d, "")])
                cells.append(fmt(m, s, len(e), tex=True))
            else:
                cells.append("--")
        params = next(groups[k][0]["params"] for k in groups if k[1] == d)
        paper = f"{PAPER_RESNET[d]:.2f}" if d in PAPER_RESNET else "--"
        L.append(f"{d} & {params/1e6:.2f}M & {cells[0]} & {cells[1]} & {paper} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_main.tex"), "w").write("\n".join(L) + "\n")

    # Variants table: every ResNet-44/56 setting, seed by seed.
    V = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Model and setting & Seeds & Each seed & Mean $\pm$ std \\", r"\midrule"]
    label = {"": "option A, FP16 (main)", "fp32": "option A, FP32", "B": "option B, FP16", "B-fp32": "option B, FP32"}
    for key in sorted((k for k in groups if k[0] == "resnet" and k[1] in (44, 56)), key=lambda k: (k[1], k[2])):
        m, s, e = stats(groups[key])
        each = ", ".join(f"{x:.2f}" for x in e)
        V.append(f"ResNet-{key[1]}, {label.get(key[2], key[2])} & {len(e)} & {each} & {fmt(m, s, len(e), tex=True)} \\\\")
    V += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_variants.tex"), "w").write("\n".join(V) + "\n")

    # Macros: \err{arch}{depth[-variant]} -> "mean +/- std", \errmean -> mean, \errbest -> best seed, \nruns.
    M = [r"% Generated by analyze.py. Do not edit by hand.",
         r"\providecommand{\err}[2]{\ifcsname err-#1-#2\endcsname\csname err-#1-#2\endcsname\else\textcolor{red}{[#1-#2 missing]}\fi}",
         r"\providecommand{\errmean}[2]{\ifcsname errmean-#1-#2\endcsname\csname errmean-#1-#2\endcsname\else\textcolor{red}{[#1-#2 missing]}\fi}",
         r"\providecommand{\errbest}[2]{\ifcsname errbest-#1-#2\endcsname\csname errbest-#1-#2\endcsname\else\textcolor{red}{[#1-#2 missing]}\fi}",
         r"\providecommand{\nruns}[2]{\ifcsname nruns-#1-#2\endcsname\csname nruns-#1-#2\endcsname\else\textcolor{red}{[#1-#2 missing]}\fi}"]
    for (a, d, v), runs in groups.items():
        k = f"{a}-{d}" + (f"-{v}" if v else "")
        k = k.split("-", 1)  # arch, rest
        m, s, e = stats(runs)
        val = fmt(m, s, len(e), tex=True)
        M.append(rf"\expandafter\def\csname err-{k[0]}-{k[1]}\endcsname{{{val}}}")
        M.append(rf"\expandafter\def\csname errmean-{k[0]}-{k[1]}\endcsname{{{m:.2f}}}")
        M.append(rf"\expandafter\def\csname errbest-{k[0]}-{k[1]}\endcsname{{{e.min():.2f}}}")
        M.append(rf"\expandafter\def\csname nruns-{k[0]}-{k[1]}\endcsname{{{len(e)}}}")
    M.append(r"\providecommand{\gap}[1]{\ifcsname gap-#1\endcsname\csname gap-#1\endcsname\else\textcolor{red}{[gap-#1 missing]}\fi}")
    for (a_, d, v), runs in groups.items():
        if a_ == "resnet" and v == "" and d in PAPER_RESNET:
            g = stats(runs)[0] - PAPER_RESNET[d]
            M.append(rf"\expandafter\def\csname gap-{d}\endcsname{{{abs(g):.2f}}}")  # |ours - paper|; the text says better/worse
    M.append(r"\providecommand{\trainerr}[2]{\ifcsname trainerr-#1-#2\endcsname\csname trainerr-#1-#2\endcsname\else\textcolor{red}{[train #1-#2 missing]}\fi}")
    for (a_, d, v), runs in groups.items():
        if v == "":  # final logged training error (minibatch, with augmentation), mean over seeds
            te = np.mean([r["log"]["train_err"][-1] for r in runs])
            M.append(rf"\expandafter\def\csname trainerr-{a_}-{d}\endcsname{{{te:.2f}}}")
    # Pairwise comparisons used in the text: difference of means and Welch's t statistic.
    M.append(r"\providecommand{\cmpdiff}[1]{\ifcsname cmpdiff-#1\endcsname\csname cmpdiff-#1\endcsname\else\textcolor{red}{[#1 missing]}\fi}")
    M.append(r"\providecommand{\cmpt}[1]{\ifcsname cmpt-#1\endcsname\csname cmpt-#1\endcsname\else\textcolor{red}{[#1 missing]}\fi}")
    for name, k1, k2 in COMPARISONS:
        if k1 in groups and k2 in groups and len(groups[k1]) > 1 and len(groups[k2]) > 1:
            m1, s1, e1 = stats(groups[k1]); m2, s2, e2 = stats(groups[k2])
            t = (m1 - m2) / np.sqrt(s1 ** 2 / len(e1) + s2 ** 2 / len(e2))
            M.append(rf"\expandafter\def\csname cmpdiff-{name}\endcsname{{{m1 - m2:.2f}}}")
            M.append(rf"\expandafter\def\csname cmpt-{name}\endcsname{{{t:.1f}}}")
    total = sum(len(r) for r in groups.values())
    M.append(rf"\providecommand{{\totalruns}}{{{total}}}")
    open(os.path.join(out, "numbers.tex"), "w").write("\n".join(M) + "\n")


def plot_panel(ax, groups, arch):
    depths = sorted(d for (a, d, v) in groups if a == arch and v == "")
    for d in depths:
        runs = groups[(arch, d, "")]
        n = min(len(r["log"]["iter"]) for r in runs)
        it = runs[0]["log"]["iter"][:n] / 1e4
        tr = np.mean([r["log"]["train_err"][:n] for r in runs], axis=0)
        te = np.mean([r["log"]["test_err"][:n] for r in runs], axis=0)
        c = COLORS.get(d)
        label = f"{'ResNet' if arch == 'resnet' else 'plain'}-{d}"
        ax.plot(it, tr, "--", color=c, linewidth=0.9)
        ax.plot(it, te, "-", color=c, linewidth=1.8, label=label)
    ax.set_ylim(0, 20)
    ax.set_xlabel("iter. (1e4)")
    ax.set_ylabel("error (%)")
    for y in (5, 10):
        ax.axhline(y, color="grey", linestyle=":", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(("Plain" if arch == "plain" else "Residual") + " networks", fontsize=10)


def plot_depth(groups, out):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for arch, color, marker in (("plain", "#b3342b", "s"), ("resnet", "#0e7c86", "o")):
        ds = sorted(d for (a, d, v) in groups if a == arch and v == "")
        if not ds:
            continue
        means, stds = [], []
        for d in ds:
            m, s, e = stats(groups[(arch, d, "")])
            means.append(m); stds.append(0 if s is None else s)
        ax.errorbar(ds, means, yerr=stds, color=color, marker=marker, capsize=3,
                    label=f"{'Plain' if arch == 'plain' else 'ResNet'} (ours)")
    pd = sorted(PAPER_RESNET)
    ax.plot(pd, [PAPER_RESNET[d] for d in pd], "o--", color="#0e7c86", mfc="white", alpha=0.8, label="ResNet (paper)")
    ax.set_xscale("log")
    ax.set_xticks([20, 32, 44, 56, 110]); ax.set_xticklabels(["20", "32", "44", "56", "110"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.minorticks_off()
    ax.set_xlabel("depth (layers)"); ax.set_ylabel("test error (%)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_depth.png"), dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups = load_runs(a.runs)
    if not groups:
        raise SystemExit(f"No finished runs found in {a.runs}/ (each needs a result.json).")
    write_summary(groups, a.out)
    write_latex(groups, a.out)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, arch in zip(axes, ("plain", "resnet")):
        plot_panel(ax, groups, arch)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig6_side_by_side.png"), dpi=300); plt.close(fig)
    plot_depth(groups, a.out)
    print(f"\nTables, figures and numbers.tex written to {a.out}/")


if __name__ == "__main__":
    main()
