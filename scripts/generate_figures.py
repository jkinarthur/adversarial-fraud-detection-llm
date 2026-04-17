"""
Generate all paper figures from experiment results JSON files.

Figures produced:
  fig1_prc_curves.pdf       — Precision-Recall curves (main models)
  fig2_ars_comparison.pdf   — ARS bar chart (Table II data)
  fig3_ablation_bars.pdf    — Ablation F1 / PRC-AUC bar chart
  fig4_traj_length.pdf      — Trajectory length F1 line plot
  fig5_hyperparam_heatmap.pdf — Hyperparameter grid heatmap (F1)
  fig6_industry_bars.pdf    — Industry cross-generalization bar chart

All figures are saved as high-res PDF (300 dpi) + PNG for easy preview.

Usage:
    cd AD-FTD
    python scripts/generate_figures.py \
        --results results \
        --out results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    _PLT_OK = True
except ImportError:
    print("[ERROR] matplotlib / numpy not installed.  Run:  pip install matplotlib numpy",
          file=sys.stderr)
    _PLT_OK = False


# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

ADFTD_COLOR = "#d62728"    # red — distinguishes AD-FTD from baselines
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#17becf", "#aec7e8",
]

MODEL_ORDER = [
    "LR", "SVM", "RF", "XGBoost",
    "LSTM", "TCN", "TCN_BiLSTM",
    "Informer", "Reformer", "TIME_LLM",
    "ParaEmb_FraudW2V",
    "AD-FTD",
]

ABLATION_ORDER = [
    "AD-FTD (full)",
    "w/o counterfactual",
    "w/o deviation encoding",
    "w/o adversarial training",
]


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] {path} not found — skipping figure", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
    plt.close(fig)
    print(f"  Saved: {out_dir / stem}.pdf / .png")


def _mean_std(val_str: str) -> Tuple[float, float]:
    """Parse '0.1234 ± 0.0056' into (0.1234, 0.0056)."""
    try:
        parts = val_str.split("±")
        return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        return float("nan"), 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — PRC curve proxy (bar chart of PRC-AUC since actual curves require
#             per-fold score arrays; we provide a bar + error-bar representation)
# ──────────────────────────────────────────────────────────────────────────────

def fig_prc_bars(data: dict, out_dir: Path) -> None:
    """Bar chart of PRC-AUC ± std for all models (approximates PRC curves)."""
    ordered = [m for m in MODEL_ORDER if m in data]
    ordered += [m for m in data if m not in ordered]

    means, stds, colors, labels = [], [], [], []
    for mname in ordered:
        row = data[mname]
        m, s = _mean_std(row.get("prc_auc", "0.0 ± 0.0"))
        means.append(m); stds.append(s)
        colors.append(ADFTD_COLOR if mname == "AD-FTD" else PALETTE[len(means) % len(PALETTE)])
        labels.append(mname.replace("_", " "))

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, capsize=3,
                  color=colors, edgecolor="white", linewidth=0.5,
                  error_kw={"elinewidth": 1.0, "ecolor": "black"})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("PRC-AUC")
    ax.set_title("PRC-AUC Comparison (mean ± std, 40 runs)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(
        handles=[mpatches.Patch(color=ADFTD_COLOR, label="AD-FTD (proposed)")],
        loc="upper left", framealpha=0.8,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "fig1_prc_bars")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — ARS comparison bar chart (Table II)
# ──────────────────────────────────────────────────────────────────────────────

def fig_ars_comparison(data: dict, out_dir: Path) -> None:
    ordered = [m for m in MODEL_ORDER if m in data]
    ordered += [m for m in data if m not in ordered]

    f1c_means, f1c_stds = [], []
    f1a_means, f1a_stds = [], []
    ars_means, ars_stds = [], []
    labels = []

    for mname in ordered:
        row = data[mname]
        mc, sc = _mean_std(row.get("f1_clean", "0 ± 0"))
        ma, sa = _mean_std(row.get("f1_adv",   "0 ± 0"))
        ar, sr = _mean_std(row.get("ars",       "0 ± 0"))
        f1c_means.append(mc); f1c_stds.append(sc)
        f1a_means.append(ma); f1a_stds.append(sa)
        ars_means.append(ar); ars_stds.append(sr)
        labels.append(mname.replace("_", " "))

    n = len(labels)
    x = np.arange(n)
    w = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Clean vs Adv F1
    ax1.bar(x - w / 2, f1c_means, w, yerr=f1c_stds, capsize=2,
            label="F1 (Clean)", color="#1f77b4", edgecolor="white")
    ax1.bar(x + w / 2, f1a_means, w, yerr=f1a_stds, capsize=2,
            label="F1 (Adv.)", color="#ff7f0e", edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax1.set_ylabel("Macro F1"); ax1.set_title("Clean vs Adversarial F1")
    ax1.set_ylim(0, 1.05); ax1.legend(loc="upper left")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5); ax1.set_axisbelow(True)

    # Right: ARS (lower is better)
    bar_colors = [ADFTD_COLOR if m == "AD-FTD" else "#7f7f7f" for m in ordered]
    ax2.bar(x, ars_means, yerr=ars_stds, capsize=2,
            color=bar_colors, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("ARS (lower = better)")
    ax2.set_title("Adversarial Robustness Score")
    ax2.set_ylim(0, max(ars_means) * 1.25 if ars_means else 0.5)
    ax2.legend(
        handles=[mpatches.Patch(color=ADFTD_COLOR, label="AD-FTD (proposed)")],
        loc="upper right", framealpha=0.8,
    )
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5); ax2.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out_dir, "fig2_ars_comparison")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — Ablation bar chart
# ──────────────────────────────────────────────────────────────────────────────

def fig_ablation(data: dict, out_dir: Path) -> None:
    ordered = [v for v in ABLATION_ORDER if v in data]
    ordered += [v for v in data if v not in ordered]

    f1_means, f1_stds = [], []
    prc_means, prc_stds = [], []
    labels = []

    for vname in ordered:
        row = data[vname]
        mf, sf = _mean_std(row.get("f1",      "0 ± 0"))
        mp, sp = _mean_std(row.get("prc_auc",  "0 ± 0"))
        f1_means.append(mf); f1_stds.append(sf)
        prc_means.append(mp); prc_stds.append(sp)
        labels.append(vname)

    n = len(labels)
    x = np.arange(n)
    w = 0.35
    colors_f1  = [ADFTD_COLOR if v == "AD-FTD (full)" else "#1f77b4" for v in ordered]
    colors_prc = [ADFTD_COLOR if v == "AD-FTD (full)" else "#ff7f0e" for v in ordered]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, f1_means, w, yerr=f1_stds, capsize=3,
           color=colors_f1, label="F1", edgecolor="white")
    ax.bar(x + w / 2, prc_means, w, yerr=prc_stds, capsize=3,
           color=colors_prc, label="PRC-AUC", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study — Component Contribution")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "fig3_ablation_bars")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4 — Trajectory length sensitivity line plot
# ──────────────────────────────────────────────────────────────────────────────

def fig_traj_length(data: dict, out_dir: Path) -> None:
    keys_sorted = sorted(data.keys(), key=lambda x: int(x.replace("T=", "")))
    T_vals = [int(k.replace("T=", "")) for k in keys_sorted]

    f1m,  f1s  = zip(*[_mean_std(data[k]["f1"])      for k in keys_sorted]) if keys_sorted else ([],[])
    pm,   ps   = zip(*[_mean_std(data[k]["prc_auc"])  for k in keys_sorted]) if keys_sorted else ([],[])
    rm,   rs   = zip(*[_mean_std(data[k]["roc_auc"])  for k in keys_sorted]) if keys_sorted else ([],[])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for vals, errs, label, color, ls in [
        (f1m, f1s, "F1",      "#d62728", "-"),
        (pm,  ps,  "PRC-AUC", "#1f77b4", "--"),
        (rm,  rs,  "ROC-AUC", "#2ca02c", ":"),
    ]:
        vals = list(vals); errs = list(errs)
        ax.plot(T_vals, vals, marker="o", color=color, linestyle=ls, label=label)
        ax.fill_between(
            T_vals,
            [v - e for v, e in zip(vals, errs)],
            [v + e for v, e in zip(vals, errs)],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Trajectory Length $T$")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity to Trajectory Length")
    ax.set_xticks(T_vals)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "fig4_traj_length")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5 — Hyperparameter heatmap (F1)
# ──────────────────────────────────────────────────────────────────────────────

def fig_hyperparam_heatmap(data: dict, out_dir: Path) -> None:
    if not data:
        return
    alphas = sorted({v["alpha"] for v in data.values()})
    betas  = sorted({v["beta"]  for v in data.values()})

    grid_f1  = np.zeros((len(alphas), len(betas)))
    grid_ars = np.zeros((len(alphas), len(betas)))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            key = f"a{alpha:.1f}_b{beta:.1f}"
            entry = data.get(key, {})
            mf, _ = _mean_std(entry.get("f1",  "0 ± 0"))
            ma, _ = _mean_std(entry.get("ars", "0 ± 0"))
            grid_f1[i, j]  = mf
            grid_ars[i, j] = ma

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, grid, title, cmap in [
        (ax1, grid_f1,  "F1",  "YlOrRd"),
        (ax2, grid_ars, "ARS (lower = better)", "YlOrRd_r"),
    ]:
        im = ax.imshow(grid, cmap=cmap, aspect="auto",
                       vmin=grid.min(), vmax=grid.max())
        ax.set_xticks(range(len(betas)));  ax.set_xticklabels([f"β={b}" for b in betas])
        ax.set_yticks(range(len(alphas))); ax.set_yticklabels([f"α={a}" for a in alphas])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(alphas)):
            for j in range(len(betas)):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.suptitle("Hyperparameter Sensitivity Grid")
    fig.tight_layout()
    _save(fig, out_dir, "fig5_hyperparam_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6 — Industry cross-generalization bars
# ──────────────────────────────────────────────────────────────────────────────

def fig_industry(data: dict, out_dir: Path) -> None:
    sectors = sorted(data.keys())
    f1m, f1s, prcm, prcs = [], [], [], []
    labels = []
    for s in sectors:
        row = data[s]
        mf, sf = _mean_std(row.get("f1",      "0 ± 0"))
        mp, sp = _mean_std(row.get("prc_auc",  "0 ± 0"))
        f1m.append(mf); f1s.append(sf)
        prcm.append(mp); prcs.append(sp)
        labels.append(s.split("–")[-1].strip() if "–" in s else s)

    n = len(labels)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, f1m, w, yerr=f1s, capsize=3,
           color="#1f77b4", label="F1", edgecolor="white")
    ax.bar(x + w / 2, prcm, w, yerr=prcs, capsize=3,
           color="#ff7f0e", label="PRC-AUC", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Cross-Industry Generalization (Leave-One-Out)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "fig6_industry_bars")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate paper figures")
    p.add_argument("--results", default="results",
                   help="Directory containing the table JSON files")
    p.add_argument("--out", default="results/figures",
                   help="Output directory for PDF/PNG figures")
    return p.parse_args()


def main():
    if not _PLT_OK:
        sys.exit(1)

    args = parse_args()
    rdir = Path(args.results)
    out_dir = Path(args.out)

    t1  = _load(rdir / "table1_main.json")
    t2  = _load(rdir / "table2_adv.json")
    t3  = _load(rdir / "table3_ablation.json")
    t4  = _load(rdir / "table4_traj_length.json")
    t5  = _load(rdir / "table5_industry.json")
    tb1 = _load(rdir / "tableB1_hyperparam.json")

    print("Generating figures...")
    if t1:  fig_prc_bars(t1, out_dir)
    if t2:  fig_ars_comparison(t2, out_dir)
    if t3:  fig_ablation(t3, out_dir)
    if t4:  fig_traj_length(t4, out_dir)
    if tb1: fig_hyperparam_heatmap(tb1, out_dir)
    if t5:  fig_industry(t5, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
