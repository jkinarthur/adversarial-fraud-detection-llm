"""
Generate LaTeX-formatted tables from experiment JSON files  →  ready to paste
into AD_FTD_IEEE_TBD.tex.

Reads:
  results/table1_main.json        → Table I  (main performance comparison)
  results/table2_adv.json         → Table II (adversarial robustness)
  results/table3_ablation.json    → Table III (ablation study)
  results/table4_traj_length.json → Table IV (trajectory length sensitivity)
  results/table5_industry.json    → Table V  (industry analysis)
  results/tableB1_hyperparam.json → Table B1 (hyperparameter sensitivity)

Outputs:
  results/latex_tables.tex        — all tables as LaTeX \begin{table}...\end{table} blocks
  results/latex_tables_summary.txt — console-friendly overview

Usage:
    cd AD-FTD
    python scripts/generate_tables.py --results results --out results/latex_tables.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BOLD_ADFTD = True   # Bold AD-FTD numbers in Table I / II

MODEL_ORDER_T1 = [
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


def _fmt(val: str, bold: bool = False) -> str:
    return r"\textbf{" + val + "}" if bold else val


def _load(path: Path) -> dict:
    if not path.exists():
        print(f"[WARN] {path} not found — skipping table", file=sys.stderr)
        return {}
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Table I — Main Performance Comparison
# ──────────────────────────────────────────────────────────────────────────────

def make_table1(data: dict) -> str:
    if not data:
        return "% Table I: data not available\n"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main Performance Comparison at $T{=}3$ (mean\,$\pm$\,std over 40 runs). "
        r"Best result per column is \textbf{bolded}. "
        r"PRC-AUC and macro F1 are primary metrics.}",
        r"\label{tab:main_results}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} "
        r"& \textbf{Accuracy} & \textbf{PRC-AUC} & \textbf{ROC-AUC} \\",
        r"\midrule",
    ]

    # Find best values (mean only) for bolding
    best: dict = {}
    for metric in ["precision", "recall", "f1", "accuracy", "prc_auc", "roc_auc"]:
        vals = []
        for mname, row in data.items():
            try:
                mean = float(row[metric].split("±")[0].strip())
                vals.append((mean, mname))
            except (KeyError, ValueError):
                pass
        if vals:
            best[metric] = max(vals, key=lambda x: x[0])[1]

    # Output in preferred order; fall back to dict order
    ordered = [m for m in MODEL_ORDER_T1 if m in data]
    ordered += [m for m in data if m not in ordered]

    section_break = {"TCN_BiLSTM": True, "TIME_LLM": True}
    prev_group = None

    for mname in ordered:
        row = data[mname]
        # Group separator
        if mname in section_break:
            lines.append(r"\midrule")
        display = mname.replace("_", r"\_")
        cells = []
        for metric in ["precision", "recall", "f1", "accuracy", "prc_auc", "roc_auc"]:
            val = row.get(metric, "—")
            b = BOLD_ADFTD and (mname == "AD-FTD") or (best.get(metric) == mname)
            cells.append(_fmt(val, bold=b))
        lines.append(f"  {display} & " + " & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Table II — Adversarial Robustness
# ──────────────────────────────────────────────────────────────────────────────

def make_table2(data: dict) -> str:
    if not data:
        return "% Table II: data not available\n"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Adversarial Robustness Comparison. "
        r"ARS$=$|F1$_\text{clean}$\,$-\,F1$_\text{adv}$| (lower is better). "
        r"Best ARS is \textbf{bolded}.}",
        r"\label{tab:adv_results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{F1 (Clean)} & \textbf{F1 (Adv.)} & \textbf{ARS} \\",
        r"\midrule",
    ]

    best_ars_mname = None
    best_ars_val   = float("inf")
    for mname, row in data.items():
        try:
            ars_mean = float(row["ars"].split("±")[0].strip())
            if ars_mean < best_ars_val:
                best_ars_val   = ars_mean
                best_ars_mname = mname
        except (ValueError, KeyError):
            pass

    ordered = [m for m in MODEL_ORDER_T1 if m in data]
    ordered += [m for m in data if m not in ordered]
    section_break = {"TCN_BiLSTM": True, "TIME_LLM": True}

    for mname in ordered:
        row = data[mname]
        if mname in section_break:
            lines.append(r"\midrule")
        display = mname.replace("_", r"\_")
        f1c = row.get("f1_clean", "—")
        f1a = row.get("f1_adv",   "—")
        ars = _fmt(row.get("ars", "—"), bold=(mname == best_ars_mname))
        lines.append(f"  {display} & {f1c} & {f1a} & {ars} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Table III — Ablation Study
# ──────────────────────────────────────────────────────────────────────────────

def make_table3(data: dict) -> str:
    if not data:
        return "% Table III: data not available\n"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation Study. Each row removes one component of AD-FTD.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} "
        r"& \textbf{PRC-AUC} & \textbf{ARS} \\",
        r"\midrule",
    ]

    ordered = [v for v in ABLATION_ORDER if v in data]
    ordered += [v for v in data if v not in ordered]

    for vname in ordered:
        row = data[vname]
        display = vname.replace("w/o", r"w/o")
        p  = row.get("precision", "—")
        r_ = row.get("recall",    "—")
        f1 = row.get("f1",        "—")
        pa = row.get("prc_auc",   "—")
        ar = row.get("ars",       "—")
        # Bold full AD-FTD
        if vname == "AD-FTD (full)":
            p, r_, f1, pa, ar = [_fmt(x, bold=True) for x in [p, r_, f1, pa, ar]]
        lines.append(f"  {display} & {p} & {r_} & {f1} & {pa} & {ar} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Table IV — Trajectory Length Sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def make_table4(data: dict) -> str:
    if not data:
        return "% Table IV: data not available\n"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Trajectory Length Sensitivity. AD-FTD performance for "
        r"$T \in \{1, 2, 3\}$.}",
        r"\label{tab:traj_length}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{$T$} & \textbf{F1} & \textbf{PRC-AUC} & \textbf{ROC-AUC} \\",
        r"\midrule",
    ]

    for key in sorted(data.keys(), key=lambda x: int(x.replace("T=", ""))):
        row = data[key]
        f1  = row.get("f1",      "—")
        pa  = row.get("prc_auc", "—")
        ra  = row.get("roc_auc", "—")
        lines.append(f"  {key} & {f1} & {pa} & {ra} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Table V — Industry Analysis
# ──────────────────────────────────────────────────────────────────────────────

def make_table5(data: dict) -> str:
    if not data:
        return "% Table V: data not available\n"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-Industry Generalization (Leave-One-Industry-Out). "
        r"$N_{\text{test}}$: held-out sector test size; $N_{\text{fraud}}$: fraud cases.}",
        r"\label{tab:industry}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lrrcccc}",
        r"\toprule",
        r"\textbf{Sector} & $N_{\text{test}}$ & $N_{\text{fraud}}$ "
        r"& \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{PRC-AUC} \\",
        r"\midrule",
    ]

    for sector_key in sorted(data.keys()):
        row = data[sector_key]
        nt  = row.get("n_test",   "—")
        nf  = row.get("n_fraud",  "—")
        p   = row.get("precision","—")
        r_  = row.get("recall",   "—")
        f1  = row.get("f1",       "—")
        pa  = row.get("prc_auc",  "—")
        display = sector_key.replace("–", r"--")
        lines.append(
            f"  {display} & {nt} & {nf} & {p} & {r_} & {f1} & {pa} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Table B1 — Hyperparameter Sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def make_tableB1(data: dict) -> str:
    if not data:
        return "% Table B1: data not available\n"

    # Collect unique alpha / beta values
    alphas = sorted({v["alpha"] for v in data.values()})
    betas  = sorted({v["beta"]  for v in data.values()})

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Hyperparameter Sensitivity: macro F1 (ARS) for different "
        r"$\alpha$ and $\beta$ values. Best F1 per row is \textbf{bolded}.}",
        r"\label{tab:hyperparam}",
    ]

    col_spec = "l" + "c" * len(betas)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = r"\textbf{$\alpha$ \textbackslash\ $\beta$}"
    for b in betas:
        header += f" & $\\beta={b}$"
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for alpha in alphas:
        # Find best F1 in this row
        row_f1s = []
        for beta in betas:
            key = f"a{alpha:.1f}_b{beta:.1f}"
            entry = data.get(key, {})
            try:
                mean = float(entry["f1"].split("±")[0].strip())
                row_f1s.append((mean, key))
            except (KeyError, ValueError):
                row_f1s.append((0.0, key))
        best_key = max(row_f1s, key=lambda x: x[0])[1]

        row_str = f"  $\\alpha={alpha}$"
        for beta in betas:
            key = f"a{alpha:.1f}_b{beta:.1f}"
            entry = data.get(key, {})
            f1v = entry.get("f1",  "—")
            arv = entry.get("ars", "—")
            cell = f"{f1v} ({arv})"
            if key == best_key:
                cell = _fmt(cell, bold=True)
            row_str += f" & {cell}"
        lines.append(row_str + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate LaTeX tables from results JSON files")
    p.add_argument("--results", default="results",
                   help="Directory containing *_results.json and table*.json files")
    p.add_argument("--out", default="results/latex_tables.tex",
                   help="Output .tex file with all table blocks")
    return p.parse_args()


def main():
    args = parse_args()
    rdir = Path(args.results)

    t1  = _load(rdir / "table1_main.json")
    t2  = _load(rdir / "table2_adv.json")
    t3  = _load(rdir / "table3_ablation.json")
    t4  = _load(rdir / "table4_traj_length.json")
    t5  = _load(rdir / "table5_industry.json")
    tb1 = _load(rdir / "tableB1_hyperparam.json")

    header = (
        "% ============================================================\n"
        "% Auto-generated LaTeX tables — AD-FTD paper\n"
        "% Run scripts/generate_tables.py to regenerate\n"
        "% ============================================================\n\n"
    )

    blocks = [
        "% ─── Table I ─────────────────────────────────────────────────\n" + make_table1(t1),
        "% ─── Table II ────────────────────────────────────────────────\n" + make_table2(t2),
        "% ─── Table III ───────────────────────────────────────────────\n" + make_table3(t3),
        "% ─── Table IV ────────────────────────────────────────────────\n" + make_table4(t4),
        "% ─── Table V ─────────────────────────────────────────────────\n" + make_table5(t5),
        "% ─── Table B1 ────────────────────────────────────────────────\n" + make_tableB1(tb1),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header)
        for block in blocks:
            f.write(block + "\n")

    print(f"LaTeX tables written to: {out_path}")

    # Console summary
    print("\n── Available tables ──")
    summaries = [
        ("Table I   (main)", t1),
        ("Table II  (adv.)", t2),
        ("Table III (abl.)", t3),
        ("Table IV  (traj)", t4),
        ("Table V   (ind.)", t5),
        ("Table B1  (hpar)", tb1),
    ]
    for label, d in summaries:
        status = "OK" if d else "MISSING"
        nrows = len(d) if d else 0
        print(f"  {label}: {status}  ({nrows} rows)")


if __name__ == "__main__":
    main()
