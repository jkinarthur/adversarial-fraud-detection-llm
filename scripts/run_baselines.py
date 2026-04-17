"""
Run all baseline models and AD-FTD for Table I and Table II.

For each model the script runs 4-fold CV × 10 resampling and records:
  - clean metrics  : Precision, Recall, F1, Accuracy, PRC-AUC, ROC-AUC
  - adversarial F1 : F1 under FGSM perturbation (ε = 0.01)
  - ARS            : |F1_clean - F1_adv|

Outputs:
  results/baselines_results.json   — raw per-run numbers
  results/table1_main.json         — mean ± std per model
  results/table2_adv.json          — adversarial comparison

Usage:
    cd AD-FTD
    python scripts/run_baselines.py \
        --features data/features/samples.pkl \
        --traj_len 3 \
        --n_folds 4 \
        --n_resample 10 \
        --device cuda \
        --out results
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Run all baselines + AD-FTD")
    p.add_argument("--features", default="data/features/samples.pkl",
                   help="Preprocessed samples pickle")
    p.add_argument("--traj_len", type=int, default=3)
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--n_resample", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.01,
                   help="FGSM perturbation magnitude for adversarial eval")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results",
                   help="Directory for JSON output files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_adftd", action="store_true",
                   help="Skip AD-FTD (use if already trained separately)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass


def fgsm_attack_numpy(X: np.ndarray, y: np.ndarray, predict_fn,
                      epsilon: float = 0.01) -> np.ndarray:
    """
    Finite-difference FGSM approximation for non-differentiable / sklearn models.
    Adds ε * sign(numerical_gradient) to each feature.
    """
    X_adv = X.copy()
    probs = predict_fn(X)
    for j in range(X.shape[1]):
        Xp = X.copy(); Xp[:, j] += 1e-4
        Xm = X.copy(); Xm[:, j] -= 1e-4
        grad_j = (predict_fn(Xp) - predict_fn(Xm)) / (2e-4)
        # Correct grad direction: want to increase loss (BCE) for fraud cases
        sign_j = np.sign(grad_j * (2 * y - 1))
        X_adv[:, j] += epsilon * sign_j
    return X_adv


def evaluate_baseline_fold(model_wrapper, X_train, y_train, X_val, y_val,
                            epsilon: float = 0.01):
    """Fit on train, evaluate clean + adversarial on val."""
    from adftd.evaluation.metrics import compute_metrics, compute_ars

    model_wrapper.fit(X_train, y_train)

    # Clean
    probs_clean = np.clip(model_wrapper.predict_proba(X_val), 1e-7, 1 - 1e-7)
    metrics_clean = compute_metrics(y_val, probs_clean)

    # Adversarial (finite-difference for non-differentiable; torch FGSM for nn)
    X_adv = fgsm_attack_numpy(X_val, y_val, model_wrapper.predict_proba, epsilon)
    probs_adv = np.clip(model_wrapper.predict_proba(X_adv), 1e-7, 1 - 1e-7)
    metrics_adv = compute_metrics(y_val, probs_adv)

    ars = compute_ars(metrics_clean["f1"], metrics_adv["f1"])
    return {
        "clean": metrics_clean,
        "adv": metrics_adv,
        "ars": ars,
    }


def run_cv(samples, build_fn, traj_len, n_folds, n_resample, epsilon,
           seed=42):
    """
    4-fold CV × n_resample resampling for a given model builder.
    Returns list of result dicts (one per fold-resample pair).
    """
    from sklearn.model_selection import StratifiedKFold  # type: ignore
    from adftd.models.baselines import flatten_samples

    X, y = flatten_samples(samples, traj_len=traj_len)
    all_results = []

    for rs in range(n_resample):
        set_seed(seed + rs)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rs)
        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = build_fn()
            result = evaluate_baseline_fold(model, X_tr, y_tr, X_val, y_val,
                                            epsilon=epsilon)
            result["resample"] = rs
            result["fold"] = fold_idx
            all_results.append(result)
            logger.debug("rs=%d fold=%d  F1=%.4f  ARS=%.4f",
                         rs, fold_idx, result["clean"]["f1"], result["ars"])

    return all_results


def aggregate(results: list) -> dict:
    """Compute mean ± std across all runs."""
    metric_keys = list(results[0]["clean"].keys())
    out = {}
    for k in metric_keys:
        vals = [r["clean"][k] for r in results]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    adv_f1 = [r["adv"]["f1"] for r in results]
    ars_vals = [r["ars"] for r in results]
    out["adv_f1"] = {"mean": float(np.mean(adv_f1)), "std": float(np.std(adv_f1))}
    out["ars"] = {"mean": float(np.mean(ars_vals)), "std": float(np.std(ars_vals))}
    return out


# ──────────────────────────────────────────────────────────────────────────────
# AD-FTD evaluation (runs full trainer, extracts clean + adv metrics)
# ──────────────────────────────────────────────────────────────────────────────

def run_adftd_cv(samples, traj_len, n_folds, n_resample, epsilon,
                 device, seed=42):
    """Run AD-FTD via ADFTDTrainer and collect metrics compatible with baselines."""
    import torch
    from adftd.config import ADFTDConfig
    from adftd.data.dataset import FraudTrajectoryDataset
    from adftd.training.trainer import ADFTDTrainer, set_seed as trainer_seed
    from adftd.evaluation.metrics import compute_metrics, compute_ars
    from adftd.training.fgsm import fgsm_perturb
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import StratifiedKFold  # type: ignore

    cfg = ADFTDConfig()
    cfg.data.traj_len = traj_len
    cfg.train.n_folds = n_folds
    cfg.train.n_resample = n_resample
    cfg.train.device = device
    cfg.train.epochs = 50
    cfg.train.early_stop_patience = 10

    traj_dim = samples[0]["trajectory"].shape[-1]
    fin_dim = samples[0]["financial_ratios"].shape[0]
    cfg.model.trajectory_dim = traj_dim
    cfg.model.financial_dim = fin_dim

    trainer = ADFTDTrainer(cfg)
    all_results = []
    labels_all = [s["label"] for s in samples]

    for rs in range(n_resample):
        trainer_seed(seed + rs)
        dataset = FraudTrajectoryDataset(
            samples, traj_len=traj_len, augment=True,
            oversample_ratio=cfg.data.oversample_ratio,
        )
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rs)
        raw_labels = np.array([s["label"] for s in samples])
        for fold_idx, (tr_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(samples)), raw_labels)
        ):
            train_ds = Subset(dataset, tr_idx.tolist())
            val_ds   = Subset(dataset, val_idx.tolist())
            train_loader = DataLoader(
                train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                collate_fn=FraudTrajectoryDataset.collate_fn,
            )
            val_loader = DataLoader(
                val_ds, batch_size=64, shuffle=False,
                collate_fn=FraudTrajectoryDataset.collate_fn,
            )
            model, _ = trainer.train_fold(train_loader, val_loader, fold_id=fold_idx)
            model.eval()
            dev = trainer.device

            # Collect predictions
            y_true, y_clean, y_adv_list = [], [], []
            for batch in val_loader:
                z = batch["z"].to(dev)
                zp = batch["z_peer"].to(dev)
                r  = batch["r"].to(dev)
                yb = batch["y"].numpy()

                z.requires_grad_(True)
                logits, _, _ = model(z, zp, r)
                loss = torch.nn.BCEWithLogitsLoss()(
                    logits, batch["y"].to(dev)
                )
                loss.backward()
                z_adv = fgsm_perturb(z, z.grad, epsilon)

                with torch.no_grad():
                    p_clean = torch.sigmoid(model(z.detach(), zp, r)[0])
                    p_adv   = torch.sigmoid(model(z_adv.detach(), zp, r)[0])

                y_true.extend(yb.tolist())
                y_clean.extend(p_clean.cpu().numpy().tolist())
                y_adv_list.extend(p_adv.cpu().numpy().tolist())

            y_true  = np.array(y_true)
            y_clean = np.array(y_clean)
            y_adv   = np.array(y_adv_list)

            mc = compute_metrics(y_true, y_clean)
            ma = compute_metrics(y_true, y_adv)
            ars = compute_ars(mc["f1"], ma["f1"])

            all_results.append({
                "clean": mc, "adv": ma, "ars": ars,
                "resample": rs, "fold": fold_idx,
            })
            logger.info("AD-FTD  rs=%d fold=%d  F1=%.4f  ARS=%.4f",
                        rs, fold_idx, mc["f1"], ars)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.features, "rb") as f:
        samples = pickle.load(f)
    logger.info("Loaded %d samples", len(samples))

    from adftd.models.baselines import build_baseline, flatten_samples, BASELINE_NAMES
    X_full, _ = flatten_samples(samples, traj_len=args.traj_len)
    input_dim = X_full.shape[1]
    logger.info("Flat feature dim: %d", input_dim)

    all_model_results: dict = {}

    # ── Baselines ──────────────────────────────────────────────────────────
    for bname in BASELINE_NAMES:
        logger.info("=== %s ===", bname)
        key = bname.lower().replace("-", "_").replace("+", "_")
        results = run_cv(
            samples,
            build_fn=lambda b=bname: build_baseline(b, input_dim, args.device, args.traj_len),
            traj_len=args.traj_len,
            n_folds=args.n_folds,
            n_resample=args.n_resample,
            epsilon=args.epsilon,
            seed=args.seed,
        )
        all_model_results[bname] = {
            "runs": results,
            "summary": aggregate(results),
        }
        logger.info("  F1 = %.4f ± %.4f",
                    all_model_results[bname]["summary"]["f1"]["mean"],
                    all_model_results[bname]["summary"]["f1"]["std"])

    # ── AD-FTD ────────────────────────────────────────────────────────────
    if not args.skip_adftd:
        logger.info("=== AD-FTD ===")
        adftd_results = run_adftd_cv(
            samples,
            traj_len=args.traj_len,
            n_folds=args.n_folds,
            n_resample=args.n_resample,
            epsilon=args.epsilon,
            device=args.device,
            seed=args.seed,
        )
        all_model_results["AD-FTD"] = {
            "runs": adftd_results,
            "summary": aggregate(adftd_results),
        }
        logger.info("  AD-FTD F1 = %.4f ± %.4f",
                    all_model_results["AD-FTD"]["summary"]["f1"]["mean"],
                    all_model_results["AD-FTD"]["summary"]["f1"]["std"])

    # ── Save raw results ───────────────────────────────────────────────────
    raw_path = out_dir / "baselines_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_model_results, f, indent=2)
    logger.info("Raw results saved to %s", raw_path)

    # ── Build Table I summary ──────────────────────────────────────────────
    table1 = {}
    for mname, data in all_model_results.items():
        s = data["summary"]
        table1[mname] = {
            metric: f'{s[metric]["mean"]:.4f} ± {s[metric]["std"]:.4f}'
            for metric in ["precision", "recall", "f1", "accuracy",
                           "prc_auc", "roc_auc"]
            if metric in s
        }
    with open(out_dir / "table1_main.json", "w") as f:
        json.dump(table1, f, indent=2)

    # ── Build Table II summary ─────────────────────────────────────────────
    table2 = {}
    for mname, data in all_model_results.items():
        s = data["summary"]
        table2[mname] = {
            "f1_clean": f'{s["f1"]["mean"]:.4f} ± {s["f1"]["std"]:.4f}',
            "f1_adv":   f'{s["adv_f1"]["mean"]:.4f} ± {s["adv_f1"]["std"]:.4f}',
            "ars":      f'{s["ars"]["mean"]:.4f} ± {s["ars"]["std"]:.4f}',
        }
    with open(out_dir / "table2_adv.json", "w") as f:
        json.dump(table2, f, indent=2)

    logger.info("Table I  -> %s", out_dir / "table1_main.json")
    logger.info("Table II -> %s", out_dir / "table2_adv.json")


if __name__ == "__main__":
    main()
