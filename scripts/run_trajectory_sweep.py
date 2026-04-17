"""
Trajectory length sensitivity sweep  →  Table IV of the paper.

Trains and evaluates AD-FTD for T ∈ {1, 2, 3} with 4-fold CV × 10 resampling.
(The paper also references T=4,5 in the appendix; add more values to --traj_lens
if the pre-processed samples were built with longer sequences.)

Outputs:
  results/traj_sweep_results.json   — per-run numbers per T
  results/table4_traj_length.json   — mean ± std summary

Usage:
    cd AD-FTD
    python scripts/run_trajectory_sweep.py \
        --features data/features/samples.pkl \
        --traj_lens 1 2 3 \
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Trajectory length sweep")
    p.add_argument("--features", default="data/features/samples.pkl")
    p.add_argument("--traj_lens", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--n_resample", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def truncate_trajectory(samples, T: int) -> list:
    """
    Return a view of samples with trajectories truncated/padded to T steps.
    If stored trajectory has fewer steps than T, we pad with zeros on the left.
    """
    out = []
    for s in samples:
        traj = np.array(s["trajectory"], dtype=np.float32)
        stored_T = traj.shape[0]
        if stored_T >= T:
            traj_t = traj[-T:]          # take last T steps
        else:
            # Pad on the left
            pad = np.zeros((T - stored_T, traj.shape[1]), dtype=np.float32)
            traj_t = np.concatenate([pad, traj], axis=0)
        new_s = dict(s); new_s["trajectory"] = traj_t
        out.append(new_s)
    return out


def run_adftd_for_T(samples, T, cfg, n_folds, n_resample, epsilon, device, seed):
    """Run AD-FTD CV loop for a specific trajectory length T."""
    from adftd.data.dataset import FraudTrajectoryDataset
    from adftd.models.adftd import ADFTD
    from adftd.training.losses import ADFTDLoss
    from adftd.training.fgsm import fgsm_perturb
    from adftd.training.trainer import EarlyStopping
    from adftd.evaluation.metrics import compute_metrics, compute_ars
    from sklearn.model_selection import StratifiedKFold  # type: ignore

    samples_T = truncate_trajectory(samples, T)
    cfg.data.traj_len = T
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    raw_labels = np.array([s["label"] for s in samples_T])
    all_runs = []

    for rs in range(n_resample):
        torch.manual_seed(seed + rs)
        np.random.seed(seed + rs)

        dataset = FraudTrajectoryDataset(
            samples_T, traj_len=T, augment=True,
            oversample_ratio=cfg.data.oversample_ratio,
        )
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rs)
        for fold_idx, (tr_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(samples_T)), raw_labels)
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

            model = ADFTD.from_config(cfg.model).to(dev)

            labels_flat = [b["y"].mean().item() for b in train_loader]
            fraud_rate = float(np.mean(labels_flat)) if labels_flat else 0.01
            pos_w = (1.0 - fraud_rate) / max(fraud_rate, 1e-6)
            criterion = ADFTDLoss(alpha=cfg.train.alpha, beta=cfg.train.beta,
                                  pos_weight=pos_w)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                         weight_decay=cfg.train.weight_decay)
            stopper = EarlyStopping(patience=cfg.train.early_stop_patience)
            best_state = None; best_val = float("inf")

            for epoch in range(cfg.train.epochs):
                model.train()
                for batch in train_loader:
                    z  = batch["z"].to(dev); zp = batch["z_peer"].to(dev)
                    r_ = batch["r"].to(dev); y  = batch["y"].to(dev)
                    z.requires_grad_(True); optimizer.zero_grad()
                    logits, z_hat, delta_z = model(z, zp, r_)
                    l_cls = nn.BCEWithLogitsLoss(
                        pos_weight=torch.tensor([pos_w], device=dev)
                    )(logits, y)
                    l_cls.backward(retain_graph=True)
                    z_adv = fgsm_perturb(z.detach(), z.grad.detach(), epsilon)
                    logits_adv, _, _ = model(z_adv, zp, r_)
                    optimizer.zero_grad()
                    total, *_ = criterion(logits, z_hat, delta_z, y,
                                          logits_adv=logits_adv)
                    total.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                vl_list = []
                with torch.no_grad():
                    for batch in val_loader:
                        z  = batch["z"].to(dev); zp = batch["z_peer"].to(dev)
                        r_ = batch["r"].to(dev); y  = batch["y"].to(dev)
                        logits, z_hat, dz = model(z, zp, r_)
                        vl, *_ = criterion(logits, z_hat, dz, y, logits_adv=logits)
                        vl_list.append(vl.item())
                vl_mean = float(np.mean(vl_list))
                if vl_mean < best_val:
                    best_val = vl_mean
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                if stopper.step(vl_mean):
                    break

            if best_state:
                model.load_state_dict(best_state)
            model.eval()

            y_true, p_c, p_a = [], [], []
            for batch in val_loader:
                z  = batch["z"].to(dev); zp = batch["z_peer"].to(dev)
                r_ = batch["r"].to(dev)
                z.requires_grad_(True)
                logits, _, _ = model(z, zp, r_)
                loss = nn.BCEWithLogitsLoss()(logits, batch["y"].to(dev))
                loss.backward()
                z_adv = fgsm_perturb(z.detach(), z.grad.detach(), epsilon)
                with torch.no_grad():
                    pc = torch.sigmoid(model(z.detach(), zp, r_)[0]).cpu().numpy()
                    pa = torch.sigmoid(model(z_adv, zp, r_)[0]).cpu().numpy()
                y_true.extend(batch["y"].numpy().tolist())
                p_c.extend(pc.tolist()); p_a.extend(pa.tolist())

            y_true = np.array(y_true)
            mc = compute_metrics(y_true, np.array(p_c))
            ma = compute_metrics(y_true, np.array(p_a))
            all_runs.append({
                "clean": mc, "adv": ma,
                "ars": compute_ars(mc["f1"], ma["f1"]),
                "resample": rs, "fold": fold_idx,
            })
            logger.info("T=%d rs=%d fold=%d F1=%.4f PRC-AUC=%.4f",
                        T, rs, fold_idx, mc["f1"], mc["prc_auc"])

    return all_runs


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.features, "rb") as f:
        samples = pickle.load(f)
    logger.info("Loaded %d samples", len(samples))

    from adftd.config import ADFTDConfig
    cfg = ADFTDConfig()
    cfg.train.epochs = args.epochs
    cfg.train.device = args.device
    cfg.model.trajectory_dim = samples[0]["trajectory"].shape[-1]
    cfg.model.financial_dim  = samples[0]["financial_ratios"].shape[0]

    all_results: dict = {}

    for T in args.traj_lens:
        logger.info("==== T = %d ====", T)
        runs = run_adftd_for_T(
            samples, T, cfg,
            n_folds=args.n_folds, n_resample=args.n_resample,
            epsilon=args.epsilon, device=args.device, seed=args.seed,
        )
        f1s   = [r["clean"]["f1"]     for r in runs]
        prauc = [r["clean"]["prc_auc"] for r in runs]
        roc   = [r["clean"]["roc_auc"] for r in runs]
        all_results[str(T)] = {
            "runs": runs,
            "summary": {
                "f1":      {"mean": float(np.mean(f1s)),   "std": float(np.std(f1s))},
                "prc_auc": {"mean": float(np.mean(prauc)), "std": float(np.std(prauc))},
                "roc_auc": {"mean": float(np.mean(roc)),   "std": float(np.std(roc))},
            },
        }
        logger.info("T=%d  F1=%.4f ± %.4f  PRC-AUC=%.4f ± %.4f",
                    T, np.mean(f1s), np.std(f1s),
                    np.mean(prauc), np.std(prauc))

    # Save raw
    with open(out_dir / "traj_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Table IV
    table4 = {
        f"T={T}": {
            "f1":      f'{all_results[str(T)]["summary"]["f1"]["mean"]:.4f} ± {all_results[str(T)]["summary"]["f1"]["std"]:.4f}',
            "prc_auc": f'{all_results[str(T)]["summary"]["prc_auc"]["mean"]:.4f} ± {all_results[str(T)]["summary"]["prc_auc"]["std"]:.4f}',
            "roc_auc": f'{all_results[str(T)]["summary"]["roc_auc"]["mean"]:.4f} ± {all_results[str(T)]["summary"]["roc_auc"]["std"]:.4f}',
        }
        for T in args.traj_lens
    }
    with open(out_dir / "table4_traj_length.json", "w") as f:
        json.dump(table4, f, indent=2)

    logger.info("Table IV -> %s", out_dir / "table4_traj_length.json")


if __name__ == "__main__":
    main()
