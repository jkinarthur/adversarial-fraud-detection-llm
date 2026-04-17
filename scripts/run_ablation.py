"""
Ablation study script  →  Table III of the paper.

Four variants evaluated with 4-fold CV × 10 resampling:
  (a) Full AD-FTD
  (b) AD-FTD w/o counterfactual  — detector sees [Z ‖ Z ‖ 0] (peer mean instead of Ẑ)
  (c) AD-FTD w/o deviation       — detector sees Z only (no ΔZ, no Ẑ)
  (d) AD-FTD w/o FGSM            — standard training, no adversarial loss

Outputs:
  results/ablation_results.json   — per-run numbers
  results/table3_ablation.json    — mean ± std

Usage:
    cd AD-FTD
    python scripts/run_ablation.py \
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
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Ablation study for AD-FTD")
    p.add_argument("--features", default="data/features/samples.pkl")
    p.add_argument("--traj_len", type=int, default=3)
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--n_resample", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Ablation model wrappers
# ──────────────────────────────────────────────────────────────────────────────

class AblationADFTD(nn.Module):
    """
    Configurable AD-FTD variant for ablation.

    Flags:
        use_counterfactual : if False, Ẑ = Z_peer (no generator learned)
        use_deviation      : if False, concatenate only Z (no Ẑ, no ΔZ)
    """

    def __init__(self, cfg, use_counterfactual: bool = True,
                 use_deviation: bool = True) -> None:
        super().__init__()
        from adftd.models.counterfactual import CounterfactualGenerator
        from adftd.models.detector import FraudDetector

        self.use_cf = use_counterfactual
        self.use_dev = use_deviation
        traj_dim = cfg.trajectory_dim
        fin_dim  = cfg.financial_dim

        if use_counterfactual:
            self.generator = CounterfactualGenerator(
                traj_dim=traj_dim, financial_dim=fin_dim,
                tcn_hidden=cfg.tcn_hidden, tcn_levels=cfg.tcn_levels,
                tcn_kernel=cfg.tcn_kernel_size, tcn_dropout=cfg.tcn_dropout,
            )

        # Detector input dimension depends on ablation
        if use_deviation:
            det_input = traj_dim * 3   # [Z ‖ Ẑ ‖ ΔZ]
        else:
            det_input = traj_dim       # Z only

        self.detector = FraudDetector(
            traj_dim=det_input,
            tcn_hidden=cfg.tcn_hidden, tcn_levels=cfg.tcn_levels,
            tcn_kernel=cfg.tcn_kernel_size, tcn_dropout=cfg.tcn_dropout,
            bilstm_hidden=cfg.bilstm_hidden, bilstm_layers=cfg.bilstm_layers,
            bilstm_dropout=cfg.bilstm_dropout,
        )
        self.traj_dim = traj_dim

    def forward(self, z: Tensor, z_peer: Tensor, r: Tensor
                ) -> Tuple[Tensor, Tensor, Tensor]:
        z_last = z[:, -1, :]   # (B, D)

        if self.use_cf:
            z_hat = self.generator(z, r, z_peer)
        else:
            z_hat = z_peer                     # no learned counterfactual

        delta_z = z_last - z_hat

        if self.use_dev:
            z_joint = torch.cat([z_last, z_hat, delta_z], dim=-1)
        else:
            z_joint = z_last                   # trajectory only

        logits = self.detector(z_joint)
        return logits, z_hat, delta_z


# ──────────────────────────────────────────────────────────────────────────────
# Training loop for ablation variants
# ──────────────────────────────────────────────────────────────────────────────

def train_ablation_fold(
    model: AblationADFTD,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg,
    use_fgsm: bool = True,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.01,
) -> Tuple[AblationADFTD, Dict]:
    from adftd.training.losses import ADFTDLoss
    from adftd.training.fgsm import fgsm_perturb
    from adftd.training.trainer import EarlyStopping

    labels = [b["y"].mean().item() for b in train_loader]
    fraud_rate = float(np.mean(labels)) if labels else 0.01
    pos_w = (1.0 - fraud_rate) / max(fraud_rate, 1e-6)

    alpha = cfg.train.alpha if use_fgsm else 0.0
    beta  = cfg.train.beta  if use_fgsm else 0.0
    criterion = ADFTDLoss(alpha=alpha, beta=beta, pos_weight=pos_w)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience)
    best_state = None
    best_val = float("inf")

    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            z    = batch["z"].to(device)
            zp   = batch["z_peer"].to(device)
            r_   = batch["r"].to(device)
            y    = batch["y"].to(device)

            z.requires_grad_(True)
            optimizer.zero_grad()

            logits, z_hat, delta_z = model(z, zp, r_)
            l_cls = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_w], device=device)
            )(logits, y)
            l_cls.backward(retain_graph=True)

            if use_fgsm and z.grad is not None:
                z_adv = fgsm_perturb(z.detach(), z.grad.detach(), epsilon)
                logits_adv, _, _ = model(z_adv, zp, r_)
            else:
                logits_adv = logits

            optimizer.zero_grad()
            total_loss, *_ = criterion(
                logits, z_hat, delta_z, y, logits_adv=logits_adv
            )
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                z  = batch["z"].to(device)
                zp = batch["z_peer"].to(device)
                r_ = batch["r"].to(device)
                y  = batch["y"].to(device)
                logits, z_hat, delta_z = model(z, zp, r_)
                vl, *_ = criterion(logits, z_hat, delta_z, y,
                                   logits_adv=logits)
                val_losses.append(vl.item())

        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in
                          model.state_dict().items()}
        if stopper.step(val_loss):
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def collect_predictions(model, val_loader, device, epsilon):
    """Collect clean + adversarial predictions from a trained AblationADFTD."""
    from adftd.training.fgsm import fgsm_perturb

    y_true, p_clean, p_adv = [], [], []
    for batch in val_loader:
        z  = batch["z"].to(device)
        zp = batch["z_peer"].to(device)
        r_ = batch["r"].to(device)
        yb = batch["y"]

        z.requires_grad_(True)
        logits, _, _ = model(z, zp, r_)
        loss = nn.BCEWithLogitsLoss()(logits, yb.to(device))
        loss.backward()
        z_adv = fgsm_perturb(z.detach(), z.grad.detach(), epsilon)

        with torch.no_grad():
            pc = torch.sigmoid(model(z.detach(), zp, r_)[0]).cpu().numpy()
            pa = torch.sigmoid(model(z_adv, zp, r_)[0]).cpu().numpy()

        y_true.extend(yb.numpy().tolist())
        p_clean.extend(pc.tolist())
        p_adv.extend(pa.tolist())

    return np.array(y_true), np.array(p_clean), np.array(p_adv)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

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
    from adftd.data.dataset import FraudTrajectoryDataset
    from adftd.evaluation.metrics import compute_metrics, compute_ars
    from sklearn.model_selection import StratifiedKFold  # type: ignore

    cfg = ADFTDConfig()
    cfg.data.traj_len   = args.traj_len
    cfg.train.epochs    = args.epochs
    cfg.train.device    = args.device
    cfg.model.trajectory_dim = samples[0]["trajectory"].shape[-1]
    cfg.model.financial_dim  = samples[0]["financial_ratios"].shape[0]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # (name, use_cf, use_dev, use_fgsm)
    variants = [
        ("AD-FTD (full)",             True,  True,  True),
        ("w/o counterfactual",        False, True,  True),
        ("w/o deviation encoding",    True,  False, True),
        ("w/o adversarial training",  True,  True,  False),
    ]

    all_results: dict = {}
    raw_labels = np.array([s["label"] for s in samples])

    for vname, use_cf, use_dev, use_fgsm in variants:
        logger.info("=== %s ===", vname)
        runs = []

        for rs in range(args.n_resample):
            torch.manual_seed(args.seed + rs)
            np.random.seed(args.seed + rs)

            dataset = FraudTrajectoryDataset(
                samples, traj_len=args.traj_len, augment=True,
                oversample_ratio=cfg.data.oversample_ratio,
            )
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True,
                                  random_state=args.seed + rs)
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

                model = AblationADFTD(cfg.model, use_cf, use_dev).to(device)
                model = train_ablation_fold(
                    model, train_loader, val_loader, cfg,
                    use_fgsm=use_fgsm, device=device, epsilon=args.epsilon,
                )

                y_true, p_clean, p_adv = collect_predictions(
                    model, val_loader, device, args.epsilon
                )
                mc = compute_metrics(y_true, p_clean)
                ma = compute_metrics(y_true, p_adv)
                ars = compute_ars(mc["f1"], ma["f1"])
                runs.append({
                    "clean": mc, "adv": ma, "ars": ars,
                    "resample": rs, "fold": fold_idx,
                })
                logger.info("  %s rs=%d fold=%d F1=%.4f", vname, rs, fold_idx, mc["f1"])

        # Aggregate
        f1s   = [r["clean"]["f1"]     for r in runs]
        prauc = [r["clean"]["prc_auc"] for r in runs]
        roc   = [r["clean"]["roc_auc"] for r in runs]
        prec  = [r["clean"]["precision"] for r in runs]
        rec   = [r["clean"]["recall"]   for r in runs]
        ars_v = [r["ars"]              for r in runs]
        all_results[vname] = {
            "runs": runs,
            "summary": {
                "f1":        {"mean": float(np.mean(f1s)),   "std": float(np.std(f1s))},
                "prc_auc":   {"mean": float(np.mean(prauc)), "std": float(np.std(prauc))},
                "roc_auc":   {"mean": float(np.mean(roc)),   "std": float(np.std(roc))},
                "precision": {"mean": float(np.mean(prec)),  "std": float(np.std(prec))},
                "recall":    {"mean": float(np.mean(rec)),   "std": float(np.std(rec))},
                "ars":       {"mean": float(np.mean(ars_v)), "std": float(np.std(ars_v))},
            },
        }

    # Save
    raw_path = out_dir / "ablation_results.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved %s", raw_path)

    # Table III summary
    table3 = {}
    for vname, data in all_results.items():
        s = data["summary"]
        table3[vname] = {
            m: f'{s[m]["mean"]:.4f} ± {s[m]["std"]:.4f}'
            for m in ["precision", "recall", "f1", "prc_auc", "roc_auc", "ars"]
        }
    with open(out_dir / "table3_ablation.json", "w") as f:
        json.dump(table3, f, indent=2)
    logger.info("Table III -> %s", out_dir / "table3_ablation.json")


if __name__ == "__main__":
    main()
