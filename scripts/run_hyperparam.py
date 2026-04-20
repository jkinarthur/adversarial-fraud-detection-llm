"""
Hyperparameter sensitivity grid search  →  Table B1 (Appendix B) of the paper.

Grid:
  α (deviation weight) ∈ {0.1, 0.3, 0.5, 0.7}
  β (adversarial weight) ∈ {0.1, 0.3, 0.5}

For each (α, β) pair: 4-fold CV × 5 resampling (reduced for speed).
Reports macro F1 and ARS.

Outputs:
  results/hyperparam_results.json    — per-config per-run numbers
  results/tableB1_hyperparam.json    — mean ± std summary grid

Usage:
    cd AD-FTD
    python scripts/run_hyperparam.py \
        --features data/features/samples.pkl \
        --traj_len 3 \
        --n_folds 4 \
        --n_resample 5 \
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
from itertools import product
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
    p = argparse.ArgumentParser(description="Hyperparameter sensitivity sweep")
    p.add_argument("--features", default="data/features/samples.pkl")
    p.add_argument("--traj_len", type=int, default=3)
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--n_resample", type=int, default=5)
    p.add_argument("--alpha_vals", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5, 0.7])
    p.add_argument("--beta_vals",  type=float, nargs="+",
                   default=[0.1, 0.3, 0.5])
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--s3_bucket", default=None,
                   help="S3 bucket for result upload")
    p.add_argument("--s3_prefix", default="adftd/results",
                   help="S3 key prefix")
    p.add_argument("--region", default="us-east-1")
    return p.parse_args()


def run_config(samples, alpha, beta, cfg_base, n_folds, n_resample,
               epsilon, device, seed):
    """Run 4-fold CV × n_resample for a specific (alpha, beta) pair."""
    from adftd.config import ADFTDConfig
    from adftd.data.dataset import FraudTrajectoryDataset
    from adftd.models.adftd import ADFTD
    from adftd.training.losses import ADFTDLoss
    from adftd.training.fgsm import fgsm_perturb
    from adftd.training.trainer import EarlyStopping
    from adftd.evaluation.metrics import compute_metrics, compute_ars
    from sklearn.model_selection import StratifiedKFold  # type: ignore

    cfg = ADFTDConfig()
    cfg.model.trajectory_dim = cfg_base.model.trajectory_dim
    cfg.model.financial_dim  = cfg_base.model.financial_dim
    cfg.data.traj_len        = cfg_base.data.traj_len
    cfg.train.epochs         = cfg_base.train.epochs
    cfg.train.alpha          = alpha
    cfg.train.beta           = beta
    cfg.train.device         = device

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    raw_labels = np.array([s["label"] for s in samples])
    all_runs = []

    for rs in range(n_resample):
        torch.manual_seed(seed + rs); np.random.seed(seed + rs)
        dataset = FraudTrajectoryDataset(
            samples, traj_len=cfg.data.traj_len, augment=True,
            oversample_ratio=cfg.data.oversample_ratio,
        )
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=seed + rs)
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

            model = ADFTD.from_config(cfg.model).to(dev)
            labels_flat = [b["y"].mean().item() for b in train_loader]
            fraud_rate  = float(np.mean(labels_flat)) if labels_flat else 0.01
            pos_w = (1.0 - fraud_rate) / max(fraud_rate, 1e-6)
            criterion = ADFTDLoss(alpha=alpha, beta=beta, pos_weight=pos_w)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                         weight_decay=cfg.train.weight_decay)
            stopper   = EarlyStopping(patience=cfg.train.early_stop_patience)
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
            ars = compute_ars(mc["f1"], ma["f1"])
            all_runs.append({
                "clean": mc, "adv": ma, "ars": ars,
                "resample": rs, "fold": fold_idx,
            })
            logger.debug("α=%.1f β=%.1f rs=%d fold=%d F1=%.4f ARS=%.4f",
                         alpha, beta, rs, fold_idx, mc["f1"], ars)

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
    cfg_base = ADFTDConfig()
    cfg_base.data.traj_len        = args.traj_len
    cfg_base.train.epochs         = args.epochs
    cfg_base.train.device         = args.device
    cfg_base.model.trajectory_dim = samples[0]["trajectory"].shape[-1]
    cfg_base.model.financial_dim  = samples[0]["financial_ratios"].shape[0]

    all_results: dict = {}

    for alpha, beta in product(args.alpha_vals, args.beta_vals):
        key = f"a{alpha:.1f}_b{beta:.1f}"
        logger.info("=== α=%.1f  β=%.1f ===", alpha, beta)
        runs = run_config(
            samples, alpha, beta, cfg_base,
            n_folds=args.n_folds, n_resample=args.n_resample,
            epsilon=args.epsilon, device=args.device, seed=args.seed,
        )
        f1s  = [r["clean"]["f1"]  for r in runs]
        ars_v = [r["ars"]          for r in runs]
        all_results[key] = {
            "alpha": alpha, "beta": beta,
            "runs": runs,
            "summary": {
                "f1":  {"mean": float(np.mean(f1s)),   "std": float(np.std(f1s))},
                "ars": {"mean": float(np.mean(ars_v)), "std": float(np.std(ars_v))},
            },
        }
        logger.info("  F1=%.4f ± %.4f  ARS=%.4f ± %.4f",
                    np.mean(f1s), np.std(f1s), np.mean(ars_v), np.std(ars_v))

    # Save raw
    with open(out_dir / "hyperparam_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Table B1: grid table
    tableB1 = {}
    for key, data in all_results.items():
        s = data["summary"]
        tableB1[key] = {
            "alpha": data["alpha"], "beta": data["beta"],
            "f1":  f'{s["f1"]["mean"]:.4f} ± {s["f1"]["std"]:.4f}',
            "ars": f'{s["ars"]["mean"]:.4f} ± {s["ars"]["std"]:.4f}',
        }
    with open(out_dir / "tableB1_hyperparam.json", "w") as f:
        json.dump(tableB1, f, indent=2)

    logger.info("Table B1 -> %s", out_dir / "tableB1_hyperparam.json")

    if args.s3_bucket:
        from adftd.config import s3_upload
        for json_file in out_dir.glob("*.json"):
            s3_upload(str(json_file), args.s3_bucket,
                      f"{args.s3_prefix}/{json_file.name}",
                      region=args.region)


if __name__ == "__main__":
    main()
