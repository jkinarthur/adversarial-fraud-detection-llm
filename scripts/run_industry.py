"""
Industry sector generalization  →  Table V of the paper.

Leave-one-industry-out (LOIO) evaluation: for each 1-digit SIC sector,
train AD-FTD on all other sectors, test on the held-out sector.

Each sample must have a "sic1" field (1-digit SIC code derived from the
4-digit SIC stored in Compustat).  The preprocess.py script writes this
field; if absent we fall back to reading the 2-digit SIC from the "sic"
field and taking the first digit.

Outputs:
  results/industry_results.json   — per-industry per-fold numbers
  results/table5_industry.json    — F1 / PRC-AUC mean ± std per sector

Usage:
    cd AD-FTD
    python scripts/run_industry.py \
        --features data/features/samples.pkl \
        --traj_len 3 \
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# SIC 1-digit sector names (NAICS mapping for readability)
SIC1_NAMES = {
    "0": "Agriculture/Mining",
    "1": "Construction/Mining",
    "2": "Manufacturing (Non-durable)",
    "3": "Manufacturing (Durable)",
    "4": "Transport/Utilities",
    "5": "Wholesale/Retail",
    "6": "Finance/Insurance/RE",
    "7": "Services",
    "8": "Health/Education",
    "9": "Public Administration",
}


def parse_args():
    p = argparse.ArgumentParser(description="Industry LOIO evaluation")
    p.add_argument("--features", default="data/features/samples.pkl")
    p.add_argument("--traj_len", type=int, default=3)
    p.add_argument("--n_resample", type=int, default=5,
                   help="Resampling iterations per held-out sector")
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


def get_sic1(sample: dict) -> str:
    """Extract 1-digit SIC code from sample."""
    if "sic1" in sample:
        return str(sample["sic1"])
    elif "sic" in sample:
        return str(sample["sic"])[:1]
    else:
        return "9"   # fallback


def train_and_eval(train_samples, test_samples, cfg, traj_len, epsilon, device, seed):
    """Train one AD-FTD model on train_samples; evaluate on test_samples."""
    from adftd.data.dataset import FraudTrajectoryDataset
    from adftd.models.adftd import ADFTD
    from adftd.training.losses import ADFTDLoss
    from adftd.training.fgsm import fgsm_perturb
    from adftd.training.trainer import EarlyStopping
    from adftd.evaluation.metrics import compute_metrics, compute_ars

    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    train_ds = FraudTrajectoryDataset(
        train_samples, traj_len=traj_len, augment=True,
        oversample_ratio=cfg.data.oversample_ratio,
    )
    test_ds = FraudTrajectoryDataset(
        test_samples, traj_len=traj_len, augment=False, oversample_ratio=1
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        collate_fn=FraudTrajectoryDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False,
        collate_fn=FraudTrajectoryDataset.collate_fn,
    )

    model = ADFTD.from_config(cfg.model).to(dev)
    labels = [b["y"].mean().item() for b in train_loader]
    fraud_rate = float(np.mean(labels)) if labels else 0.01
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
            total, *_ = criterion(logits, z_hat, delta_z, y, logits_adv=logits_adv)
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl_list = []
        with torch.no_grad():
            for batch in train_loader:  # use train loss as proxy (no val split here)
                z  = batch["z"].to(dev); zp = batch["z_peer"].to(dev)
                r_ = batch["r"].to(dev); y  = batch["y"].to(dev)
                logits, z_hat, dz = model(z, zp, r_)
                vl, *_ = criterion(logits, z_hat, dz, y, logits_adv=logits)
                vl_list.append(vl.item())
        vl_mean = float(np.mean(vl_list))
        if vl_mean < best_val:
            best_val = vl_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(vl_mean):
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    y_true, p_c = [], []
    with torch.no_grad():
        for batch in test_loader:
            z  = batch["z"].to(dev); zp = batch["z_peer"].to(dev)
            r_ = batch["r"].to(dev)
            pc = torch.sigmoid(model(z, zp, r_)[0]).cpu().numpy()
            y_true.extend(batch["y"].numpy().tolist())
            p_c.extend(pc.tolist())

    return compute_metrics(np.array(y_true), np.array(p_c))


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
    cfg.data.traj_len = args.traj_len
    cfg.train.epochs  = args.epochs
    cfg.train.device  = args.device
    cfg.model.trajectory_dim = samples[0]["trajectory"].shape[-1]
    cfg.model.financial_dim  = samples[0]["financial_ratios"].shape[0]

    # Group by SIC1
    sectors: dict = {}
    for s in samples:
        key = get_sic1(s)
        sectors.setdefault(key, []).append(s)

    logger.info("Sectors found: %s",
                {k: len(v) for k, v in sorted(sectors.items())})

    all_results: dict = {}

    for held_out in sorted(sectors.keys()):
        test_samples  = sectors[held_out]
        train_samples = [s for k, ss in sectors.items() if k != held_out
                         for s in ss]
        if len(test_samples) < 10 or sum(s["label"] for s in test_samples) == 0:
            logger.warning("Skipping sector %s: too few / no fraud samples",
                           held_out)
            continue

        logger.info("LOIO sector=%s  train=%d  test=%d",
                    held_out, len(train_samples), len(test_samples))
        runs = []
        for rs in range(args.n_resample):
            metrics = train_and_eval(
                train_samples, test_samples, cfg,
                traj_len=args.traj_len, epsilon=args.epsilon,
                device=args.device, seed=args.seed + rs,
            )
            runs.append(metrics)
            logger.info("  sector=%s rs=%d F1=%.4f", held_out, rs, metrics["f1"])

        f1s   = [r["f1"]     for r in runs]
        prauc = [r["prc_auc"] for r in runs]
        prec  = [r["precision"] for r in runs]
        rec   = [r["recall"]   for r in runs]
        sname = SIC1_NAMES.get(held_out, f"Sector {held_out}")
        all_results[held_out] = {
            "name": sname,
            "n_test": len(test_samples),
            "n_fraud_test": sum(s["label"] for s in test_samples),
            "runs": runs,
            "summary": {
                "f1":        {"mean": float(np.mean(f1s)),   "std": float(np.std(f1s))},
                "prc_auc":   {"mean": float(np.mean(prauc)), "std": float(np.std(prauc))},
                "precision": {"mean": float(np.mean(prec)),  "std": float(np.std(prec))},
                "recall":    {"mean": float(np.mean(rec)),   "std": float(np.std(rec))},
            },
        }

    # Save raw
    with open(out_dir / "industry_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Table V
    table5 = {}
    for sid, data in sorted(all_results.items()):
        s = data["summary"]
        table5[f'{sid} – {data["name"]}'] = {
            "n_test":   data["n_test"],
            "n_fraud":  data["n_fraud_test"],
            "precision": f'{s["precision"]["mean"]:.4f} ± {s["precision"]["std"]:.4f}',
            "recall":    f'{s["recall"]["mean"]:.4f} ± {s["recall"]["std"]:.4f}',
            "f1":        f'{s["f1"]["mean"]:.4f} ± {s["f1"]["std"]:.4f}',
            "prc_auc":   f'{s["prc_auc"]["mean"]:.4f} ± {s["prc_auc"]["std"]:.4f}',
        }
    with open(out_dir / "table5_industry.json", "w") as f:
        json.dump(table5, f, indent=2)

    logger.info("Table V -> %s", out_dir / "table5_industry.json")

    if args.s3_bucket:
        from adftd.config import s3_upload
        for json_file in out_dir.glob("*.json"):
            s3_upload(str(json_file), args.s3_bucket,
                      f"{args.s3_prefix}/{json_file.name}",
                      region=args.region)


if __name__ == "__main__":
    main()
