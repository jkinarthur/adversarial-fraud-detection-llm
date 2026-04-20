"""
Training script for AD-FTD.

Usage:
    python scripts/train.py \
        --features data/features/samples.pkl \
        --traj_len 3 \
        --epochs 50 \
        --device cuda

AWS usage:
    python scripts/train.py \
        --features data/features/samples.pkl \
        --device cuda --num_gpus 4 \
        --s3_bucket my-bucket --s3_prefix adftd/checkpoints \
        --resume_checkpoint checkpoints/adftd_r0_f0.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train AD-FTD model")
    p.add_argument("--features", default="data/features/samples.pkl")
    p.add_argument("--traj_len", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--n_resample", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--seed", type=int, default=42)
    # ── AWS arguments ──────────────────────────────────────────────────────
    p.add_argument("--num_gpus", type=int, default=1,
                   help="Number of GPUs for DataParallel (1 = single GPU)")
    p.add_argument("--s3_bucket", default=None,
                   help="S3 bucket; if set, upload checkpoints after each fold")
    p.add_argument("--s3_prefix", default="adftd/checkpoints",
                   help="S3 key prefix for checkpoint uploads")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--resume_checkpoint", default=None,
                   help="Path (or s3://) to a checkpoint to resume from "
                        "(skips resamples/folds already completed)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve features file (supports s3://) ────────────────────────────
    from src.adftd.config import resolve_s3_path
    features_path = resolve_s3_path(args.features, region=args.region)
    with open(features_path, "rb") as f:
        samples = pickle.load(f)
    logger.info("Loaded %d samples", len(samples))

    # ── Build config ──────────────────────────────────────────────────────
    from src.adftd.config import ADFTDConfig, AWSConfig
    cfg = ADFTDConfig()
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.alpha = args.alpha
    cfg.train.beta = args.beta
    cfg.train.epsilon = args.epsilon
    cfg.train.n_folds = args.n_folds
    cfg.train.n_resample = args.n_resample
    cfg.train.device = args.device
    cfg.train.checkpoint_dir = args.checkpoint_dir
    cfg.train.seed = args.seed
    cfg.aws = AWSConfig(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        region=args.region,
        num_gpus=args.num_gpus,
    )

    # Auto-detect dims from first sample
    traj_dim = samples[0]["trajectory"].shape[-1]
    financial_dim = samples[0]["financial_ratios"].shape[0]
    cfg.model.trajectory_dim = traj_dim
    cfg.model.financial_dim = financial_dim
    cfg.data.traj_len = args.traj_len

    logger.info("trajectory_dim=%d  financial_dim=%d", traj_dim, financial_dim)

    # ── Build dataset ─────────────────────────────────────────────────────
    from src.adftd.data.dataset import FraudTrajectoryDataset
    dataset = FraudTrajectoryDataset(
        samples,
        traj_len=args.traj_len,
        augment=True,
        oversample_ratio=10,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    from src.adftd.training.trainer import ADFTDTrainer
    trainer = ADFTDTrainer(cfg)
    results = trainer.cross_validate(dataset)

    # ── Aggregate and report ──────────────────────────────────────────────
    from src.adftd.evaluation.metrics import evaluate_model
    from torch.utils.data import DataLoader
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_metrics = []
    for model, history in results:
        model = model.to(device)
        full_loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=FraudTrajectoryDataset.collate_fn,
        )
        m = evaluate_model(model, full_loader, device)
        all_metrics.append(m)
        logger.info(
            "PRC-AUC=%.4f  F1=%.4f  ARS=%.4f",
            m["clean"]["prc_auc"], m["clean"]["f1"], m["ars"]
        )

    # Summary
    prc_aucs = [m["clean"]["prc_auc"] for m in all_metrics]
    f1s = [m["clean"]["f1"] for m in all_metrics]
    arss = [m["ars"] for m in all_metrics]
    import numpy as np
    summary = {
        "prc_auc_mean": float(np.mean(prc_aucs)),
        "prc_auc_std": float(np.std(prc_aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "ars_mean": float(np.mean(arss)),
    }
    logger.info("=== Summary ===")
    for k, v in summary.items():
        logger.info("  %s: %.4f", k, v)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.checkpoint_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s/summary.json", args.checkpoint_dir)

    # ── Upload all checkpoints + summary to S3 ────────────────────────────
    if args.s3_bucket:
        from src.adftd.config import s3_upload
        for pt_file in Path(args.checkpoint_dir).glob("*.pt"):
            s3_upload(str(pt_file),
                      args.s3_bucket,
                      f"{args.s3_prefix}/{pt_file.name}",
                      region=args.region)
        s3_upload(str(summary_path),
                  args.s3_bucket,
                  f"{args.s3_prefix}/summary.json",
                  region=args.region)


if __name__ == "__main__":
    main()
