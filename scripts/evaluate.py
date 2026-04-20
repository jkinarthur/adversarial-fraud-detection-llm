"""
Standalone evaluation script.

Loads a saved checkpoint and evaluates on a held-out samples file.

Usage:
    python scripts/evaluate.py \
        --checkpoint checkpoints/adftd_r0_f0.pt \
        --features data/features/samples.pkl \
        --device cuda

AWS usage (load directly from S3):
    python scripts/evaluate.py \
        --checkpoint s3://my-bucket/adftd/checkpoints/adftd_r0_f0.pt \
        --features s3://my-bucket/adftd/features/samples.pkl \
        --region us-east-1
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AD-FTD checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Local path or s3://bucket/key to model checkpoint")
    p.add_argument("--features", default="data/features/samples.pkl",
                   help="Local path or s3://bucket/key to samples pickle")
    p.add_argument("--traj_len", type=int, default=3)
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="eval_results.json")
    p.add_argument("--region", default="us-east-1",
                   help="AWS region (only needed when using s3:// paths)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve S3 paths to local files ───────────────────────────────────
    from src.adftd.config import resolve_s3_path
    features_path = resolve_s3_path(args.features, region=args.region)
    checkpoint_path = resolve_s3_path(args.checkpoint, region=args.region)

    with open(features_path, "rb") as f:
        samples = pickle.load(f)

    traj_dim = samples[0]["trajectory"].shape[-1]
    financial_dim = samples[0]["financial_ratios"].shape[0]

    from src.adftd.config import ModelConfig
    cfg = ModelConfig(trajectory_dim=traj_dim, financial_dim=financial_dim)

    from src.adftd.models.adftd import ADFTD
    device = torch.device(args.device)
    model = ADFTD.from_config(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded checkpoint %s", args.checkpoint)

    from src.adftd.data.dataset import FraudTrajectoryDataset
    from torch.utils.data import DataLoader
    dataset = FraudTrajectoryDataset(samples, traj_len=args.traj_len,
                                     augment=False, oversample_ratio=1)
    loader = DataLoader(
        dataset, batch_size=64, shuffle=False,
        collate_fn=FraudTrajectoryDataset.collate_fn,
    )

    from src.adftd.evaluation.metrics import evaluate_model
    results = evaluate_model(model, loader, device,
                             epsilon=args.epsilon,
                             cost_ratios=[100.0, 500.0, 1000.0])

    logger.info("=== Clean ===")
    for k, v in results["clean"].items():
        logger.info("  %s: %.4f", k, v)
    logger.info("ARS: %.4f", results["ars"])
    logger.info("=== Expected Cost ===")
    for ratio, ec in results["expected_cost"].items():
        logger.info("  c_FN/c_FP=%.0f: min_EC=%.4f  threshold=%.2f  F1@τ=%.4f",
                    ratio, ec["min_ec"], ec["threshold"], ec["f1_at_threshold"])

    with open(args.out, "w") as f:
        # Convert int/float64 keys to strings for JSON
        out = dict(results)
        out["expected_cost"] = {
            str(k): v for k, v in results["expected_cost"].items()
        }
        json.dump(out, f, indent=2)
    logger.info("Results saved to %s", args.out)


if __name__ == "__main__":
    main()
