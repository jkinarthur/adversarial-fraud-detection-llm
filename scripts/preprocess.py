"""
Preprocessing script.

Downloads and caches EDGAR-CORPUS text, then builds change trajectory
features for all companies, saving results to data/features/.

Usage:
    python scripts/preprocess.py \
        --backend openai \
        --aaer_csv data/raw/aaer_labels.csv \
        --compustat_csv data/raw/compustat_ratios.csv \
        --out_dir data/features \
        --start_year 1995 \
        --end_year 2019

Set OPENAI_API_KEY (or HF_TOKEN for llama3) before running.
Approximate cost with GPT-3.5: ~USD 3,500 for full corpus.

AWS usage:
    python scripts/preprocess.py --backend llama3 \
        --s3_bucket my-bucket --s3_prefix adftd/features \
        --resume
    Use --resume to skip already-processed CIKs (safe on spot instances).
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="openai", choices=["openai", "llama3"])
    p.add_argument("--aaer_csv", default="data/raw/aaer_labels.csv")
    p.add_argument("--compustat_csv", default="data/raw/compustat_ratios.csv")
    p.add_argument("--out_dir", default="data/features")
    p.add_argument("--edgar_cache", default="data/raw/edgar")
    p.add_argument("--start_year", type=int, default=1995)
    p.add_argument("--end_year", type=int, default=2019)
    p.add_argument("--traj_len", type=int, default=3)
    p.add_argument("--lm_dict", default=None,
                   help="Path to Loughran-McDonald master dictionary CSV (optional)")
    # ── AWS arguments ──────────────────────────────────────────────────────
    p.add_argument("--s3_bucket", default=None,
                   help="S3 bucket name; if set, upload outputs on completion")
    p.add_argument("--s3_prefix", default="adftd/features",
                   help="S3 key prefix for uploaded artefacts")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--resume", action="store_true",
                   help="Skip CIKs already recorded in progress.json "
                        "(safe restart after spot-instance interruption)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: load set of already-processed CIKs ────────────────────────
    progress_file = out_dir / "progress.json"
    processed_ciks: set = set()
    if args.resume and progress_file.exists():
        with open(progress_file) as f:
            processed_ciks = set(json.load(f).get("processed_ciks", []))
        logger.info("Resume mode: %d CIKs already processed", len(processed_ciks))

    # ── Load word categories ──────────────────────────────────────────────
    from src.adftd.features.fraud_w2v import WORD_CATEGORIES, load_lm_wordlists
    word_cats = WORD_CATEGORIES
    if args.lm_dict:
        word_cats = load_lm_wordlists(args.lm_dict)
        logger.info("Using LM master dictionary from %s", args.lm_dict)

    # ── Build embedder ────────────────────────────────────────────────────
    from src.adftd.features.embedders import build_embedder
    embedder = build_embedder(args.backend)
    embed_fn = embedder.embed

    # ── Load EDGAR text ───────────────────────────────────────────────────
    from src.adftd.data.edgar_loader import iter_edgar_filings
    logger.info("Loading EDGAR-CORPUS %d–%d ...", args.start_year, args.end_year)
    texts_by_cik: dict = {}
    for filing in iter_edgar_filings(
        cache_dir=args.edgar_cache,
        start_year=args.start_year,
        end_year=args.end_year,
    ):
        cik = filing["cik"]
        year = filing["year"]
        texts_by_cik.setdefault(cik, {})[year] = filing["text"]

    logger.info("Loaded texts for %d unique CIKs", len(texts_by_cik))

    # ── Build trajectories ────────────────────────────────────────────────
    from src.adftd.features.trajectory import build_trajectory_sequence
    trajectories: dict = {}   # (cik, year) -> (T, D)
    total_ciks = len(texts_by_cik)
    for idx, (cik, year_texts) in enumerate(texts_by_cik.items()):
        if cik in processed_ciks:
            continue
        result = build_trajectory_sequence(
            texts_by_year=year_texts,
            embed_fn=embed_fn,
            word_categories=word_cats,
            traj_len=args.traj_len,
        )
        if result is None:
            processed_ciks.add(cik)
            continue
        for year, traj in result.items():
            trajectories[(cik, year)] = traj
        processed_ciks.add(cik)

        # Save progress every 100 CIKs for spot-instance safety
        if (idx + 1) % 100 == 0:
            with open(progress_file, "w") as f:
                json.dump({"processed_ciks": list(processed_ciks)}, f)
            logger.info("Progress: %d / %d CIKs", idx + 1, total_ciks)

    logger.info("Built %d (cik, year) trajectory tensors", len(trajectories))

    # ── Load AAER labels ──────────────────────────────────────────────────
    from src.adftd.data.aaer_matcher import load_aaer_labels
    labels = load_aaer_labels(args.aaer_csv)

    # ── Load Compustat financial ratios ───────────────────────────────────
    compustat_df = pd.read_csv(args.compustat_csv, dtype={"cik": str})
    ratio_cols = [c for c in compustat_df.columns
                  if c not in ("cik", "gvkey", "fyear", "sic", "at")]
    logger.info("Financial ratio columns: %s", ratio_cols)

    financials: dict = {}
    for _, row in compustat_df.iterrows():
        cik = str(row["cik"]).strip().lstrip("0")
        year = int(row["fyear"])
        vals = row[ratio_cols].values.astype(np.float32)
        financials[(cik, year)] = np.nan_to_num(vals)

    # ── Build peer means ──────────────────────────────────────────────────
    from src.adftd.data.peer_group import build_peer_groups
    train_ciks = list({k[0] for k in trajectories})
    peer_map = build_peer_groups(compustat_df, train_ciks=train_ciks)

    peer_means: dict = {}
    for (cik, year), traj in trajectories.items():
        from src.adftd.data.peer_group import get_peers
        peer_ciks = get_peers(peer_map, cik, year)
        peer_vecs = [
            trajectories[(pc, year)][-1]           # last step of peer trajectory
            for pc in peer_ciks
            if (pc, year) in trajectories
        ]
        if peer_vecs:
            peer_means[(cik, year)] = np.stack(peer_vecs).mean(axis=0)
        else:
            peer_means[(cik, year)] = np.zeros_like(traj[-1])

    # ── Build samples ─────────────────────────────────────────────────────
    from src.adftd.data.dataset import build_samples
    samples = build_samples(
        trajectories=trajectories,
        peer_means=peer_means,
        financials=financials,
        labels=labels,
        traj_len=args.traj_len,
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = out_dir / "samples.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(samples, f)
    logger.info("Saved %d samples to %s", len(samples), out_path)

    # ── Upload to S3 (if configured) ──────────────────────────────────────
    if args.s3_bucket:
        from src.adftd.config import s3_upload
        s3_upload(str(out_path),
                  args.s3_bucket,
                  f"{args.s3_prefix}/samples.pkl",
                  region=args.region)
        # Also upload AAER CSV for traceability
        s3_upload(args.aaer_csv,
                  args.s3_bucket,
                  f"{args.s3_prefix}/aaer_labels.csv",
                  region=args.region)


if __name__ == "__main__":
    main()
