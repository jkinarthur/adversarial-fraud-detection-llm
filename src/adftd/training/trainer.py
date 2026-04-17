"""
AD-FTD Trainer.

Implements 4-fold cross-validation with 10 resampling iterations,
early stopping, and FGSM adversarial training as described in §V-B of the paper.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold  # type: ignore

from ..config import ADFTDConfig
from ..models.adftd import ADFTD
from ..data.dataset import FraudTrajectoryDataset
from .losses import ADFTDLoss
from .fgsm import fgsm_perturb

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class ADFTDTrainer:
    def __init__(self, cfg: ADFTDConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(
            cfg.train.device if torch.cuda.is_available() else "cpu"
        )
        logger.info("Using device: %s", self.device)

    # ──────────────────────────────────────────────────────────────────────
    # Single fold training
    # ──────────────────────────────────────────────────────────────────────
    def train_fold(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_id: int = 0,
    ) -> Tuple[ADFTD, Dict]:
        cfg = self.cfg
        model = ADFTD.from_config(cfg.model).to(self.device)

        # Class-imbalance weight from training data
        labels = [b["y"].mean().item() for b in train_loader]
        fraud_rate = np.mean(labels) if labels else 0.01
        pos_weight = (1.0 - fraud_rate) / max(fraud_rate, 1e-6)

        criterion = ADFTDLoss(
            alpha=cfg.train.alpha,
            beta=cfg.train.beta,
            pos_weight=pos_weight,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
        )
        stopper = EarlyStopping(patience=cfg.train.early_stop_patience)

        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": []
        }

        best_state: Optional[dict] = None
        best_val = float("inf")

        for epoch in range(cfg.train.epochs):
            model.train()
            train_losses = []

            for batch in train_loader:
                z = batch["z"].to(self.device)           # (B, T, D)
                z_peer = batch["z_peer"].to(self.device)  # (B, D)
                r = batch["r"].to(self.device)            # (B, F)
                y = batch["y"].to(self.device)            # (B,)

                # ── Forward pass (clean) ──────────────────────────────────
                z.requires_grad_(True)
                optimizer.zero_grad()

                logits, z_hat, delta_z = model(z, z_peer, r)
                l_cls = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor([pos_weight], device=self.device)
                )(logits, y)
                l_cls.backward(retain_graph=True)

                # ── FGSM adversarial perturbation ─────────────────────────
                z_adv = fgsm_perturb(z, l_cls, epsilon=cfg.train.epsilon)
                z.requires_grad_(False)

                # Forward on adversarial input
                logits_adv, _, _ = model(z_adv, z_peer, r)

                # ── Total loss ────────────────────────────────────────────
                total_loss, _ = criterion(logits, logits_adv,
                                          z[:, -1, :], z_hat, y)
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(total_loss.item())

            avg_train = np.mean(train_losses)
            avg_val = self._eval_loss(model, criterion, val_loader)
            scheduler.step(avg_val)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)

            if avg_val < best_val:
                best_val = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0:
                logger.info(
                    "[Fold %d | Epoch %3d] train=%.4f  val=%.4f",
                    fold_id, epoch, avg_train, avg_val
                )

            if stopper.step(avg_val):
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state:
            model.load_state_dict(best_state)

        return model, history

    def _eval_loss(
        self,
        model: ADFTD,
        criterion: ADFTDLoss,
        loader: DataLoader,
    ) -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                z = batch["z"].to(self.device)
                z_peer = batch["z_peer"].to(self.device)
                r = batch["r"].to(self.device)
                y = batch["y"].to(self.device)
                logits, z_hat, _ = model(z, z_peer, r)
                # For val loss use only cls + dev (no FGSM)
                l, _ = criterion(logits, logits, z[:, -1, :], z_hat, y)
                losses.append(l.item())
        return float(np.mean(losses))

    # ──────────────────────────────────────────────────────────────────────
    # Cross-validation
    # ──────────────────────────────────────────────────────────────────────
    def cross_validate(
        self,
        dataset: FraudTrajectoryDataset,
    ) -> List[Tuple[ADFTD, Dict]]:
        cfg = self.cfg
        labels = [dataset[i]["y"].item() for i in range(len(dataset))]
        indices = list(range(len(dataset)))

        all_results = []
        for resample in range(cfg.train.n_resample):
            set_seed(cfg.train.seed + resample)
            skf = StratifiedKFold(
                n_splits=cfg.train.n_folds,
                shuffle=True,
                random_state=cfg.train.seed + resample,
            )
            for fold_id, (train_idx, val_idx) in enumerate(
                skf.split(indices, labels)
            ):
                train_sub = Subset(dataset, train_idx)
                val_sub = Subset(dataset, val_idx)
                train_loader = DataLoader(
                    train_sub,
                    batch_size=cfg.train.batch_size,
                    shuffle=True,
                    collate_fn=FraudTrajectoryDataset.collate_fn,
                    drop_last=True,
                )
                val_loader = DataLoader(
                    val_sub,
                    batch_size=cfg.train.batch_size,
                    shuffle=False,
                    collate_fn=FraudTrajectoryDataset.collate_fn,
                )
                global_fold = resample * cfg.train.n_folds + fold_id
                logger.info("=== Resample %d / Fold %d ===", resample, fold_id)
                model, history = self.train_fold(train_loader, val_loader,
                                                 fold_id=global_fold)
                all_results.append((model, history))

                # Save checkpoint
                ckpt_dir = Path(cfg.train.checkpoint_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    ckpt_dir / f"adftd_r{resample}_f{fold_id}.pt",
                )

        return all_results
