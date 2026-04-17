"""
PyTorch Dataset for AD-FTD.

Each sample is a (T, trajectory_dim) observed trajectory + peer mean + financial ratios + label.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FraudTrajectoryDataset(Dataset):
    """
    Dataset of fraud detection samples.

    Each sample:
        z        : (T, traj_dim)  — observed change trajectory
        z_peer   : (traj_dim,)   — peer-group mean trajectory for period t
        r        : (financial_dim,) — financial ratios at period t
        y        : scalar int    — fraud label (0/1)
    """

    def __init__(
        self,
        samples: List[Dict],
        traj_len: int = 3,
        augment: bool = False,
        oversample_ratio: int = 1,
    ) -> None:
        self.traj_len = traj_len
        self.augment = augment
        self._data = self._prepare(samples, oversample_ratio)

    def _prepare(self, samples: List[Dict], oversample_ratio: int) -> List[Dict]:
        fraud = [s for s in samples if s["label"] == 1]
        legit = [s for s in samples if s["label"] == 0]
        # Oversample fraud class
        if oversample_ratio > 1 and fraud:
            fraud = fraud * oversample_ratio
        return legit + fraud

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self._data[idx]
        z = torch.tensor(s["trajectory"], dtype=torch.float32)     # (T, D)
        z_peer = torch.tensor(s["peer_mean"], dtype=torch.float32)  # (D,)
        r = torch.tensor(s["financial_ratios"], dtype=torch.float32)  # (F,)
        y = torch.tensor(s["label"], dtype=torch.float32)

        if self.augment and np.random.rand() < 0.1:
            z = z + torch.randn_like(z) * 0.01

        return {"z": z, "z_peer": z_peer, "r": r, "y": y,
                "cik": s.get("cik", ""), "year": s.get("year", 0)}

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        keys = ["z", "z_peer", "r", "y"]
        out = {k: torch.stack([b[k] for b in batch]) for k in keys}
        out["cik"] = [b["cik"] for b in batch]
        out["year"] = torch.tensor([b["year"] for b in batch])
        return out


def build_samples(
    trajectories: Dict,   # (cik, year) -> np.ndarray (T, traj_dim)
    peer_means: Dict,     # (cik, year) -> np.ndarray (traj_dim,)
    financials: Dict,     # (cik, year) -> np.ndarray (financial_dim,)
    labels: Dict,         # (cik, year) -> int
    traj_len: int = 3,
) -> List[Dict]:
    """Combine all feature sources into sample dicts."""
    samples = []
    for (cik, year), traj in trajectories.items():
        if traj is None or np.isnan(traj).any():
            continue
        pm = peer_means.get((cik, year))
        fr = financials.get((cik, year))
        if pm is None or fr is None:
            continue
        samples.append({
            "cik": cik,
            "year": year,
            "trajectory": traj.astype(np.float32),
            "peer_mean": pm.astype(np.float32),
            "financial_ratios": fr.astype(np.float32),
            "label": int(labels.get((cik, year), 0)),
        })
    logger.info("Built %d samples (%d fraud)",
                len(samples), sum(s["label"] for s in samples))
    return samples
