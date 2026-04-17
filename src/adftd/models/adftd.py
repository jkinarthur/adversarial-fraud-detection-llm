"""
Full AD-FTD model.

Combines CounterfactualGenerator (g_φ) and FraudDetector (f_θ) into a
single nn.Module with a unified forward pass.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .counterfactual import CounterfactualGenerator
from .detector import FraudDetector
from ..config import ModelConfig


class ADFTD(nn.Module):
    """
    Adversarial Disclosure-Aware Fraud Detection model.

    forward() returns:
        logits  : (B,)         raw logit for fraud class
        z_hat   : (B, traj_dim) estimated counterfactual trajectory
        delta_z : (B, traj_dim) deviation encoding Z - Ẑ
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.generator = CounterfactualGenerator(
            traj_dim=cfg.trajectory_dim,
            financial_dim=cfg.financial_dim,
            tcn_hidden=cfg.tcn_hidden,
            tcn_levels=cfg.tcn_levels,
            tcn_kernel=cfg.tcn_kernel_size,
            tcn_dropout=cfg.tcn_dropout,
        )
        self.detector = FraudDetector(
            traj_dim=cfg.trajectory_dim,
            tcn_hidden=cfg.tcn_hidden,
            tcn_levels=cfg.tcn_levels,
            tcn_kernel=cfg.tcn_kernel_size,
            tcn_dropout=cfg.tcn_dropout,
            bilstm_hidden=cfg.bilstm_hidden,
            bilstm_layers=cfg.bilstm_layers,
            bilstm_dropout=cfg.bilstm_dropout,
        )
        self.traj_dim = cfg.trajectory_dim

    def forward(
        self,
        z: Tensor,          # (B, T, traj_dim)  observed trajectory sequence
        z_peer: Tensor,     # (B, traj_dim)     peer-group mean
        r: Tensor,          # (B, financial_dim) financial ratios
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Step 1: generate counterfactual using all history except last step
        z_hat = self.generator(z, r, z_peer)         # (B, traj_dim)

        # Step 2: deviation
        z_last = z[:, -1, :]                         # (B, traj_dim)
        delta_z = z_last - z_hat                     # (B, traj_dim)

        # Step 3: joint triple (single time step for detector)
        z_joint = torch.cat([z_last, z_hat, delta_z], dim=-1)   # (B, 3·D)

        # Step 4: fraud classification
        logits = self.detector(z_joint)              # (B,)

        return logits, z_hat, delta_z

    def predict_proba(
        self,
        z: Tensor,
        z_peer: Tensor,
        r: Tensor,
    ) -> Tensor:
        """Returns (B,) fraud probability after sigmoid."""
        logits, _, _ = self.forward(z, z_peer, r)
        return torch.sigmoid(logits)

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "ADFTD":
        return cls(cfg)
