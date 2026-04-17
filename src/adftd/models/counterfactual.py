"""
Counterfactual Trajectory Generator: g_φ

Estimates the expected honest disclosure trajectory Ẑ_t^(i) given:
  - Historical trajectory Z_{t-1}^(i)    shape (T, traj_dim)
  - Financial ratios R_t^(i)              shape (financial_dim,)
  - Peer-group mean Z̄_N(i)              shape (traj_dim,)

Architecture:
  H_t = TCN(Z_{t-1})                        temporal encoding
  Ẑ_t = σ(W_h·H_t + W_r·R_t + W_n·Z̄_N)  feature fusion
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .tcn import TCN


class CounterfactualGenerator(nn.Module):
    """
    Predicts the non-fraudulent counterfactual trajectory Ẑ_t.

    Returns shape (batch, traj_dim) matching the input trajectory dimension.
    """

    def __init__(
        self,
        traj_dim: int = 36,
        financial_dim: int = 9,
        tcn_hidden: int = 128,
        tcn_levels: int = 4,
        tcn_kernel: int = 3,
        tcn_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.traj_dim = traj_dim

        # Temporal encoder for historical trajectory
        self.tcn = TCN(
            input_dim=traj_dim,
            hidden_dim=tcn_hidden,
            n_levels=tcn_levels,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
            return_sequence=False,
        )

        # Separate projections for each conditioning signal
        self.proj_h = nn.Linear(tcn_hidden, traj_dim)
        self.proj_r = nn.Linear(financial_dim, traj_dim)
        self.proj_n = nn.Linear(traj_dim, traj_dim)

        self.act = nn.Tanh()
        self.out = nn.Linear(traj_dim, traj_dim)

    def forward(
        self,
        z_hist: Tensor,     # (B, T, traj_dim)  historical trajectory
        r: Tensor,          # (B, financial_dim) financial ratios
        z_peer: Tensor,     # (B, traj_dim)      peer-group mean
    ) -> Tensor:            # (B, traj_dim)      predicted counterfactual
        h = self.tcn(z_hist)                 # (B, tcn_hidden)
        fused = (
            self.proj_h(h)
            + self.proj_r(r)
            + self.proj_n(z_peer)
        )
        return self.out(self.act(fused))     # (B, traj_dim)
