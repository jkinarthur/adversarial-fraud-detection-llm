"""
Fraud Detector: f_θ

Input joint triple:
    Z̃_t = [Z_t ‖ Ẑ_t ‖ ΔZ_t]  shape (B, 3·traj_dim)

Expands to sequence (B, T=1, 3·traj_dim), passes through TCN → BiLSTM → sigmoid.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .tcn import TCN


class FraudDetector(nn.Module):
    """
    TCN + BiLSTM classifier operating on the concatenated triple [Z‖Ẑ‖ΔZ].
    """

    def __init__(
        self,
        traj_dim: int = 36,
        tcn_hidden: int = 128,
        tcn_levels: int = 4,
        tcn_kernel: int = 3,
        tcn_dropout: float = 0.2,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        bilstm_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = traj_dim * 3    # [Z ‖ Ẑ ‖ ΔZ]

        self.tcn = TCN(
            input_dim=input_dim,
            hidden_dim=tcn_hidden,
            n_levels=tcn_levels,
            kernel_size=tcn_kernel,
            dropout=tcn_dropout,
            return_sequence=True,
        )

        self.bilstm = nn.LSTM(
            input_size=tcn_hidden,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=bilstm_dropout if bilstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(bilstm_dropout)
        self.fc = nn.Linear(bilstm_hidden * 2, 1)   # *2 for bidirectional

    def forward(self, z_joint: Tensor) -> Tensor:
        """
        z_joint: (B, T, 3·traj_dim)  or  (B, 3·traj_dim) [single step]
        Returns: (B,) fraud probability logits (before sigmoid)
        """
        if z_joint.dim() == 2:
            z_joint = z_joint.unsqueeze(1)           # (B, 1, D)

        h_tcn = self.tcn(z_joint)                    # (B, T, tcn_hidden)
        h_lstm, _ = self.bilstm(h_tcn)               # (B, T, 2·bilstm_hidden)
        h_last = h_lstm[:, -1, :]                    # (B, 2·bilstm_hidden)
        return self.fc(self.dropout(h_last)).squeeze(-1)   # (B,)
