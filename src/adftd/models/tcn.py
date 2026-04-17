"""
Temporal Convolutional Network (TCN) with dilated causal convolutions.

Architecture follows Bai et al. (2018) "An Empirical Evaluation of Generic
Convolutional and Recurrent Networks for Sequence Modeling".
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class _CausalConv1d(nn.Module):
    """Causal 1-D convolution with left-only padding."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel, dilation=dilation, padding=self.padding
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)[:, :, : x.size(2)]


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.downsample(x) if self.downsample else x
        return torch.relu(self.net(x) + residual)


class TCN(nn.Module):
    """
    Temporal Convolutional Network.

    Input:  (batch, seq_len, input_dim)
    Output: (batch, seq_len, hidden_dim)   [return_sequence=True]
            (batch, hidden_dim)             [return_sequence=False, last step]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        return_sequence: bool = False,
    ) -> None:
        super().__init__()
        self.return_sequence = return_sequence
        channels = [input_dim] + [hidden_dim] * n_levels
        layers = []
        for i in range(n_levels):
            dilation = 2 ** i
            layers.append(
                _ResidualBlock(channels[i], channels[i + 1], kernel_size,
                               dilation, dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) -> (B, D, T) for Conv1d
        out = self.network(x.transpose(1, 2)).transpose(1, 2)  # (B, T, hidden)
        if self.return_sequence:
            return out
        return out[:, -1, :]   # last time step
