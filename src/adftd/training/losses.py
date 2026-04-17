"""
Combined loss function for AD-FTD.

L = L_cls + α·L_dev + β·L_adv

L_cls : binary cross-entropy (fraud classification)
L_dev : MSE between observed trajectory and counterfactual (deviation regularisation)
L_adv : cross-entropy on FGSM-perturbed trajectory (adversarial robustness)
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ADFTDLoss(nn.Module):
    """
    Joint loss for AD-FTD.

    Parameters
    ----------
    alpha : weight for deviation regularisation term
    beta  : weight for adversarial loss term
    pos_weight : positive class weight for BCEWithLogitsLoss (handles imbalance)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        pos_weight: float = 97.0,   # ~1:97 imbalance ratio at T=3
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(
        self,
        logits: Tensor,       # (B,)
        logits_adv: Tensor,   # (B,)  logits on adversarially perturbed input
        z_last: Tensor,       # (B, traj_dim)
        z_hat: Tensor,        # (B, traj_dim)
        y: Tensor,            # (B,)
    ) -> Tuple[Tensor, dict]:
        """
        Returns (total_loss, component_dict).
        """
        device = logits.device
        self.bce.pos_weight = self.bce.pos_weight.to(device)

        l_cls = self.bce(logits, y)
        l_dev = F.mse_loss(z_hat, z_last.detach())
        l_adv = self.bce(logits_adv, y)

        total = l_cls + self.alpha * l_dev + self.beta * l_adv
        components = {
            "loss_cls": l_cls.item(),
            "loss_dev": l_dev.item(),
            "loss_adv": l_adv.item(),
            "loss_total": total.item(),
        }
        return total, components
