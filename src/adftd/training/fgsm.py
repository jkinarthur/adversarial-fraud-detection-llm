"""
FGSM (Fast Gradient Sign Method) adversarial perturbation — embedding space.

Operates on the observed trajectory tensor Z, NOT on raw text.
This is an embedding-space proxy for adversarial robustness, as explicitly
discussed in §IV-E of the paper.  Language-space adversarial training using
LLM-generated rewrites is planned as future work.
"""
from __future__ import annotations

import torch
from torch import Tensor


def fgsm_perturb(
    z: Tensor,          # (B, T, traj_dim)  requires_grad must be set externally
    loss: Tensor,       # scalar loss to differentiate
    epsilon: float = 0.01,
) -> Tensor:
    """
    Compute FGSM adversarial perturbation on the trajectory tensor.

    Usage pattern:
        z.requires_grad_(True)
        logits, z_hat, delta_z = model(z, z_peer, r)
        l_cls = criterion(logits, y)
        l_cls.backward(retain_graph=True)
        z_adv = fgsm_perturb(z, l_cls, epsilon)
        z.requires_grad_(False)

    Returns a new tensor (detached, no grad) with perturbation applied.
    """
    if z.grad is None:
        raise RuntimeError(
            "z.grad is None — call loss.backward() before fgsm_perturb()."
        )
    with torch.no_grad():
        z_adv = z + epsilon * z.grad.sign()
    return z_adv.detach()
