"""
Evaluation metrics for AD-FTD.

Primary metrics (§V-D):
  - PRC-AUC  (Precision-Recall Curve AUC)  — primary
  - Macro F1                               — primary
  - Precision, Recall, Accuracy
  - ROC-AUC  (reported for comparability, not as primary metric)

Additional metrics:
  - ARS  (Adversarial Robustness Score): |F1_clean - F1_adv| — lower is better
  - Expected Cost at multiple cost-ratio thresholds (Elkan 2001)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (  # type: ignore
        precision_recall_curve,
        roc_auc_score,
        auc,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    logger.warning("scikit-learn not installed — metrics unavailable.")


# ──────────────────────────────────────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute the full metric suite given true labels and predicted probabilities.

    Parameters
    ----------
    y_true  : (N,) int array of 0/1 labels
    y_score : (N,) float array of fraud probabilities
    threshold : decision threshold for binary metrics

    Returns dict with keys:
        prc_auc, roc_auc, f1, precision, recall, accuracy
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    y_pred = (y_score >= threshold).astype(int)

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_score)
    prc_auc = auc(rec_curve, prec_curve)

    try:
        roc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc = float("nan")

    return {
        "prc_auc": float(prc_auc),
        "roc_auc": float(roc),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def compute_ars(f1_clean: float, f1_adv: float) -> float:
    """Adversarial Robustness Score: absolute F1 drop (lower is better)."""
    return abs(f1_clean - f1_adv)


# ──────────────────────────────────────────────────────────────────────────────
# Expected Cost (Elkan 2001)
# ──────────────────────────────────────────────────────────────────────────────

def expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fn_over_fp: float = 100.0,
) -> Tuple[float, float, float]:
    """
    Compute the minimum expected cost over all thresholds.

    EC(τ) = c_FN · FNR(τ) · p + c_FP · FPR(τ) · (1-p)

    where c_FN/c_FP = cost_fn_over_fp.

    Returns (min_ec, optimal_threshold, optimal_f1)
    """
    p = float(y_true.mean())
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn required")

    thresholds = np.linspace(0.01, 0.99, 200)
    best_ec = float("inf")
    best_tau = 0.5
    best_f1 = 0.0

    for tau in thresholds:
        y_pred = (y_score >= tau).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fnr = fn / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        ec = cost_fn_over_fp * fnr * p + 1.0 * fpr * (1.0 - p)
        if ec < best_ec:
            best_ec = ec
            best_tau = tau
            best_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return best_ec, best_tau, best_f1


def expected_cost_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_ratios: Optional[List[float]] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Compute Expected Cost for multiple cost ratios.
    Returns dict: cost_ratio -> {min_ec, threshold, f1_at_threshold}
    """
    if cost_ratios is None:
        cost_ratios = [100.0, 500.0, 1000.0]
    results = {}
    for ratio in cost_ratios:
        ec, tau, f1 = expected_cost(y_true, y_score, cost_fn_over_fp=ratio)
        results[ratio] = {"min_ec": ec, "threshold": tau, "f1_at_threshold": f1}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Model evaluation on a DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
    epsilon: float = 0.01,
    cost_ratios: Optional[List[float]] = None,
) -> Dict:
    """
    Full evaluation pass: clean metrics + adversarial metrics + expected cost.

    Returns a dict with all metrics for clean and adversarial conditions.
    """
    model.eval()

    all_y, all_score_clean, all_score_adv = [], [], []

    for batch in loader:
        z = batch["z"].to(device)
        z_peer = batch["z_peer"].to(device)
        r = batch["r"].to(device)
        y = batch["y"].cpu().numpy()

        # Clean
        with torch.no_grad():
            proba_clean = model.predict_proba(z, z_peer, r).cpu().numpy()

        # Adversarial (FGSM)
        z_adv = _fgsm_eval(model, z, z_peer, r, epsilon)
        with torch.no_grad():
            proba_adv = model.predict_proba(z_adv, z_peer, r).cpu().numpy()

        all_y.append(y)
        all_score_clean.append(proba_clean)
        all_score_adv.append(proba_adv)

    y_true = np.concatenate(all_y)
    score_clean = np.concatenate(all_score_clean)
    score_adv = np.concatenate(all_score_adv)

    metrics_clean = compute_metrics(y_true, score_clean)
    metrics_adv = compute_metrics(y_true, score_adv)
    ars = compute_ars(metrics_clean["f1"], metrics_adv["f1"])

    ec_table = expected_cost_table(y_true, score_clean, cost_ratios or [100.0, 500.0, 1000.0])

    return {
        "clean": metrics_clean,
        "adversarial": metrics_adv,
        "ars": ars,
        "expected_cost": ec_table,
        "n_samples": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
    }


def _fgsm_eval(model, z: Tensor, z_peer: Tensor, r: Tensor,
               epsilon: float) -> Tensor:
    """Compute FGSM adversarial perturbation for evaluation."""
    z_in = z.clone().requires_grad_(True)
    logits, _, _ = model(z_in, z_peer, r)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        torch.zeros_like(logits)   # worst-case: push toward non-fraud
    )
    loss.backward()
    with torch.no_grad():
        z_adv = z_in + epsilon * z_in.grad.sign()
    return z_adv.detach()
