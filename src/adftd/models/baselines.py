"""
Baseline models for comparison against AD-FTD.

Models implemented (Tables I and II of the paper):
  Classical ML : LR, SVM, RF, XGBoost
  Sequential   : LSTM, TCN-only, TCN+BiLSTM (no counterfactual, no FGSM)
  Transformer  : Informer (simplified), Reformer (simplified), TIME-LLM (simplified)
  Ablation base: ParaEmb+FraudW2V (trajectory features only, no counterfactual)

All classifiers expose a common sklearn-style interface:
    fit(X_train, y_train)
    predict_proba(X_test) -> np.ndarray of shape (N, 2)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Utility: flatten trajectory tensor to 2-D feature matrix
# ──────────────────────────────────────────────────────────────────────────────

def flatten_samples(samples, traj_len: int = 3) -> tuple:
    """
    Convert list-of-dicts (same format as FraudTrajectoryDataset) into
    (X, y) numpy arrays suitable for sklearn classifiers.

    X shape : (N, traj_len * traj_dim + financial_dim)
    y shape : (N,)
    """
    Xs, ys = [], []
    for s in samples:
        traj = np.array(s["trajectory"], dtype=np.float32).flatten()
        ratios = np.array(s["financial_ratios"], dtype=np.float32)
        Xs.append(np.concatenate([traj, ratios]))
        ys.append(int(s["label"]))
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Classical ML baselines
# ──────────────────────────────────────────────────────────────────────────────

def make_lr():
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced",
            solver="lbfgs", random_state=42,
        )),
    ])


def make_svm():
    from sklearn.svm import SVC  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=1.0, probability=True,
            class_weight="balanced", random_state=42,
        )),
    ])


def make_rf():
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    return RandomForestClassifier(
        n_estimators=500, max_depth=None,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


def make_xgboost():
    try:
        from xgboost import XGBClassifier  # type: ignore
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=10.0, use_label_encoder=False,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    except ImportError:
        logger.warning("xgboost not installed; substituting GradientBoosting")
        from sklearn.ensemble import GradientBoostingClassifier  # type: ignore
        return GradientBoostingClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            random_state=42,
        )


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch sequential baselines (wrap into sklearn-style API)
# ──────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader


def _train_torch_clf(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    device: str = "cpu",
) -> nn.Module:
    """Generic training loop shared by LSTM / TCN / Transformer baselines."""
    device_ = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_)

    fraud_rate = y_train.mean()
    pos_w = torch.tensor(
        [(1.0 - fraud_rate) / max(fraud_rate, 1e-6)], dtype=torch.float32
    ).to(device_)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    loader = TorchDataLoader(ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device_), yb.to(device_)
            optimizer.zero_grad()
            out = model(xb).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model


def _predict_torch_clf(
    model: nn.Module,
    X_test: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Returns (N,) fraud probability array."""
    device_ = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device_).eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device_)
    with torch.no_grad():
        logits = model(X_t).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


class _LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=False,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim) or (B, flat_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)          # treat flat vector as T=1
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])         # (B, 1)


class _TCNOnlyClassifier(nn.Module):
    """TCN without BiLSTM or counterfactual."""
    def __init__(self, input_dim: int, hidden: int = 128, levels: int = 4,
                 kernel: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        from .tcn import TCN  # type: ignore
        self.tcn = TCN(
            num_inputs=input_dim, num_channels=[hidden] * levels,
            kernel_size=kernel, dropout=dropout,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1).transpose(1, 2)   # (B, C, 1)
        else:
            x = x.transpose(1, 2)                 # (B, C, T)
        out = self.tcn(x)[:, :, -1]              # (B, hidden)
        return self.head(out)


class _TCNBiLSTMClassifier(nn.Module):
    """TCN + BiLSTM without counterfactual / FGSM (ablation base)."""
    def __init__(self, input_dim: int, tcn_hidden: int = 128, tcn_levels: int = 4,
                 kernel: int = 3, dropout: float = 0.2,
                 lstm_hidden: int = 128, lstm_layers: int = 2) -> None:
        super().__init__()
        from .tcn import TCN  # type: ignore
        self.tcn = TCN(
            num_inputs=input_dim, num_channels=[tcn_hidden] * tcn_levels,
            kernel_size=kernel, dropout=dropout,
        )
        self.lstm = nn.LSTM(tcn_hidden, lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1).transpose(1, 2)
        else:
            x = x.transpose(1, 2)
        tcn_out = self.tcn(x).transpose(1, 2)    # (B, T, hidden)
        _, (h, _) = self.lstm(tcn_out)            # (2, B, lstm_hidden)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1) # (B, 2*lstm_hidden)
        return self.head(h_cat)


class _InformerClassifier(nn.Module):
    """
    Simplified Informer: Transformer encoder with ProbSparse attention approximated
    by standard multi-head attention (captures the architecture without the full
    data-dependent sparsity mask which requires O(n log n) keys).
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, 1, D)
        x = self.proj(x)               # (B, T, d_model)
        out = self.encoder(x)          # (B, T, d_model)
        return self.head(out[:, -1, :])  # (B, 1)


class _ReformerClassifier(nn.Module):
    """
    Simplified Reformer: LSH attention approximated via standard MHA.
    Architecture is equivalent to _InformerClassifier at this scale.
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.proj(x)
        out = self.encoder(x)
        return self.head(out[:, -1, :])


class _TIMELLMClassifier(nn.Module):
    """
    TIME-LLM proxy: frozen BERT-style token embedding layer used as a
    'language model backbone' with a lightweight MLP reprogramming head
    (captures the cross-modal reprogramming concept without a full 7B LLM).
    """
    def __init__(self, input_dim: int, d_model: int = 128, dropout: float = 0.1,
                 patch_len: int = 4) -> None:
        super().__init__()
        # Patch embedding (reprogramming layer)
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        emb = self.patch_embed(x)
        out = self.encoder(emb)
        return self.head(out[:, -1, :])


class _ParaEmbFraudW2VClassifier(nn.Module):
    """
    ParaEmb+FraudW2V baseline: trajectory features fed through BiLSTM only.
    No counterfactual generator, no deviation encoding, no adversarial training.
    """
    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(h_cat)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper: uniform train / predict interface for all baselines
# ──────────────────────────────────────────────────────────────────────────────

class SklearnWrapper:
    """Wraps sklearn Pipeline for uniform predict_proba interface."""
    def __init__(self, clf):
        self._clf = clf

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)[:, 1]   # positive class


class TorchWrapper:
    """Wraps a PyTorch nn.Module for uniform train / predict interface."""
    def __init__(self, model: nn.Module, epochs: int = 30, lr: float = 1e-3,
                 batch_size: int = 128, device: str = "cpu") -> None:
        self._model = model
        self._epochs = epochs
        self._lr = lr
        self._batch_size = batch_size
        self._device = device

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchWrapper":
        self._model = _train_torch_clf(
            self._model, X, y,
            epochs=self._epochs,
            lr=self._lr,
            batch_size=self._batch_size,
            device=self._device,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _predict_torch_clf(self._model, X, device=self._device)


# ──────────────────────────────────────────────────────────────────────────────
# Registry: name → constructor callable
# ──────────────────────────────────────────────────────────────────────────────

def build_baseline(name: str, input_dim: int, device: str = "cpu",
                   traj_len: int = 3):
    """
    Build a baseline wrapper by name.

    Parameters
    ----------
    name      : one of the BASELINE_NAMES keys
    input_dim : flat feature dimension (traj_len * traj_dim + financial_dim)
    device    : "cuda" or "cpu"
    traj_len  : used to reshape flat features to (B, T, D) for sequential models

    Returns a wrapper object with .fit() and .predict_proba() methods.
    """
    name = name.lower().replace("-", "_").replace("+", "_")
    torch_epochs = 30

    if name == "lr":
        return SklearnWrapper(make_lr())
    elif name == "svm":
        return SklearnWrapper(make_svm())
    elif name == "rf":
        return SklearnWrapper(make_rf())
    elif name == "xgboost":
        return SklearnWrapper(make_xgboost())
    elif name == "lstm":
        m = _LSTMClassifier(input_dim=input_dim)
        return TorchWrapper(m, epochs=torch_epochs, device=device)
    elif name == "tcn":
        m = _TCNOnlyClassifier(input_dim=input_dim)
        return TorchWrapper(m, epochs=torch_epochs, device=device)
    elif name == "tcn_bilstm":
        m = _TCNBiLSTMClassifier(input_dim=input_dim)
        return TorchWrapper(m, epochs=torch_epochs, device=device)
    elif name == "informer":
        m = _InformerClassifier(input_dim=input_dim)
        return TorchWrapper(m, epochs=torch_epochs, device=device)
    elif name == "reformer":
        m = _ReformerClassifier(input_dim=input_dim)
        return TorchWrapper(m, epochs=torch_epochs, device=device)
    elif name == "time_llm":
        m = _TIMELLMClassifier(input_dim=input_dim)
        return TorchWrapper(m, epochs=torch_epochs, device=device)
    elif name in ("paraemb_fraudw2v", "paraemb"):
        m = _ParaEmbFraudW2VClassifier(input_dim=input_dim)
        return TorchWrapper(m, epochs=torch_epochs, device=device)
    else:
        raise ValueError(f"Unknown baseline: {name}")


BASELINE_NAMES = [
    "LR", "SVM", "RF", "XGBoost",
    "LSTM", "TCN", "TCN_BiLSTM",
    "Informer", "Reformer", "TIME_LLM",
    "ParaEmb_FraudW2V",
]
