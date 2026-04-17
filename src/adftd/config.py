"""
AD-FTD Configuration
All hyperparameters and paths are centralised here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    # EDGAR-CORPUS (HuggingFace dataset)
    edgar_cache_dir: str = "data/raw/edgar"
    # AAER CSV downloaded from SEC (columns: gvkey, fyear, aaer_num)
    aaer_csv: str = "data/raw/aaer_labels.csv"
    # Compustat financial ratios CSV (columns: gvkey, fyear, ratio_*)
    compustat_csv: str = "data/raw/compustat_ratios.csv"
    # Preprocessed trajectory tensors (saved after feature extraction)
    features_dir: str = "data/features"
    # Years included
    start_year: int = 1995
    end_year: int = 2019
    # Trajectory length T
    traj_len: int = 3
    # Fraud oversampling ratio
    oversample_ratio: int = 10
    # Cross-validation folds
    n_folds: int = 4
    # Resampling iterations
    n_resample: int = 10
    # Minimum peer-group size before SIC super-group fallback
    min_peer_size: int = 5


@dataclass
class EmbedderConfig:
    backend: str = "openai"          # "openai" | "llama3"
    openai_model: str = "text-embedding-3-small"   # or "text-embedding-ada-002"
    openai_batch_size: int = 100
    openai_max_retries: int = 5
    llama3_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    llama3_load_in_4bit: bool = True
    llama3_batch_size: int = 8
    embed_dim: int = 1536            # GPT-3.5 ada-002 / text-embedding-3-small


@dataclass
class FeatureConfig:
    # FraudW2V word categories (6 Loughran-McDonald + 3 LIWC)
    word_categories: List[str] = field(default_factory=lambda: [
        "lm_negative", "lm_positive", "lm_uncertainty",
        "lm_litigious", "lm_strong_modal", "lm_weak_modal",
        "liwc_comparatives", "liwc_reward", "liwc_discrepancy",
    ])
    # Paragraph change types
    change_types: List[str] = field(default_factory=lambda: [
        "added", "deleted", "upgraded", "downgraded",
    ])
    # Derived: n_categories × n_change_types = 9 × 4 = 36
    @property
    def trajectory_dim(self) -> int:
        return len(self.word_categories) * len(self.change_types)

    # Number of financial ratio features (Compustat)
    financial_dim: int = 9
    # Similarity threshold for paragraph alignment
    para_sim_threshold: float = 0.7


@dataclass
class ModelConfig:
    # TCN (shared between generator and detector)
    tcn_hidden: int = 128
    tcn_levels: int = 4
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2
    # BiLSTM detector
    bilstm_hidden: int = 128
    bilstm_layers: int = 2
    bilstm_dropout: float = 0.2
    # Counterfactual generator output matches trajectory_dim
    # (set dynamically from FeatureConfig)
    trajectory_dim: int = 36
    financial_dim: int = 9


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    # Loss weights
    alpha: float = 0.5     # deviation regularisation
    beta: float = 0.3      # adversarial loss
    # FGSM perturbation magnitude
    epsilon: float = 0.01
    # Cost-sensitive evaluation thresholds
    cost_ratios: List[float] = field(default_factory=lambda: [100.0, 500.0, 1000.0])
    device: str = "cuda"   # falls back to cpu if unavailable
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class ADFTDConfig:
    data: DataConfig = field(default_factory=DataConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        # Sync trajectory_dim from feature config to model config
        self.model.trajectory_dim = self.features.trajectory_dim
        self.model.financial_dim = self.features.financial_dim
