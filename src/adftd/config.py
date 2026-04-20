"""
AD-FTD Configuration
All hyperparameters and paths are centralised here.
"""
from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


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
class AWSConfig:
    """AWS / S3 configuration for cloud execution."""
    s3_bucket: Optional[str] = None      # e.g. "my-adftd-bucket"; None = local only
    s3_prefix: str = "adftd"             # key prefix inside the bucket
    region: str = "us-east-1"
    num_gpus: int = 1                    # 1 = single GPU; >1 = DataParallel


# ─────────────────────────────────────────────────────────────────────────────
# S3 I/O helpers  (boto3 is an optional dep; errors are logged, not raised)
# ─────────────────────────────────────────────────────────────────────────────

def s3_upload(local_path: str, bucket: str, key: str,
              region: str = "us-east-1") -> None:
    """Upload a local file to S3.  Silently logs on failure."""
    try:
        import boto3  # type: ignore
        boto3.client("s3", region_name=region).upload_file(
            str(local_path), bucket, key
        )
        logger.info("S3 upload: %s → s3://%s/%s", local_path, bucket, key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 upload failed (%s → s3://%s/%s): %s",
                        local_path, bucket, key, exc)


def s3_download(bucket: str, key: str, local_path: str,
                region: str = "us-east-1") -> bool:
    """Download an S3 object to a local path.  Returns True on success."""
    try:
        import boto3  # type: ignore
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3", region_name=region).download_file(
            bucket, key, str(local_path)
        )
        logger.info("S3 download: s3://%s/%s → %s", bucket, key, local_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("S3 download failed (s3://%s/%s → %s): %s",
                        bucket, key, local_path, exc)
        return False


def resolve_s3_path(path: str, local_cache_dir: str = "/tmp/adftd_cache",
                    region: str = "us-east-1") -> str:
    """
    If *path* starts with ``s3://``, download it to *local_cache_dir* and
    return the local path.  Otherwise return *path* unchanged.
    """
    if not path.startswith("s3://"):
        return path
    without_scheme = path[5:]
    bucket, _, key = without_scheme.partition("/")
    local = os.path.join(local_cache_dir, os.path.basename(key))
    s3_download(bucket, key, local, region)
    return local


@dataclass
class ADFTDConfig:
    data: DataConfig = field(default_factory=DataConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)

    def __post_init__(self) -> None:
        # Sync trajectory_dim from feature config to model config
        self.model.trajectory_dim = self.features.trajectory_dim
        self.model.financial_dim = self.features.financial_dim
