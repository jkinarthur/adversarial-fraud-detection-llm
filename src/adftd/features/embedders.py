"""
Paragraph embedders: GPT-3.5 (OpenAI) and Llama-3-8B-Instruct (local, 4-bit).

Both implement the same interface:
    embedder.embed(texts: List[str]) -> np.ndarray  shape (N, embed_dim)
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (N, embed_dim) float32 array."""


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI / GPT-3.5
# ──────────────────────────────────────────────────────────────────────────────
class OpenAIEmbedder(BaseEmbedder):
    """
    Wraps OpenAI embedding API.
    Set OPENAI_API_KEY environment variable before use.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 5,
    ) -> None:
        try:
            import openai  # type: ignore
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable not set."
            )
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed(self, texts: List[str]) -> np.ndarray:
        results: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            emb = self._embed_batch(batch)
            results.append(emb)
        return np.vstack(results).astype(np.float32)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        for attempt in range(self.max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self.model, input=texts
                )
                vectors = [d.embedding for d in sorted(
                    response.data, key=lambda x: x.index
                )]
                return np.array(vectors, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                wait = 2 ** attempt
                logger.warning("OpenAI API error (attempt %d): %s; retrying in %ds",
                               attempt + 1, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"OpenAI embedding failed after {self.max_retries} retries")


# ──────────────────────────────────────────────────────────────────────────────
# Llama-3-8B-Instruct (4-bit, local, zero API cost)
# ──────────────────────────────────────────────────────────────────────────────
class Llama3Embedder(BaseEmbedder):
    """
    Mean-pool the last hidden states of Llama-3-8B-Instruct as paragraph embeddings.
    Requires: pip install transformers bitsandbytes accelerate

    ~1.2 pp PRC-AUC trade-off vs GPT-3.5 per paper ablation.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_4bit: bool = True,
        batch_size: int = 8,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Install transformers + bitsandbytes: "
                "pip install transformers bitsandbytes accelerate"
            ) from exc

        import os
        hf_token = os.environ.get("HF_TOKEN")

        quant_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        logger.info("Loading Llama-3 model %s (4-bit=%s) ...", model_id, load_in_4bit)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=False
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModel.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=False,
        )
        self._model.eval()
        self.batch_size = batch_size
        self._torch = torch

    def embed(self, texts: List[str]) -> np.ndarray:
        import torch
        results: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            enc = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model(**enc)
            # Mean-pool last hidden state over non-padding tokens
            mask = enc["attention_mask"].unsqueeze(-1).float()
            hidden = out.last_hidden_state * mask
            mean_emb = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            results.append(mean_emb.float().cpu().numpy())
        return np.vstack(results).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
def build_embedder(backend: str, **kwargs) -> BaseEmbedder:
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    if backend == "llama3":
        return Llama3Embedder(**kwargs)
    raise ValueError(f"Unknown embedder backend '{backend}'. Choose 'openai' or 'llama3'.")
