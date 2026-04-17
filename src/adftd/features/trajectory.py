"""
Change trajectory construction.

For each company and consecutive year pair (t-1, t):
  1. Align paragraphs via cosine similarity of embeddings (ParaEmb).
  2. Classify each paragraph as: ADDED, DELETED, UPGRADED, DOWNGRADED.
  3. Aggregate FraudW2V category scores per change type.
  4. Produce Z_t of shape (n_change_types, n_word_categories) = (4, 9) = 36-dim flat vector.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from .fraud_w2v import WORD_CATEGORIES, score_paragraph

logger = logging.getLogger(__name__)

CHANGE_TYPES = ["added", "deleted", "upgraded", "downgraded"]
N_WORD_CATS = len(WORD_CATEGORIES)  # 9
N_CHANGE_TYPES = len(CHANGE_TYPES)  # 4
TRAJECTORY_DIM = N_CHANGE_TYPES * N_WORD_CATS  # 36


def _split_paragraphs(text: str, min_words: int = 20) -> List[str]:
    """Split MD&A text into paragraphs, filtering very short ones."""
    paras = [p.strip() for p in text.split("\n\n") if len(p.split()) >= min_words]
    if not paras:
        # Fallback: sentence-level split
        import re
        paras = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.split()) >= min_words]
    return paras


def _align_paragraphs(
    embeds_prev: np.ndarray,
    embeds_curr: np.ndarray,
    threshold: float = 0.7,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Greedy bipartite matching via cosine similarity.

    Returns (matched_prev, matched_curr, added_idx, deleted_idx).
    matched_prev[i] is aligned with matched_curr[i].
    """
    if embeds_prev.shape[0] == 0 or embeds_curr.shape[0] == 0:
        return [], [], list(range(len(embeds_curr))), list(range(len(embeds_prev)))

    sim = cosine_similarity(embeds_curr, embeds_prev)  # (n_curr, n_prev)
    matched_curr, matched_prev, added, deleted = [], [], [], []

    assigned_prev = set()
    for i in range(len(embeds_curr)):
        j = int(sim[i].argmax())
        if sim[i, j] >= threshold and j not in assigned_prev:
            matched_curr.append(i)
            matched_prev.append(j)
            assigned_prev.add(j)
        else:
            added.append(i)

    for j in range(len(embeds_prev)):
        if j not in assigned_prev:
            deleted.append(j)

    return matched_prev, matched_curr, added, deleted


def build_change_trajectory(
    text_prev: str,
    text_curr: str,
    embed_fn,                     # callable: List[str] -> np.ndarray
    word_categories: Optional[Dict] = None,
    sim_threshold: float = 0.7,
) -> np.ndarray:
    """
    Build change trajectory vector Z_t ∈ R^{36} for a company-year pair.

    embed_fn: function that takes List[str] and returns (N, embed_dim) array.
    Returns float32 vector of shape (TRAJECTORY_DIM,).
    """
    if word_categories is None:
        word_categories = WORD_CATEGORIES

    paras_prev = _split_paragraphs(text_prev)
    paras_curr = _split_paragraphs(text_curr)

    if not paras_prev or not paras_curr:
        return np.zeros(TRAJECTORY_DIM, dtype=np.float32)

    emb_prev = embed_fn(paras_prev)   # (n_prev, d)
    emb_curr = embed_fn(paras_curr)   # (n_curr, d)

    matched_prev, matched_curr, added_idx, deleted_idx = _align_paragraphs(
        emb_prev, emb_curr, threshold=sim_threshold
    )

    # Score per paragraph
    scores_prev = np.array([score_paragraph(p, word_categories) for p in paras_prev])
    scores_curr = np.array([score_paragraph(p, word_categories) for p in paras_curr])

    # Aggregate by change type
    agg = np.zeros((N_CHANGE_TYPES, N_WORD_CATS), dtype=np.float32)

    # ADDED paragraphs (new in curr)
    if added_idx:
        agg[0] = scores_curr[added_idx].mean(axis=0)

    # DELETED paragraphs (removed from prev)
    if deleted_idx:
        agg[1] = scores_prev[deleted_idx].mean(axis=0)

    # MATCHED → UPGRADED or DOWNGRADED based on mean score change
    for ip, ic in zip(matched_prev, matched_curr):
        delta = scores_curr[ic] - scores_prev[ip]
        if delta.mean() >= 0:
            agg[2] += delta  # UPGRADED
        else:
            agg[3] += np.abs(delta)  # DOWNGRADED
    if matched_prev:
        agg[2] /= len(matched_prev)
        agg[3] /= len(matched_prev)

    return agg.flatten().astype(np.float32)


def build_trajectory_sequence(
    texts_by_year: Dict[int, str],
    embed_fn,
    word_categories: Optional[Dict] = None,
    sim_threshold: float = 0.7,
    traj_len: int = 3,
) -> Optional[np.ndarray]:
    """
    Build (T, TRAJECTORY_DIM) trajectory matrix for a company.

    texts_by_year: {year: md_a_text}
    Returns None if insufficient data.
    """
    years = sorted(texts_by_year.keys())
    if len(years) < 2:
        return None

    # Build per-year change vectors
    annual: Dict[int, np.ndarray] = {}
    for i in range(1, len(years)):
        yr_prev, yr_curr = years[i - 1], years[i]
        # Only consecutive years
        if yr_curr - yr_prev != 1:
            continue
        annual[yr_curr] = build_change_trajectory(
            texts_by_year[yr_prev],
            texts_by_year[yr_curr],
            embed_fn,
            word_categories=word_categories,
            sim_threshold=sim_threshold,
        )

    result: Dict[int, np.ndarray] = {}
    for yr in sorted(annual.keys()):
        seq_years = [y for y in sorted(annual.keys()) if y <= yr][-traj_len:]
        if len(seq_years) < traj_len:
            # Pad with zeros at the start
            pad = traj_len - len(seq_years)
            vecs = [np.zeros(TRAJECTORY_DIM, dtype=np.float32)] * pad
            vecs += [annual[y] for y in seq_years]
        else:
            vecs = [annual[y] for y in seq_years]
        result[yr] = np.stack(vecs, axis=0)   # (T, TRAJECTORY_DIM)

    return result   # year -> (T, D)
