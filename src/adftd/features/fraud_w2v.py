"""
FraudW2V word-category scoring.

Assigns each paragraph a score for each of the 9 fraud-sensitive word categories
(6 Loughran-McDonald + 3 LIWC) using pre-built word lists.

The official LM wordlist: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
LIWC dictionaries require a licence; a freely-distributable approximation is
provided here using open LIWC-like word lists.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Built-in minimal word lists (representative subsets; replace with full lists)
# ──────────────────────────────────────────────────────────────────────────────

_LM_NEGATIVE = frozenset([
    "loss", "losses", "decline", "decreased", "impaired", "impairment",
    "adverse", "adversely", "uncertain", "risk", "risks", "fail", "failed",
    "failure", "default", "defaulted", "concern", "concerns", "negative",
    "deteriorated", "deterioration", "write-off", "write-down", "restructure",
    "restructuring", "downgrade", "litigation", "penalty", "penalties",
])

_LM_POSITIVE = frozenset([
    "growth", "increase", "increased", "improved", "improvement", "strong",
    "strengthened", "profitable", "profitability", "record", "exceeded",
    "exceed", "gains", "gain", "opportunity", "opportunities", "favorable",
    "favourable", "expansion", "expanding", "successful", "success",
])

_LM_UNCERTAINTY = frozenset([
    "uncertain", "uncertainty", "approximately", "estimate", "estimated",
    "believe", "believes", "possible", "possibly", "could", "might", "may",
    "contingent", "contingency", "variable", "varies", "fluctuate",
])

_LM_LITIGIOUS = frozenset([
    "litigation", "lawsuit", "legal", "court", "judge", "judgment",
    "settlement", "plaintiff", "defendant", "regulatory", "violation",
    "fraud", "alleged", "allegation", "investigation",
])

_LM_STRONG_MODAL = frozenset([
    "will", "must", "require", "requires", "required", "shall", "obligated",
    "certain", "always", "never",
])

_LM_WEAK_MODAL = frozenset([
    "could", "might", "may", "should", "suggest", "suggests", "appears",
    "seem", "seems", "likely", "unlikely", "possibly",
])

_LIWC_COMPARATIVES = frozenset([
    "more", "less", "greater", "smaller", "higher", "lower", "better",
    "worse", "larger", "fewer", "compared", "comparison", "relative",
    "versus", "vs",
])

_LIWC_REWARD = frozenset([
    "profit", "profits", "reward", "earn", "earns", "earned", "income",
    "gain", "bonus", "benefit", "benefits", "incentive", "dividend",
])

_LIWC_DISCREPANCY = frozenset([
    "should", "ought", "would", "need", "needs", "require", "requires",
    "expect", "expects", "expected", "discrepancy", "gap", "shortfall",
])

WORD_CATEGORIES: Dict[str, frozenset] = {
    "lm_negative": _LM_NEGATIVE,
    "lm_positive": _LM_POSITIVE,
    "lm_uncertainty": _LM_UNCERTAINTY,
    "lm_litigious": _LM_LITIGIOUS,
    "lm_strong_modal": _LM_STRONG_MODAL,
    "lm_weak_modal": _LM_WEAK_MODAL,
    "liwc_comparatives": _LIWC_COMPARATIVES,
    "liwc_reward": _LIWC_REWARD,
    "liwc_discrepancy": _LIWC_DISCREPANCY,
}


def score_paragraph(text: str, categories: Optional[Dict] = None) -> np.ndarray:
    """
    Compute word-category score vector for a single paragraph.
    Returns float32 array of shape (n_categories,) = (9,).
    Scores are normalised by total word count (avoids length bias).
    """
    if categories is None:
        categories = WORD_CATEGORIES
    tokens = re.findall(r"[a-z]+", text.lower())
    total = max(len(tokens), 1)
    scores = np.array(
        [sum(1 for t in tokens if t in wset) / total
         for wset in categories.values()],
        dtype=np.float32,
    )
    return scores


def score_document(paragraphs: List[str], categories: Optional[Dict] = None) -> np.ndarray:
    """
    Score all paragraphs in a document.
    Returns (n_paragraphs, n_categories) array.
    """
    return np.stack([score_paragraph(p, categories) for p in paragraphs])


def load_lm_wordlists(lm_csv_path: str) -> Dict[str, frozenset]:
    """
    Load the official Loughran-McDonald master dictionary CSV.
    CSV columns include: Word, Negative, Positive, Uncertainty, Litigious,
                         StrongModal, WeakModal, ...
    Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
    """
    import pandas as pd  # type: ignore
    df = pd.read_csv(lm_csv_path)
    df["Word"] = df["Word"].str.lower()

    mapping = {
        "lm_negative": "Negative",
        "lm_positive": "Positive",
        "lm_uncertainty": "Uncertainty",
        "lm_litigious": "Litigious",
        "lm_strong_modal": "StrongModal",
        "lm_weak_modal": "WeakModal",
    }
    updated: Dict[str, frozenset] = dict(WORD_CATEGORIES)
    for cat_key, col in mapping.items():
        if col in df.columns:
            words = df.loc[df[col] != 0, "Word"].tolist()
            updated[cat_key] = frozenset(words)
            logger.info("Loaded %d words for %s", len(words), cat_key)
    return updated
