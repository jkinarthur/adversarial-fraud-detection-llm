"""
EDGAR-CORPUS loader.
Loads 10-K MD&A sections from the HuggingFace EDGAR-CORPUS dataset.
Dataset: https://huggingface.co/datasets/eloukas/edgar-corpus
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def iter_edgar_filings(
    cache_dir: str = "data/raw/edgar",
    start_year: int = 1995,
    end_year: int = 2019,
    section: str = "section_7",   # MD&A
) -> Iterator[Dict]:
    """
    Yields dicts with keys: cik, company_name, filing_date, year, text.

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("Install 'datasets': pip install datasets") from exc

    for year in range(start_year, end_year + 1):
        logger.info("Loading EDGAR-CORPUS year %d ...", year)
        try:
            ds = load_dataset(
                "eloukas/edgar-corpus",
                name=str(year),
                split="train",
                cache_dir=cache_dir,
            )
        except Exception as exc:          # noqa: BLE001
            logger.warning("Could not load year %d: %s", year, exc)
            continue

        for row in ds:
            text = row.get(section, "") or ""
            if not text.strip():
                continue
            yield {
                "cik": str(row.get("cik", "")),
                "company_name": row.get("company_name", ""),
                "filing_date": row.get("period_of_report", ""),
                "year": year,
                "text": text,
            }


def load_edgar_cache(features_dir: str) -> Dict:
    """
    Load pre-saved EDGAR text cache (cik -> {year -> text}).
    Expects Parquet files saved by preprocess.py.
    """
    import pandas as pd  # type: ignore

    cache: Dict = {}
    p = Path(features_dir)
    for fp in p.glob("edgar_text_*.parquet"):
        df = pd.read_parquet(fp)
        for _, row in df.iterrows():
            cik = str(row["cik"])
            year = int(row["year"])
            cache.setdefault(cik, {})[year] = row["text"]
    return cache
