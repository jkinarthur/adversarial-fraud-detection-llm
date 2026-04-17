"""
AAER (Accounting and Auditing Enforcement Release) matcher.

Maps SEC AAER fraud labels to company CIKs/GVKEYs and fiscal years.

Expected CSV format (aaer_labels.csv):
    gvkey, cik, company_name, fyear, aaer_num, misstatement_start, misstatement_end
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

# Fraud label: (cik, year) -> 1 if fraudulent, else 0
FraudLabels = Dict[Tuple[str, int], int]


def load_aaer_labels(aaer_csv: str) -> FraudLabels:
    """
    Returns a dict mapping (cik, year) -> 1 for firm-years covered by an AAER.

    The misstatement period (misstatement_start..misstatement_end) is used
    to assign fraud labels, NOT the AAER release date, to align detection
    with the period when fraud actually occurred.
    """
    path = Path(aaer_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"AAER labels CSV not found: {path}\n"
            "Download from: https://www.sec.gov/divisions/enforce/friactions.htm\n"
            "and convert to CSV with columns: gvkey, cik, fyear, "
            "misstatement_start, misstatement_end, aaer_num"
        )

    df = pd.read_csv(path, dtype=str)
    required = {"cik", "fyear"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AAER CSV missing columns: {missing}")

    labels: FraudLabels = {}

    for _, row in df.iterrows():
        cik = str(row["cik"]).strip().lstrip("0")
        # Use misstatement range when available, else single fiscal year
        start = _parse_year(row.get("misstatement_start", row["fyear"]))
        end = _parse_year(row.get("misstatement_end", row["fyear"]))
        if start is None or end is None:
            continue
        for yr in range(start, end + 1):
            labels[(cik, yr)] = 1

    logger.info("Loaded %d fraud firm-year observations from %s", len(labels), path)
    return labels


def get_label(labels: FraudLabels, cik: str, year: int) -> int:
    cik_norm = str(cik).strip().lstrip("0")
    return labels.get((cik_norm, year), 0)


def _parse_year(val) -> Optional[int]:
    try:
        return int(str(val)[:4])
    except (ValueError, TypeError):
        return None
