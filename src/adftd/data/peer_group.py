"""
Peer-group construction.

Industry: 2-digit SIC code.
Size:     decile rank of total assets within SIC × year cell.
Fallback: 1-digit SIC super-group when peer count < min_peer_size.

Groups are built ONLY from the training partition to prevent look-ahead bias.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)

# (cik, year) -> list of peer ciks
PeerMap = Dict[Tuple[str, int], List[str]]


def build_peer_groups(
    compustat_df: pd.DataFrame,
    train_ciks: Optional[List[str]] = None,
    min_peer_size: int = 5,
) -> PeerMap:
    """
    Build peer groups from a Compustat DataFrame.

    Expected columns: cik, gvkey, fyear, sic, at (total assets).
    If train_ciks is provided, peer groups are restricted to those CIKs
    (call this with training-set CIKs only to avoid look-ahead).

    Returns PeerMap: (cik, year) -> [peer_cik, ...]
    """
    df = compustat_df.copy()
    df["cik"] = df["cik"].astype(str).str.strip().str.lstrip("0")
    df["fyear"] = df["fyear"].astype(int)
    df["sic2"] = df["sic"].astype(str).str.zfill(4).str[:2]
    df["sic1"] = df["sic"].astype(str).str.zfill(4).str[:1]
    df["at"] = pd.to_numeric(df["at"], errors="coerce")

    if train_ciks is not None:
        train_set = {str(c).strip().lstrip("0") for c in train_ciks}
        df_train = df[df["cik"].isin(train_set)]
    else:
        df_train = df

    # Assign asset decile per sic2 × year
    def _decile(series: pd.Series) -> pd.Series:
        if series.isna().all():
            return pd.Series(5, index=series.index)   # midpoint fallback
        return pd.qcut(series.rank(method="first"), q=10, labels=False,
                       duplicates="drop").fillna(5).astype(int)

    df_train = df_train.copy()
    df_train["asset_decile"] = (
        df_train.groupby(["sic2", "fyear"])["at"]
        .transform(_decile)
    )

    peer_map: PeerMap = {}

    for _, row in df.iterrows():
        cik = row["cik"]
        year = int(row["fyear"])
        sic2 = row["sic2"]
        sic1 = row["sic1"]
        decile = _get_decile(df_train, cik, year)

        # Primary: same sic2 + same asset decile
        mask = (
            (df_train["sic2"] == sic2) &
            (df_train["fyear"] == year) &
            (df_train["asset_decile"] == decile) &
            (df_train["cik"] != cik)
        )
        peers = df_train.loc[mask, "cik"].tolist()

        # Fallback: same sic1
        if len(peers) < min_peer_size:
            mask_fallback = (
                (df_train["sic1"] == sic1) &
                (df_train["fyear"] == year) &
                (df_train["cik"] != cik)
            )
            peers = df_train.loc[mask_fallback, "cik"].tolist()

        peer_map[(cik, year)] = peers

    return peer_map


def _get_decile(df_train: pd.DataFrame, cik: str, year: int) -> int:
    rows = df_train[(df_train["cik"] == cik) & (df_train["fyear"] == year)]
    if rows.empty or "asset_decile" not in rows.columns:
        return 5
    return int(rows["asset_decile"].iloc[0])


def get_peers(peer_map: PeerMap, cik: str, year: int) -> List[str]:
    cik_norm = str(cik).strip().lstrip("0")
    return peer_map.get((cik_norm, year), [])
