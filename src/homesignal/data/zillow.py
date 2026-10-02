"""Zillow Research ZHVI / ZORI loaders.

Zillow publishes one wide CSV per index with a row per region and a column per month-end.
We keep the wide layout (index = ZIP, columns = month start timestamps) because the momentum
features are cheap to compute as column shifts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from homesignal.config import DataConfig
from homesignal.data.download import fetch

log = logging.getLogger(__name__)

META_COLS = [
    "RegionID",
    "SizeRank",
    "RegionName",
    "RegionType",
    "StateName",
    "State",
    "City",
    "Metro",
    "CountyName",
]


def _month_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if len(c) == 10 and c[4] == "-" and c[7] == "-"]


def _to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide frame indexed by 5-digit ZIP with month-start Timestamp columns."""
    month_cols = _month_columns(df)
    wide = df.set_index("RegionName")[month_cols].astype("float64")
    wide.columns = pd.to_datetime(wide.columns).to_period("M").to_timestamp()
    wide.index = wide.index.astype(str).str.zfill(5)
    wide.index.name = "zip"
    wide = wide[~wide.index.duplicated()]
    return wide.sort_index(axis=1)


def _read_zillow_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"RegionName": str})


def load_zhvi(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (if needed) and load the ZIP-level ZHVI.

    Returns ``(wide_values, metadata)`` where ``metadata`` has one row per ZIP with
    ``state``, ``metro``, ``county_name``, ``city`` and ``size_rank``.
    """
    path = fetch(
        cfg.zillow.zhvi_zip_url, cfg.raw_dir / "zillow" / "zhvi_zip.csv", user_agent=cfg.user_agent
    )
    df = _read_zillow_csv(path)
    wide = _to_wide(df)
    meta = (
        df[[c for c in META_COLS if c in df.columns]]
        .rename(
            columns={
                "RegionName": "zip",
                "State": "state",
                "Metro": "metro",
                "CountyName": "county_name",
                "City": "city",
                "SizeRank": "size_rank",
            }
        )
        .assign(zip=lambda d: d["zip"].astype(str).str.zfill(5))
        .drop(columns=["RegionID", "RegionType", "StateName"], errors="ignore")
        .drop_duplicates("zip")
        .set_index("zip")
        .loc[wide.index]
    )
    log.info(
        "ZHVI: %d ZIPs x %d months (%s → %s)",
        *wide.shape,
        pd.Timestamp(wide.columns[0]).date(),
        pd.Timestamp(wide.columns[-1]).date(),
    )
    return wide, meta


def load_zori(cfg: DataConfig) -> pd.DataFrame:
    """Download (if needed) and load the ZIP-level ZORI as a wide frame."""
    path = fetch(
        cfg.zillow.zori_zip_url, cfg.raw_dir / "zillow" / "zori_zip.csv", user_agent=cfg.user_agent
    )
    wide = _to_wide(_read_zillow_csv(path))
    log.info("ZORI: %d ZIPs x %d months", *wide.shape)
    return wide
