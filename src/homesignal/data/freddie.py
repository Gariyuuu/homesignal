"""Freddie Mac Primary Mortgage Market Survey (weekly 30-year fixed rate, 1971 →)."""

from __future__ import annotations

import logging

import pandas as pd

from homesignal.config import DataConfig
from homesignal.data.download import fetch

log = logging.getLogger(__name__)


def load_mortgage_rates(cfg: DataConfig) -> pd.DataFrame:
    """Return ``month, mortgage_rate_30y`` = mean of the weekly surveys published in that month."""
    path = fetch(
        cfg.freddie.pmms_url,
        cfg.raw_dir / "freddie" / "PMMS_history.csv",
        user_agent=cfg.user_agent,
    )
    df = pd.read_csv(path, usecols=["date", "pmms30"])
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df["pmms30"] = pd.to_numeric(df["pmms30"], errors="coerce")
    df = df.dropna(subset=["pmms30"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    out = df.groupby("month")["pmms30"].mean().rename("mortgage_rate_30y").reset_index()
    log.info(
        "PMMS: %d months (%s → %s)", len(out), out["month"].min().date(), out["month"].max().date()
    )
    return out
