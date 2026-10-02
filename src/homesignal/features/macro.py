"""Macro features (mortgage rates, unemployment, inflation) with explicit publication lags.

A series with lag ``k`` is shifted forward by ``k`` months before joining, so the value seen
at origin ``t`` is the one for month ``t - k`` — the latest print that was public at ``t``.
"""

from __future__ import annotations

import pandas as pd

from homesignal.config import FeatureConfig
from homesignal.data.crosswalks import STATE_FIPS_TO_ABBR

MACRO_FEATURES = [
    "mortgage_rate_30y",
    "mortgage_rate_chg_3m",
    "mortgage_rate_chg_12m",
    "national_unemployment",
    "national_unemployment_chg_12m",
    "cpi_yoy",
    "state_unemployment",
    "state_unemployment_chg_12m",
    "national_growth_12m",
]


def _lagged(series: pd.Series, lag: int) -> pd.Series:
    """Shift a month-indexed series forward by ``lag`` months.

    The value for month ``t - lag`` is what a user sees at origin ``t``.
    """
    s = series.sort_index()
    s.index = pd.DatetimeIndex(s.index) + pd.DateOffset(months=lag)
    return s


def national_macro_table(
    bls: pd.DataFrame, mortgage: pd.DataFrame, cfg: FeatureConfig
) -> pd.DataFrame:
    """Month-indexed national macro features (already lagged)."""
    lags = cfg.publication_lags
    mr = mortgage.set_index("month")["mortgage_rate_30y"]
    unemp = bls[bls["series_id"] == "LNS14000000"].set_index("month")["value"]
    cpi = bls[bls["series_id"] == "CUUR0000SA0"].set_index("month")["value"]
    out = pd.DataFrame(
        {
            "mortgage_rate_30y": _lagged(mr, lags.mortgage_rate),
            "mortgage_rate_chg_3m": _lagged(mr - mr.shift(3), lags.mortgage_rate),
            "mortgage_rate_chg_12m": _lagged(mr - mr.shift(12), lags.mortgage_rate),
            "national_unemployment": _lagged(unemp, lags.national_unemployment),
            "national_unemployment_chg_12m": _lagged(
                unemp - unemp.shift(12), lags.national_unemployment
            ),
            "cpi_yoy": _lagged((cpi / cpi.shift(12) - 1.0) * 100.0, lags.cpi),
        }
    )
    out.index.name = "month"
    return out


def state_macro_table(bls: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Long table ``state, month, state_unemployment, state_unemployment_chg_12m`` (lagged)."""
    lag = cfg.publication_lags.state_unemployment
    st = bls[bls["series_id"].str.startswith("LASST")].copy()
    st["state"] = st["series_id"].str[5:7].map(STATE_FIPS_TO_ABBR)
    frames = []
    for state, grp in st.groupby("state"):
        s = grp.set_index("month")["value"].sort_index()
        frames.append(
            pd.DataFrame(
                {
                    "state_unemployment": _lagged(s, lag),
                    "state_unemployment_chg_12m": _lagged(s - s.shift(12), lag),
                }
            ).assign(state=state)
        )
    out = pd.concat(frames).rename_axis("month").reset_index()
    return out[["state", "month", "state_unemployment", "state_unemployment_chg_12m"]]
