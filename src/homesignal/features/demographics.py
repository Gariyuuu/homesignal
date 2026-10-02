"""ACS demographic features assigned to the months in which each vintage was actually available."""

from __future__ import annotations

import pandas as pd

from homesignal.config import DataConfig
from homesignal.data.acs import derive_acs_features, vintage_available_from

ACS_LEVEL_FEATURES = [
    "acs_population",
    "acs_median_age",
    "acs_share_under18",
    "acs_share_25_34",
    "acs_share_65plus",
    "acs_bachelor_share",
    "acs_median_income",
    "acs_housing_units",
    "acs_vacancy_rate",
    "acs_renter_share",
    "acs_median_gross_rent",
    "acs_median_home_value",
    "acs_unemployment_rate",
    "acs_lfpr",
]
ACS_GROWTH_FEATURES = ["acs_pop_growth", "acs_income_growth", "acs_housing_unit_growth"]
ACS_FEATURES = ACS_LEVEL_FEATURES + ACS_GROWTH_FEATURES


def acs_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per ``zcta × vintage`` with level and vintage-over-vintage growth features.

    Growth features compare a vintage with the previous one (two 5-year windows offset by
    one year), so they are slow-moving proxies rather than true annual growth rates.
    """
    feats = derive_acs_features(raw).sort_values(["zcta", "vintage"])
    prev = feats.groupby("zcta")[
        ["acs_population", "acs_median_income", "acs_housing_units", "vintage"]
    ].shift(1)
    consecutive = (feats["vintage"] - prev["vintage"]) == 1
    for name, col in [
        ("acs_pop_growth", "acs_population"),
        ("acs_income_growth", "acs_median_income"),
        ("acs_housing_unit_growth", "acs_housing_units"),
    ]:
        growth = (feats[col] / prev[col] - 1.0) * 100.0
        feats[name] = growth.where(consecutive & (prev[col] > 0))
    return feats


def vintage_as_of(cfg: DataConfig, months: pd.DatetimeIndex) -> pd.Series:
    """For each month, the newest ACS vintage available for use (NaN if none yet)."""
    avail = sorted((vintage_available_from(cfg, v), v) for v in cfg.acs.vintages)
    out = pd.Series(float("nan"), index=months, name="acs_vintage")
    for from_month, v in avail:
        out[months >= from_month] = v
    return out


def attach_demographics(
    panel: pd.DataFrame, cfg: DataConfig, acs_raw: pd.DataFrame, zip_to_zcta: pd.Series
) -> pd.DataFrame:
    """Join ACS features onto a long panel (``zip``, ``month``) using as-of vintages."""
    feats = acs_feature_table(acs_raw)
    months = pd.DatetimeIndex(sorted(panel["month"].unique()))
    vintage_map = vintage_as_of(cfg, months)
    panel = panel.copy()
    panel["acs_vintage"] = panel["month"].map(vintage_map)
    panel["zcta"] = panel["zip"].map(zip_to_zcta)
    merged = panel.merge(
        feats[["zcta", "vintage", *ACS_FEATURES]].rename(columns={"vintage": "acs_vintage"}),
        on=["zcta", "acs_vintage"],
        how="left",
    )
    return merged.drop(columns=["zcta"])
