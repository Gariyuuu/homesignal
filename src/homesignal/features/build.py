"""Assemble the monthly modelling panel: one row per ZIP × month with target and all features."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from homesignal.config import DataConfig, FeatureConfig
from homesignal.data import acs, bls, freddie, zillow
from homesignal.data.crosswalks import zip_to_zcta
from homesignal.features.demographics import ACS_FEATURES, attach_demographics
from homesignal.features.geography import GEO_FEATURES, geography_table
from homesignal.features.lags import (
    forward_target,
    momentum_feature_names,
    momentum_features,
    pct_change,
)
from homesignal.features.macro import MACRO_FEATURES, national_macro_table, state_macro_table

log = logging.getLogger(__name__)

VALUATION_FEATURES = [
    "price_to_income",
    "price_to_rent_zori",
    "price_to_rent_acs",
    "zhvi_to_acs_value",
    "zhvi_rel_state",
    "zhvi_rel_metro",
    "rent_growth_12m_zori",
]
KEY_COLS = ["zip", "month"]
TARGET = "target_growth_12m"
META_COLS = ["zhvi", "acs_vintage"]


def feature_names(cfg: FeatureConfig) -> dict[str, list[str]]:
    """Feature names by group for a feature config."""
    return {
        "momentum": momentum_feature_names(
            cfg.momentum_horizons, cfg.volatility_window, cfg.peak_window
        ),
        "valuation": VALUATION_FEATURES,
        "demographics": ACS_FEATURES,
        "macro": MACRO_FEATURES,
        "geography": GEO_FEATURES,
    }


def all_feature_names(cfg: FeatureConfig) -> list[str]:
    """Flat list of all feature names."""
    return [n for names in feature_names(cfg).values() for n in names]


def _stack(name: str, wide: pd.DataFrame) -> pd.Series:
    s = cast(pd.Series, wide.stack())
    s.name = name
    return s


def _relative_to_group(wide: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Pct difference of each ZIP's value from the median of its group, per month."""
    med = wide.groupby(groups.reindex(wide.index)).transform("median")
    return (wide / med - 1.0) * 100.0


def build_panel(
    data_cfg: DataConfig, feat_cfg: FeatureConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load all sources and build the full monthly panel plus a coverage report."""
    zhvi, meta = zillow.load_zhvi(data_cfg)
    zori = zillow.load_zori(data_cfg)
    acs_raw = acs.load_acs(data_cfg)
    bls_df = bls.load_bls(data_cfg)
    mortgage = freddie.load_mortgage_rates(data_cfg)
    return assemble_panel(zhvi, meta, zori, acs_raw, bls_df, mortgage, data_cfg, feat_cfg)


def assemble_panel(
    zhvi: pd.DataFrame,
    meta: pd.DataFrame,
    zori: pd.DataFrame,
    acs_raw: pd.DataFrame,
    bls_df: pd.DataFrame,
    mortgage: pd.DataFrame,
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure panel assembly from in-memory inputs (used by the pipeline and by offline tests).

    ``zhvi``/``zori`` are wide (index ZIP, columns month-start); ``meta`` is indexed by ZIP;
    ``acs_raw`` is the stacked ACS cell table; ``bls_df`` is long (series_id, month, value);
    ``mortgage`` has ``month, mortgage_rate_30y``.
    """
    zori = zori.reindex(index=zhvi.index, columns=zhvi.columns)
    start = pd.Timestamp(data_cfg.panel.start_month)
    horizon = feat_cfg.horizon_months

    # --- wide computations (all trailing, except the target) -------------------------------
    wide_feats = momentum_features(
        zhvi, feat_cfg.momentum_horizons, feat_cfg.volatility_window, feat_cfg.peak_window
    )
    wide_feats["zhvi_rel_state"] = _relative_to_group(zhvi, meta["state"])
    wide_feats["zhvi_rel_metro"] = _relative_to_group(zhvi, meta["metro"])
    wide_feats["price_to_rent_zori"] = zhvi / (zori * 12.0)
    wide_feats["rent_growth_12m_zori"] = pct_change(zori, 12)
    target_wide = forward_target(zhvi, horizon)

    keep_cols = [c for c in zhvi.columns if pd.Timestamp(c) >= start]
    pieces = [_stack("zhvi", zhvi[keep_cols]), _stack(TARGET, target_wide[keep_cols])]
    pieces += [_stack(name, w[keep_cols]) for name, w in wide_feats.items()]
    panel = pd.concat(pieces, axis=1).reset_index()
    panel.columns = ["zip", "month", *panel.columns[2:]]
    panel = panel[panel["zhvi"].notna()].reset_index(drop=True)
    log.info("panel: %d rows after dropping months without ZHVI", len(panel))

    # --- national market regime: median 12m growth across ZIPs at t --------------------------
    nat = wide_feats["growth_12m"].median(axis=0)
    nat.name = "national_growth_12m"
    panel = panel.merge(nat.rename_axis("month").reset_index(), on="month", how="left")

    # --- demographics (as-of vintage) ---------------------------------------------------------
    zctas = set(acs_raw["zcta"].unique())
    mapping, coverage = zip_to_zcta(zhvi.index, zctas)
    panel = attach_demographics(panel, data_cfg, acs_raw, mapping)
    panel["price_to_income"] = panel["zhvi"] / panel["acs_median_income"]
    panel["price_to_rent_acs"] = panel["zhvi"] / (panel["acs_median_gross_rent"] * 12.0)
    panel["zhvi_to_acs_value"] = panel["zhvi"] / panel["acs_median_home_value"]

    # --- macro ---------------------------------------------------------------------------------
    panel = panel.merge(
        national_macro_table(bls_df, mortgage, feat_cfg).reset_index(), on="month", how="left"
    )
    geo = geography_table(meta, feat_cfg.metro_size_buckets)
    panel = panel.merge(geo.reset_index(), on="zip", how="left")
    panel = panel.merge(state_macro_table(bls_df, feat_cfg), on=["state", "month"], how="left")

    # --- tidy ----------------------------------------------------------------------------------
    names = all_feature_names(feat_cfg)
    panel = (
        panel[[*KEY_COLS, TARGET, *META_COLS, *names]].sort_values(KEY_COLS).reset_index(drop=True)
    )
    numeric = [c for c in names if c not in GEO_FEATURES]
    panel[numeric] = panel[numeric].replace([np.inf, -np.inf], np.nan).astype("float32")
    panel[TARGET] = panel[TARGET].astype("float32")
    panel["zhvi"] = panel["zhvi"].astype("float32")

    report = coverage_report(panel, zhvi, coverage, feat_cfg)
    return panel, report


def coverage_report(
    panel: pd.DataFrame, zhvi: pd.DataFrame, zcta_coverage: dict[str, float], cfg: FeatureConfig
) -> dict[str, Any]:
    """Summarise geography coverage and missingness so the docs can report it honestly."""
    recent = [c for c in zhvi.columns if pd.Timestamp(c) >= pd.Timestamp("2012-01-01")]
    complete = zhvi[recent].notna().all(axis=1).mean()
    names = all_feature_names(cfg)
    missing = {c: float(panel[c].isna().mean()) for c in names}
    modelable = panel[panel[TARGET].notna()]
    return {
        "n_zips": int(zhvi.shape[0]),
        "zhvi_months": [
            str(pd.Timestamp(zhvi.columns[0]).date()),
            str(pd.Timestamp(zhvi.columns[-1]).date()),
        ],
        "share_zips_complete_zhvi_since_2012": float(complete),
        "zip_to_zcta": zcta_coverage,
        "panel_rows": int(len(panel)),
        "rows_with_target": int(len(modelable)),
        "panel_months": [str(panel["month"].min().date()), str(panel["month"].max().date())],
        "feature_missing_share": missing,
        "target_summary": {k: float(v) for k, v in modelable[TARGET].describe().items()},
    }


def save_panel(panel: pd.DataFrame, report: dict[str, Any], processed_dir: Path) -> Path:
    """Write the panel parquet and coverage JSON."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "panel.parquet"
    panel.to_parquet(out, index=False)
    (processed_dir / "coverage.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %s (%d rows, %d cols)", out, *panel.shape)
    return out


def load_panel(processed_dir: Path) -> pd.DataFrame:
    """Load the processed panel."""
    return pd.read_parquet(processed_dir / "panel.parquet")
