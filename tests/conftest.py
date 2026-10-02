"""Synthetic, fully offline fixtures shaped exactly like the real loader outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from homesignal.config import (
    DataConfig,
    EvaluationConfig,
    FeatureConfig,
    ModelConfig,
    load_data_config,
    load_evaluation_config,
    load_feature_config,
    load_model_config,
)
from homesignal.data.acs import TABLE_CELLS, cell_name

ZIPS = [
    "94110",
    "78702",
    "33139",
    "48226",
    "10025",
    "85004",
    "02134",
    "60614",
    "97209",
    "30312",
    "80202",
    "98101",
]
STATES = ["CA", "TX", "FL", "MI", "NY", "AZ", "MA", "IL", "OR", "GA", "CO", "WA"]
METROS = [
    "SF",
    "Austin",
    "Miami",
    "Detroit",
    "NYC",
    "Phoenix",
    "Boston",
    "Chicago",
    "Portland",
    "Atlanta",
    "Denver",
    "Seattle",
]
MONTHS = pd.date_range("2008-01-01", "2026-08-01", freq="MS")


def make_zhvi(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(150_000, 900_000, size=len(ZIPS))
    drift = rng.normal(0.004, 0.002, size=len(ZIPS))
    noise = rng.normal(0, 0.01, size=(len(ZIPS), len(MONTHS)))
    logp = np.log(base)[:, None] + np.cumsum(drift[:, None] + noise, axis=1)
    wide = pd.DataFrame(np.exp(logp), index=pd.Index(ZIPS, name="zip"), columns=MONTHS)
    wide.iloc[0, :24] = np.nan  # one ZIP with a late start
    return wide


def make_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "size_rank": range(len(ZIPS)),
            "state": STATES,
            "city": METROS,
            "metro": [m + " metro" for m in METROS],
            "county_name": METROS,
        },
        index=pd.Index(ZIPS, name="zip"),
    )


def make_zori(zhvi: pd.DataFrame) -> pd.DataFrame:
    zori = (zhvi / 300.0).loc[ZIPS[:6], zhvi.columns >= "2015-01-01"]
    return zori


def make_acs_raw(vintages: list[int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for v in vintages:
        cols = {}
        for table, lines in TABLE_CELLS.items():
            if table == "B15003" and v < 2012 or table == "B23025" and v < 2012:
                continue
            for line in lines:
                cols[cell_name(table, line)] = rng.integers(100, 10_000, size=len(ZIPS)).astype(
                    float
                )
        df = pd.DataFrame(cols)
        df["B19013_001"] = rng.uniform(30_000, 150_000, size=len(ZIPS)) * (1 + 0.02 * (v - 2011))
        df["B25002_001"] = df["B25002_002"] + df["B25002_003"]
        df["B25003_001"] = df["B25003_002"] + df["B25003_003"]
        df.insert(0, "vintage", v)
        df.insert(0, "zcta", ZIPS)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def make_bls(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-01", "2026-08-01", freq="MS")
    rows = []
    for sid in ["LNS14000000", "CUUR0000SA0"] + [
        f"LASST{f}0000000000003"
        for f in ("06", "48", "12", "26", "36", "04", "25", "17", "41", "13", "08", "53")
    ]:
        level = 100.0 if sid.startswith("CUUR") else 5.0
        vals = level + np.cumsum(rng.normal(0, 0.1, size=len(months)))
        rows.append(pd.DataFrame({"series_id": sid, "month": months, "value": vals}))
    return pd.concat(rows, ignore_index=True)


def make_mortgage(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2000-01-01", "2026-09-01", freq="MS")
    return pd.DataFrame(
        {"month": months, "mortgage_rate_30y": 5 + np.cumsum(rng.normal(0, 0.05, size=len(months)))}
    )


@pytest.fixture
def data_cfg(tmp_path) -> DataConfig:
    cfg = load_data_config()
    cfg = cfg.model_copy(
        update={
            "raw_dir": tmp_path / "raw",
            "interim_dir": tmp_path / "interim",
            "processed_dir": tmp_path / "processed",
        }
    )
    cfg.panel.start_month = "2010-01"
    return cfg


@pytest.fixture
def feat_cfg() -> FeatureConfig:
    return load_feature_config()


@pytest.fixture
def eval_cfg() -> EvaluationConfig:
    return load_evaluation_config()


@pytest.fixture
def model_cfg() -> ModelConfig:
    cfg = load_model_config()
    cfg.lightgbm.update({"n_estimators": 30, "min_child_samples": 5, "num_leaves": 7})
    cfg.random_forest.update({"n_estimators": 10, "min_samples_leaf": 2})
    return cfg


@pytest.fixture
def synthetic_inputs(data_cfg: DataConfig) -> dict:
    zhvi = make_zhvi()
    return {
        "zhvi": zhvi,
        "meta": make_meta(),
        "zori": make_zori(zhvi),
        "acs_raw": make_acs_raw(data_cfg.acs.vintages),
        "bls_df": make_bls(),
        "mortgage": make_mortgage(),
    }
