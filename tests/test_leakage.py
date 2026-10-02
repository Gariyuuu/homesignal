"""Leakage tests: no feature at origin t may depend on data published after t."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from homesignal.data.acs import vintage_available_from
from homesignal.features.build import TARGET, all_feature_names, assemble_panel
from homesignal.features.geography import GEO_FEATURES

T = pd.Timestamp("2019-06-01")


def _perturb_after(inputs: dict, data_cfg, cutoff: pd.Timestamp, seed: int = 1) -> dict:
    """Return a copy of the inputs where everything strictly after ``cutoff`` is scrambled."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in inputs.items()}
    for key in ("zhvi", "zori"):
        w = out[key]
        future = w.columns > cutoff
        w.loc[:, future] = w.loc[:, future] * rng.uniform(0.5, 1.5, size=(len(w), future.sum()))
    b = out["bls_df"]
    mask = b["month"] > cutoff
    b.loc[mask, "value"] = b.loc[mask, "value"] * rng.uniform(0.5, 1.5, size=mask.sum())
    m = out["mortgage"]
    mask = m["month"] > cutoff
    m.loc[mask, "mortgage_rate_30y"] = m.loc[mask, "mortgage_rate_30y"] + rng.uniform(
        1, 3, size=mask.sum()
    )
    a = out["acs_raw"]
    not_yet = a["vintage"].map(lambda v: vintage_available_from(data_cfg, int(v)) > cutoff)
    cells = [c for c in a.columns if c not in ("zcta", "vintage")]
    a.loc[not_yet, cells] = a.loc[not_yet, cells] * rng.uniform(
        0.5, 1.5, size=(int(not_yet.sum()), len(cells))
    )
    return out


def test_no_feature_depends_on_future_data(synthetic_inputs, data_cfg, feat_cfg):
    panel, _ = assemble_panel(**synthetic_inputs, data_cfg=data_cfg, feat_cfg=feat_cfg)
    perturbed = _perturb_after(synthetic_inputs, data_cfg, T)
    panel2, _ = assemble_panel(**perturbed, data_cfg=data_cfg, feat_cfg=feat_cfg)
    features = all_feature_names(feat_cfg)
    past = panel["month"] <= T
    a = panel.loc[past].set_index(["zip", "month"])[features]
    b = panel2.loc[past].set_index(["zip", "month"])[features]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6)
    # Sanity: the target at those rows DID change, so the perturbation was real.
    ta = panel.loc[past & (panel["month"] > T - pd.DateOffset(months=12)), TARGET]
    tb = panel2.loc[past & (panel2["month"] > T - pd.DateOffset(months=12)), TARGET]
    assert not np.allclose(ta.dropna(), tb.dropna())


@pytest.mark.parametrize("lag_key", ["cpi", "national_unemployment", "state_unemployment"])
def test_macro_publication_lag_respected(synthetic_inputs, data_cfg, feat_cfg, lag_key):
    """A series with lag k must not react to its own value for months > t - k."""
    lag = getattr(feat_cfg.publication_lags, lag_key)
    cutoff = T - pd.DateOffset(months=lag)
    series_prefix = {
        "cpi": "CUUR",
        "national_unemployment": "LNS14",
        "state_unemployment": "LASST",
    }[lag_key]
    feature = {
        "cpi": "cpi_yoy",
        "national_unemployment": "national_unemployment",
        "state_unemployment": "state_unemployment",
    }[lag_key]
    inputs2 = {k: v.copy() for k, v in synthetic_inputs.items()}
    b = inputs2["bls_df"]
    mask = b["series_id"].str.startswith(series_prefix) & (b["month"] > cutoff)
    b.loc[mask, "value"] = b.loc[mask, "value"] + 100.0
    p1, _ = assemble_panel(**synthetic_inputs, data_cfg=data_cfg, feat_cfg=feat_cfg)
    p2, _ = assemble_panel(**inputs2, data_cfg=data_cfg, feat_cfg=feat_cfg)
    at_t = p1["month"] == T
    np.testing.assert_allclose(p1.loc[at_t, feature], p2.loc[at_t, feature], rtol=1e-6)
    after = p1["month"] == T + pd.DateOffset(months=lag)
    assert not np.allclose(p1.loc[after, feature], p2.loc[after, feature])


def test_acs_vintage_assignment_respects_release_dates(synthetic_inputs, data_cfg, feat_cfg):
    panel, _ = assemble_panel(**synthetic_inputs, data_cfg=data_cfg, feat_cfg=feat_cfg)
    used = panel.dropna(subset=["acs_vintage"])
    for v, grp in used.groupby("acs_vintage"):
        assert grp["month"].min() >= vintage_available_from(data_cfg, int(v))
    # Before the first vintage is available there must be no demographics at all.
    first = min(vintage_available_from(data_cfg, v) for v in data_cfg.acs.vintages)
    early = panel[panel["month"] < first]
    assert early["acs_median_income"].isna().all()
    assert early["acs_vintage"].isna().all()


def test_target_is_never_imputed(synthetic_inputs, data_cfg, feat_cfg):
    panel, _ = assemble_panel(**synthetic_inputs, data_cfg=data_cfg, feat_cfg=feat_cfg)
    last = panel["month"].max()
    tail = panel[panel["month"] > last - pd.DateOffset(months=feat_cfg.horizon_months)]
    assert tail[TARGET].isna().all()


def test_no_raw_identifiers_in_features(feat_cfg):
    names = all_feature_names(feat_cfg)
    for banned in ("zip", "zcta", "metro", "county_name", "city", "month", "year", "size_rank"):
        assert banned not in names
    assert set(GEO_FEATURES) <= set(names)
