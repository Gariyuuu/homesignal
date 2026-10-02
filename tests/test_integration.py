"""Offline end-to-end run on synthetic data: panel → modelling frame → backtest → final model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from homesignal.evaluation.backtest import fit_predict_fold
from homesignal.evaluation.metrics import regression_metrics
from homesignal.evaluation.splits import backtest_folds
from homesignal.features.build import TARGET, all_feature_names, assemble_panel, save_panel
from homesignal.models.baselines import TrailingGrowthBaseline
from homesignal.models.dataset import load_modeling_frame
from homesignal.models.predict import explain_row
from homesignal.models.registry import make_model


def test_pipeline_end_to_end_and_reproducible(
    synthetic_inputs, data_cfg, feat_cfg, eval_cfg, model_cfg
):
    panel, report = assemble_panel(**synthetic_inputs, data_cfg=data_cfg, feat_cfg=feat_cfg)
    assert set(all_feature_names(feat_cfg)) <= set(panel.columns)
    assert report["rows_with_target"] > 0
    save_panel(panel, report, data_cfg.processed_dir)

    frame = load_modeling_frame(data_cfg, feat_cfg, eval_cfg, model_cfg)
    assert frame.df["month"].dt.month.isin(eval_cfg.origin_months).all()
    assert frame.df[TARGET].notna().all()

    fold = backtest_folds(eval_cfg)[-1]
    results = []
    for _ in range(2):
        out, _elapsed = fit_predict_fold(frame, fold, "lightgbm", model_cfg)
        results.append(out["pred"].to_numpy())
    np.testing.assert_array_equal(results[0], results[1])  # fixed seed → identical runs

    for name in ("ridge", "random_forest", "baseline_national_mean"):
        out, _ = fit_predict_fold(frame, fold, name, model_cfg)
        m = regression_metrics(out[TARGET], out["pred"])
        assert np.isfinite(m["mae"]) and m["n"] == len(out)

    # local explanation works on a fitted LightGBM
    months = frame.df["month"]
    model = make_model("lightgbm", model_cfg, frame.numeric, frame.categorical)
    model.fit(frame.X(fold.train_mask(months)), frame.y(fold.train_mask(months)))
    bundle = {"model": model, "features": frame.features, "categorical": frame.categorical}
    res = explain_row(bundle, frame.df.iloc[[0]], top=3)
    assert len(res["top_contributions"]) == 3 and np.isfinite(res["prediction_pct"])


def test_trailing_baseline_uses_growth_12m():
    X = pd.DataFrame({"growth_12m": [1.0, np.nan, -2.0]})
    np.testing.assert_array_equal(TrailingGrowthBaseline().predict(X), [1.0, 0.0, -2.0])
