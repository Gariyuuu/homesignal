"""Expanding-window backtest across model families, logging per-fold metrics and predictions."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from homesignal.config import ROOT, DataConfig, EvaluationConfig, FeatureConfig, ModelConfig
from homesignal.evaluation.metrics import regression_metrics_by_origin, summarise_folds
from homesignal.evaluation.splits import Fold, backtest_folds
from homesignal.models.dataset import ModelingFrame, load_modeling_frame
from homesignal.models.registry import make_model
from homesignal.tracking import log_run

log = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "baseline_zero",
    "baseline_trailing_12m",
    "baseline_national_mean",
    "ridge",
    "lasso",
    "random_forest",
    "lightgbm",
]
PRED_DIR = ROOT / "models" / "backtest_predictions"
RESULTS_DIR = ROOT / "reports" / "results"


def fit_predict_fold(
    frame: ModelingFrame, fold: Fold, model_name: str, cfg: ModelConfig
) -> tuple[pd.DataFrame, float]:
    """Fit one model on a fold's training rows and predict its validation rows."""
    months = frame.df["month"]
    tr, va = fold.train_mask(months), fold.val_mask(months)
    model = make_model(model_name, cfg, frame.numeric, frame.categorical)
    t0 = time.time()
    model.fit(frame.X(tr), frame.y(tr))
    pred = np.asarray(model.predict(frame.X(va)), dtype=float)
    elapsed = time.time() - t0
    out = frame.df.loc[va, ["zip", "month", "target_growth_12m"]].copy()
    out["pred"] = pred
    out["fold"] = fold.name
    return out, elapsed


def run_backtest(
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    eval_cfg: EvaluationConfig,
    model_cfg: ModelConfig,
    models: list[str] | None = None,
    tag: str = "main",
    exclude_features: list[str] | None = None,
) -> dict[str, Any]:
    """Run every model over every backtest fold; write results JSON and prediction parquet files."""
    models = models or DEFAULT_MODELS
    frame = load_modeling_frame(data_cfg, feat_cfg, eval_cfg, model_cfg, exclude=exclude_features)
    folds = backtest_folds(eval_cfg)
    results: dict[str, Any] = {
        "tag": tag,
        "features": frame.features,
        "excluded": exclude_features or [],
        "models": {},
    }
    pred_dir = PRED_DIR / tag
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name in models:
        per_fold: list[dict[str, float]] = []
        preds = []
        for fold in folds:
            out, elapsed = fit_predict_fold(frame, fold, name, model_cfg)
            m = regression_metrics_by_origin(out["target_growth_12m"], out["pred"], out["month"])
            m["fold"] = fold.name  # type: ignore[assignment]
            m["train_rows"] = float(fold.train_mask(frame.df["month"]).sum())
            m["fit_seconds"] = elapsed
            per_fold.append(m)
            preds.append(out)
            log.info(
                "%-24s %s MAE=%.3f RMSE=%.3f R2=%.3f rho=%.3f cs_rho=%.3f dir=%.3f (%.0fs)",
                name,
                fold.name,
                m["mae"],
                m["rmse"],
                m["r2"],
                m["spearman"],
                m["cs_spearman"],
                m["directional_accuracy"],
                elapsed,
            )
        summary = summarise_folds(per_fold)
        results["models"][name] = {"per_fold": per_fold, "summary": summary}
        pd.concat(preds).to_parquet(pred_dir / f"{name}.parquet", index=False)
        log_run(
            {
                "kind": "backtest",
                "tag": tag,
                "model": name,
                "n_features": len(frame.features),
                "excluded": exclude_features or [],
                "summary": summary,
                "per_fold": per_fold,
            }
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"backtest_{tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out_path)
    return results


def load_results(tag: str = "main") -> dict[str, Any] | None:
    """Load a saved backtest results file."""
    p: Path = RESULTS_DIR / f"backtest_{tag}.json"
    if not p.exists():
        return None
    data: dict[str, Any] = json.loads(p.read_text())
    return data
