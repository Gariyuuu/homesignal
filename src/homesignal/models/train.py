"""Final model: train on all pre-holdout origins, evaluate once on the holdout, save artifacts."""

from __future__ import annotations

import json
import logging
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from homesignal.config import ROOT, DataConfig, EvaluationConfig, FeatureConfig, ModelConfig
from homesignal.data.download import read_meta
from homesignal.evaluation.metrics import regression_metrics_by_origin
from homesignal.evaluation.splits import holdout_fold
from homesignal.features.build import TARGET
from homesignal.models.dataset import load_modeling_frame
from homesignal.models.registry import TUNED_PARAMS_PATH, load_tuned_params, make_model
from homesignal.tracking import log_run

log = logging.getLogger(__name__)

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "homesignal_lightgbm.joblib"
METADATA_PATH = MODEL_DIR / "homesignal_lightgbm.metadata.json"
HOLDOUT_PRED_PATH = MODEL_DIR / "holdout_predictions.parquet"
HOLDOUT_BASELINES = ["baseline_zero", "baseline_trailing_12m", "baseline_national_mean"]


def _source_versions(data_cfg: DataConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, p in {
        "zillow_zhvi": data_cfg.raw_dir / "zillow" / "zhvi_zip.csv",
        "zillow_zori": data_cfg.raw_dir / "zillow" / "zori_zip.csv",
        "freddie_pmms": data_cfg.raw_dir / "freddie" / "PMMS_history.csv",
    }.items():
        meta = read_meta(p)
        if meta:
            out[name] = {"downloaded_at": meta["downloaded_at"], "sha256": meta["sha256"]}
    out["acs_vintages"] = data_cfg.acs.vintages
    return out


def train_final(
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    eval_cfg: EvaluationConfig,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    """Fit the final LightGBM (tuned params if available) and score the untouched holdout once."""
    frame = load_modeling_frame(data_cfg, feat_cfg, eval_cfg, model_cfg)
    months = frame.df["month"]
    fold = holdout_fold(eval_cfg, months.max())
    tr, ho = fold.train_mask(months), fold.val_mask(months)
    model_name = "lightgbm_tuned" if load_tuned_params() else "lightgbm"
    model = make_model(model_name, model_cfg, frame.numeric, frame.categorical)
    model.fit(frame.X(tr), frame.y(tr))
    holdout = frame.df.loc[ho, ["zip", "month", TARGET]].copy()
    holdout["pred"] = np.asarray(model.predict(frame.X(ho)), dtype=float)
    metrics = {
        "final_model": regression_metrics_by_origin(
            holdout[TARGET], holdout["pred"], holdout["month"]
        )
    }
    for b in HOLDOUT_BASELINES:
        bm = make_model(b, model_cfg, frame.numeric, frame.categorical).fit(
            frame.X(tr), frame.y(tr)
        )
        bp = np.asarray(bm.predict(frame.X(ho)), dtype=float)
        holdout[f"pred_{b}"] = bp
        metrics[b] = regression_metrics_by_origin(holdout[TARGET], bp, holdout["month"])
    per_year = {
        str(y): regression_metrics_by_origin(g[TARGET], g["pred"], g["month"])
        for y, g in holdout.groupby(holdout["month"].dt.year)
    }
    log.info(
        "holdout %s→%s: %s",
        fold.val_start.date(),
        fold.val_end.date(),
        {k: round(v, 3) for k, v in metrics["final_model"].items()},
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": frame.features, "categorical": frame.categorical}, MODEL_PATH
    )
    holdout.to_parquet(HOLDOUT_PRED_PATH, index=False)
    metadata = {
        "model": model_name,
        "params": model.get_params() if hasattr(model, "get_params") else {},
        "tuned_params_file": str(TUNED_PARAMS_PATH.relative_to(ROOT))
        if load_tuned_params()
        else None,
        "target": TARGET,
        "horizon_months": feat_cfg.horizon_months,
        "features": frame.features,
        "categorical_features": frame.categorical,
        "training_origins": [str(months[tr].min().date()), str(months[tr].max().date())],
        "training_rows": int(tr.sum()),
        "holdout_origins": [str(fold.val_start.date()), str(fold.val_end.date())],
        "holdout_rows": int(ho.sum()),
        "holdout_metrics": metrics,
        "holdout_metrics_by_year": per_year,
        "seed": model_cfg.seed,
        "versions": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("lightgbm", "scikit-learn", "pandas", "numpy", "shap")},
        },
        "data_sources": _source_versions(data_cfg),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    log_run(
        {
            "kind": "holdout",
            "model": model_name,
            "metrics": metrics,
            "by_year": per_year,
            "training_rows": int(tr.sum()),
        }
    )
    return metadata


def load_final_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the saved final model bundle."""
    data: dict[str, Any] = joblib.load(path)
    return data


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    """Load the final model metadata."""
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def predictions_frame(path: Path = HOLDOUT_PRED_PATH) -> pd.DataFrame:
    """Load the saved holdout predictions."""
    return pd.read_parquet(path)
