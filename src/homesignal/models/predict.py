"""Predict for one ZIP at one origin month, with a SHAP breakdown of the top features."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import shap

from homesignal.config import (
    load_data_config,
    load_evaluation_config,
    load_feature_config,
    load_model_config,
)
from homesignal.features.build import TARGET, load_panel
from homesignal.models.dataset import encode_categoricals
from homesignal.models.train import load_final_model, load_metadata


def explain_row(bundle: dict[str, Any], row: pd.DataFrame, top: int = 8) -> dict[str, Any]:
    """Prediction plus the ``top`` features by absolute SHAP contribution for one row."""
    model = bundle["model"]
    features: list[str] = bundle["features"]
    X = encode_categoricals(row, bundle["categorical"])[features]
    pred = float(model.predict(X)[0])
    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X))[0]
    base = float(np.asarray(explainer.expected_value).ravel()[0])
    order = np.argsort(-np.abs(shap_values))[:top]
    contributions = [
        {
            "feature": features[i],
            "value": None
            if pd.isna(X.iloc[0, i])
            else (X.iloc[0, i] if isinstance(X.iloc[0, i], str) else float(X.iloc[0, i])),
            "shap": float(shap_values[i]),
        }
        for i in order
    ]
    return {"prediction_pct": pred, "base_value_pct": base, "top_contributions": contributions}


def predict_one(zip_code: str, month: str, top: int = 8) -> dict[str, Any]:
    """Look up the panel row for ``zip_code`` at ``month`` and explain the prediction."""
    data_cfg = load_data_config()
    panel = load_panel(data_cfg.processed_dir)
    origin = pd.Timestamp(month).to_period("M").to_timestamp()
    row = panel[(panel["zip"] == zip_code.zfill(5)) & (panel["month"] == origin)]
    if row.empty:
        raise SystemExit(f"no panel row for ZIP {zip_code} at {origin.date()}")
    bundle = load_final_model()
    meta = load_metadata()
    result = explain_row(bundle, row, top=top)
    actual = row[TARGET].iloc[0]
    result.update(
        {
            "zip": zip_code.zfill(5),
            "origin_month": str(origin.date()),
            "zhvi_at_origin": float(row["zhvi"].iloc[0]),
            "actual_growth_pct": None if pd.isna(actual) else float(actual),
            "in_training_window": str(origin.date()) <= meta["training_origins"][1],
            "disclaimer": (
                "Educational project. Not financial, lending, appraisal or investment advice."
            ),
        }
    )
    return result


def predict_cli(zip_code: str, month: str, top: int = 8) -> None:
    """Print a prediction as JSON."""
    load_feature_config(), load_evaluation_config(), load_model_config()  # validate configs
    print(json.dumps(predict_one(zip_code, month, top=top), indent=2))
