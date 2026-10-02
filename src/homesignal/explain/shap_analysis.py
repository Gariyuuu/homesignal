"""SHAP global importance, dependence plots and local explanations for the final model."""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from homesignal.config import ROOT, DataConfig, EvaluationConfig, FeatureConfig, ModelConfig
from homesignal.models.dataset import load_modeling_frame
from homesignal.models.predict import explain_row
from homesignal.models.train import load_final_model, load_metadata

log = logging.getLogger(__name__)

FIG_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "results"
EXAMPLE_ZIPS = ["94110", "78702", "33139", "48226", "10025", "85004"]


def run_shap(
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    eval_cfg: EvaluationConfig,
    model_cfg: ModelConfig,
    sample_size: int = 20_000,
    top_dependence: int = 6,
) -> dict[str, Any]:
    """Compute SHAP values on a seeded sample of pre-holdout rows; write figures and JSON."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_final_model()
    meta = load_metadata()
    frame = load_modeling_frame(data_cfg, feat_cfg, eval_cfg, model_cfg)
    train_end = pd.Timestamp(meta["training_origins"][1])
    pool = frame.df[frame.df["month"] <= train_end]
    sample = pool.sample(min(sample_size, len(pool)), random_state=model_cfg.seed)
    X = sample[bundle["features"]]
    explainer = shap.TreeExplainer(bundle["model"])
    values = np.asarray(explainer.shap_values(X))
    importance = pd.Series(np.abs(values).mean(axis=0), index=bundle["features"]).sort_values(
        ascending=False
    )

    shap.summary_plot(values, X, show=False, max_display=25)
    plt.title("SHAP summary (mean |SHAP| ranks features; colour = feature value)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 7))
    importance.head(25)[::-1].plot.barh(ax=ax)
    ax.set_xlabel("mean |SHAP| (pp of 12m growth)")
    ax.set_title("Global feature importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_importance.png", dpi=130)
    plt.close(fig)

    numeric_top = [f for f in importance.index if f not in bundle["categorical"]][:top_dependence]
    for f in numeric_top:
        shap.dependence_plot(f, values, X, interaction_index=None, show=False)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_dependence_{f}.png", dpi=120, bbox_inches="tight")
        plt.close()

    # Local explanations at the last origin inside the training window and at the latest origin.
    latest = frame.df["month"].max()
    local = []
    for z in EXAMPLE_ZIPS:
        for origin in (train_end, latest):
            row = frame.df[(frame.df["zip"] == z) & (frame.df["month"] == origin)]
            if row.empty:
                continue
            res = explain_row(bundle, row, top=6)
            res.update(
                {
                    "zip": z,
                    "origin": str(origin.date()),
                    "actual": float(row["target_growth_12m"].iloc[0]),
                }
            )
            local.append(res)

    result = {
        "sample_rows": int(len(sample)),
        "sample_origins": [str(sample["month"].min().date()), str(sample["month"].max().date())],
        "importance": importance.round(4).to_dict(),
        "dependence_features": numeric_top,
        "local_examples": local,
    }
    (RESULTS_DIR / "shap.json").write_text(json.dumps(result, indent=2))
    log.info("SHAP top features: %s", list(importance.head(8).round(3).items()))
    return result
