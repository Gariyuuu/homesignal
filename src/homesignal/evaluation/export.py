"""Export the benchmark dataset: one Parquet with keys, target, features and split labels."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from homesignal.config import DataConfig, EvaluationConfig, FeatureConfig, ModelConfig
from homesignal.evaluation.splits import backtest_folds, is_origin_month, split_label
from homesignal.features.build import TARGET, all_feature_names, load_panel

log = logging.getLogger(__name__)


def export_benchmark(
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    eval_cfg: EvaluationConfig,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    """Write ``data/processed/homesignal_benchmark.parquet`` and a schema JSON next to it."""
    panel = load_panel(data_cfg.processed_dir)
    df = panel[is_origin_month(panel["month"], eval_cfg) & panel[TARGET].notna()].copy()
    df["split"] = split_label(df["month"], eval_cfg)
    for fold in backtest_folds(eval_cfg):
        df[f"fold_{fold.name}"] = pd.Series("", index=df.index, dtype="string")
        df.loc[fold.train_mask(df["month"]), f"fold_{fold.name}"] = "train"
        df.loc[fold.val_mask(df["month"]), f"fold_{fold.name}"] = "val"
    features = all_feature_names(feat_cfg)
    cols = [
        "zip",
        "month",
        "split",
        *[c for c in df.columns if c.startswith("fold_")],
        TARGET,
        "zhvi",
        "acs_vintage",
        *features,
    ]
    df = df[cols].reset_index(drop=True)
    out = data_cfg.processed_dir / "homesignal_benchmark.parquet"
    df.to_parquet(out, index=False)
    schema = {
        "file": out.name,
        "rows": int(len(df)),
        "columns": {c: str(df[c].dtype) for c in df.columns},
        "target": TARGET,
        "features": features,
        "origin_months": eval_cfg.origin_months,
        "first_origin": eval_cfg.first_origin,
        "holdout_start": eval_cfg.holdout_start,
        "gap_months": eval_cfg.gap_months,
        "folds": [
            {
                "name": f.name,
                "train_end": str(f.train_end.date()),
                "val_start": str(f.val_start.date()),
                "val_end": str(f.val_end.date()),
            }
            for f in backtest_folds(eval_cfg)
        ],
        "split_counts": df["split"].value_counts().to_dict(),
    }
    (data_cfg.processed_dir / "homesignal_benchmark.schema.json").write_text(
        json.dumps(schema, indent=2)
    )
    log.info("exported %s (%d rows, %d cols)", out, *df.shape)
    return schema
