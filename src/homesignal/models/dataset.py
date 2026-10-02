"""Turn the processed panel into a modelling frame with a fixed feature list and dtypes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from homesignal.config import DataConfig, EvaluationConfig, FeatureConfig, ModelConfig
from homesignal.data.crosswalks import STATE_TO_DIVISION, STATE_TO_REGION
from homesignal.evaluation.splits import is_origin_month
from homesignal.features.build import TARGET, all_feature_names, load_panel
from homesignal.features.geography import BUCKET_LABELS

log = logging.getLogger(__name__)

# Fixed category levels so train/validation encodings never depend on which rows are present.
CATEGORY_LEVELS: dict[str, list[str]] = {
    "state": sorted(STATE_TO_REGION),
    "region": sorted(set(STATE_TO_REGION.values())),
    "division": sorted(set(STATE_TO_DIVISION.values())),
    "metro_size_bucket": BUCKET_LABELS,
}


@dataclass
class ModelingFrame:
    """Modelling rows plus the feature specification used for them."""

    df: pd.DataFrame
    features: list[str]
    categorical: list[str]

    @property
    def numeric(self) -> list[str]:
        """Numeric feature names."""
        return [f for f in self.features if f not in self.categorical]

    def X(self, mask: pd.Series | None = None) -> pd.DataFrame:  # noqa: N802
        """Feature matrix (optionally masked)."""
        d = self.df if mask is None else self.df[mask]
        return d[self.features]

    def y(self, mask: pd.Series | None = None) -> pd.Series:
        """Target vector (optionally masked)."""
        d = self.df if mask is None else self.df[mask]
        return d[TARGET]


def encode_categoricals(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Cast categorical columns to pandas ``category`` with the fixed level sets."""
    df = df.copy()
    for c in categorical:
        df[c] = pd.Categorical(df[c].astype("object"), categories=CATEGORY_LEVELS[c])
    return df


def select_features(
    feat_cfg: FeatureConfig, model_cfg: ModelConfig, exclude: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Resolve the feature list after exclusions; returns ``(features, categorical)``."""
    drop = set(model_cfg.exclude_features) | set(exclude or [])
    features = [f for f in all_feature_names(feat_cfg) if f not in drop]
    categorical = [c for c in model_cfg.categorical_features if c in features]
    return features, categorical


def load_modeling_frame(
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    eval_cfg: EvaluationConfig,
    model_cfg: ModelConfig,
    exclude: list[str] | None = None,
    require_target: bool = True,
) -> ModelingFrame:
    """Load panel rows at modelling origins (with a known target by default)."""
    panel = load_panel(data_cfg.processed_dir)
    mask = is_origin_month(panel["month"], eval_cfg)
    if require_target:
        mask &= panel[TARGET].notna()
    df = panel[mask].reset_index(drop=True)
    features, categorical = select_features(feat_cfg, model_cfg, exclude)
    df = encode_categoricals(df, categorical)
    log.info(
        "modelling frame: %d rows, %d features (%s → %s)",
        len(df),
        len(features),
        df["month"].min().date(),
        df["month"].max().date(),
    )
    return ModelingFrame(df, features, categorical)
