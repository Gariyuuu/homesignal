"""Typed configuration loaded from the YAML files in ``configs/``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


class ZillowConfig(BaseModel):
    """Zillow Research download URLs."""

    zhvi_zip_url: str
    zori_zip_url: str


class ACSConfig(BaseModel):
    """Census ACS 5-year summary-file settings."""

    base_url: str
    legacy_vintages: list[int]
    table_based_vintages: list[int]
    tables: list[str]
    default_availability_lag_months: int = 13
    availability_overrides: dict[int, str] = Field(default_factory=dict)

    @property
    def vintages(self) -> list[int]:
        """All vintages, oldest first."""
        return sorted(self.legacy_vintages + self.table_based_vintages)


class BLSConfig(BaseModel):
    """BLS public API settings."""

    api_url: str
    national_series: list[str]
    state_unemployment: bool = True
    start_year: int = 2000


class FreddieConfig(BaseModel):
    """Freddie Mac PMMS settings."""

    pmms_url: str


class PanelConfig(BaseModel):
    """Panel assembly settings."""

    start_month: str = "2000-01"


class DataConfig(BaseModel):
    """``configs/data.yaml``."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    user_agent: str
    zillow: ZillowConfig
    acs: ACSConfig
    bls: BLSConfig
    freddie: FreddieConfig
    panel: PanelConfig

    def resolve(self, root: Path = ROOT) -> DataConfig:
        """Return a copy with directories made absolute relative to ``root``."""
        return self.model_copy(
            update={
                "raw_dir": root / self.raw_dir,
                "interim_dir": root / self.interim_dir,
                "processed_dir": root / self.processed_dir,
            }
        )


class PublicationLags(BaseModel):
    """Months by which each macro series is lagged before use at origin ``t``."""

    cpi: int = 1
    national_unemployment: int = 1
    state_unemployment: int = 2
    mortgage_rate: int = 0


class FeatureConfig(BaseModel):
    """``configs/features.yaml``."""

    horizon_months: int = 12
    momentum_horizons: list[int]
    volatility_window: int = 12
    peak_window: int = 36
    publication_lags: PublicationLags
    metro_size_buckets: list[int]


class EvaluationConfig(BaseModel):
    """``configs/evaluation.yaml``."""

    origin_months: list[int]
    first_origin: str
    backtest_validation_years: list[int]
    gap_months: int = 12
    holdout_start: str


class LinearConfig(BaseModel):
    """Linear model hyper-parameters."""

    ridge_alpha: float = 10.0
    lasso_alpha: float = 0.01


class TuningConfig(BaseModel):
    """Optuna budget."""

    n_trials: int = 25
    folds: list[int]


class ModelConfig(BaseModel):
    """``configs/model.yaml``."""

    seed: int = 42
    target: str = "target_growth_12m"
    exclude_features: list[str] = Field(default_factory=list)
    categorical_features: list[str]
    baselines: list[str]
    linear: LinearConfig
    random_forest: dict[str, Any]
    lightgbm: dict[str, Any]
    tuning: TuningConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_data_config(path: Path | None = None) -> DataConfig:
    """Load and resolve the data config."""
    return DataConfig(**_load_yaml(path or CONFIG_DIR / "data.yaml")).resolve()


def load_feature_config(path: Path | None = None) -> FeatureConfig:
    """Load the feature config."""
    return FeatureConfig(**_load_yaml(path or CONFIG_DIR / "features.yaml"))


def load_evaluation_config(path: Path | None = None) -> EvaluationConfig:
    """Load the evaluation config."""
    return EvaluationConfig(**_load_yaml(path or CONFIG_DIR / "evaluation.yaml"))


def load_model_config(path: Path | None = None) -> ModelConfig:
    """Load the model config."""
    return ModelConfig(**_load_yaml(path or CONFIG_DIR / "model.yaml"))
