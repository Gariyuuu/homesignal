"""Model factory. Every model exposes ``fit(X, y)`` and ``predict(X)`` on pandas frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from homesignal.config import ROOT, ModelConfig
from homesignal.models.baselines import BASELINES

TUNED_PARAMS_PATH = ROOT / "models" / "lightgbm_tuned_params.json"

MODEL_NAMES = [
    "baseline_zero",
    "baseline_trailing_12m",
    "baseline_national_mean",
    "ridge",
    "lasso",
    "random_forest",
    "lightgbm",
    "lightgbm_tuned",
    "mlp",
]


class Model(Protocol):
    """Minimal estimator protocol."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:  # noqa: D102
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D102
        ...


class Winsorizer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Clip each column to percentiles learned on the training data.

    Ratio features (price-to-income, price-to-rent) have extreme tails when a denominator is
    tiny; linear models extrapolate wildly on such rows, trees do not.
    """

    def __init__(self, lower: float = 0.5, upper: float = 99.5) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> Winsorizer:
        """Learn per-column clip bounds."""
        arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(arr, self.lower, axis=0)
        self.hi_ = np.nanpercentile(arr, self.upper, axis=0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Clip to the learned bounds (NaNs pass through)."""
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [
        ("winsor", Winsorizer()),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def load_tuned_params(path: Path = TUNED_PARAMS_PATH) -> dict[str, Any] | None:
    """Best LightGBM parameters from ``homesignal tune``, if present."""
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data["params"]


def make_model(name: str, cfg: ModelConfig, numeric: list[str], categorical: list[str]) -> Model:
    """Instantiate a model by name with seeds fixed from the config."""
    seed = cfg.seed
    if name.startswith("baseline_"):
        baseline: Model = BASELINES[name.removeprefix("baseline_")]()
        return baseline
    if name == "ridge":
        return Pipeline(
            [
                ("prep", _preprocessor(numeric, categorical, True)),
                ("model", Ridge(alpha=cfg.linear.ridge_alpha)),
            ]
        )
    if name == "lasso":
        return Pipeline(
            [
                ("prep", _preprocessor(numeric, categorical, True)),
                ("model", Lasso(alpha=cfg.linear.lasso_alpha, max_iter=5000, random_state=seed)),
            ]
        )
    if name == "random_forest":
        rf = RandomForestRegressor(random_state=seed, n_jobs=-1, **cfg.random_forest)
        return Pipeline([("prep", _preprocessor(numeric, categorical, False)), ("model", rf)])
    if name == "mlp":
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=seed
        )
        return Pipeline([("prep", _preprocessor(numeric, categorical, True)), ("model", mlp)])
    if name in ("lightgbm", "lightgbm_tuned"):
        params = dict(cfg.lightgbm)
        if name == "lightgbm_tuned":
            tuned = load_tuned_params()
            if tuned is None:
                raise RuntimeError("no tuned parameters found; run `homesignal tune` first")
            params.update(tuned)
        return lgb.LGBMRegressor(
            random_state=seed, n_jobs=-1, verbose=-1, deterministic=True, **params
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
