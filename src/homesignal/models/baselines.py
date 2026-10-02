"""Naive baselines every model must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ZeroBaseline:
    """Predict zero growth."""

    name = "baseline_zero"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ZeroBaseline:
        """No-op."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Zeros."""
        return np.zeros(len(X))


class TrailingGrowthBaseline:
    """Predict that the trailing 12-month growth continues."""

    name = "baseline_trailing_12m"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TrailingGrowthBaseline:
        """No-op."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Trailing growth (NaN → 0)."""
        return X["growth_12m"].fillna(0.0).to_numpy(dtype=float)


class NationalMeanBaseline:
    """Predict the mean training target (the historical national average growth)."""

    name = "baseline_national_mean"

    def __init__(self) -> None:
        self.mean_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NationalMeanBaseline:
        """Store the training mean."""
        self.mean_ = float(np.nanmean(y.to_numpy(dtype=float)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Constant."""
        return np.full(len(X), self.mean_)


BASELINES: dict[
    str, type[ZeroBaseline] | type[TrailingGrowthBaseline] | type[NationalMeanBaseline]
] = {
    "zero": ZeroBaseline,
    "trailing_12m": TrailingGrowthBaseline,
    "national_mean": NationalMeanBaseline,
}
