"""Time-based splits: expanding-window backtest folds with a gap, plus a final holdout.

Origins are month-start timestamps. A row with origin ``t`` has a target observed at
``t + horizon``. For a validation block starting at ``v0``, training origins must satisfy
``t + gap <= v0`` so that no training target window overlaps the validation period.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from homesignal.config import EvaluationConfig


@dataclass(frozen=True)
class Fold:
    """One backtest fold."""

    name: str
    train_end: pd.Timestamp  # inclusive last training origin
    val_start: pd.Timestamp  # inclusive
    val_end: pd.Timestamp  # inclusive

    def train_mask(self, months: pd.Series) -> pd.Series:
        """Boolean mask of training rows."""
        return months <= self.train_end

    def val_mask(self, months: pd.Series) -> pd.Series:
        """Boolean mask of validation rows."""
        return (months >= self.val_start) & (months <= self.val_end)


def _last_origin_in_year(year: int, origin_months: list[int]) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=max(origin_months), day=1)


def _first_origin_in_year(year: int, origin_months: list[int]) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=min(origin_months), day=1)


def backtest_folds(cfg: EvaluationConfig) -> list[Fold]:
    """Expanding-window folds, one per validation year, all strictly before the holdout."""
    holdout = pd.Timestamp(cfg.holdout_start)
    folds = []
    for year in cfg.backtest_validation_years:
        v0 = _first_origin_in_year(year, cfg.origin_months)
        v1 = _last_origin_in_year(year, cfg.origin_months)
        if v1 >= holdout:
            raise ValueError(
                f"validation year {year} overlaps the holdout starting {holdout.date()}"
            )
        folds.append(Fold(f"val_{year}", v0 - pd.DateOffset(months=cfg.gap_months), v0, v1))
    return folds


def holdout_fold(cfg: EvaluationConfig, last_origin: pd.Timestamp) -> Fold:
    """Return the final fold: train before the holdout (minus the gap), evaluate on the holdout."""
    h0 = pd.Timestamp(cfg.holdout_start)
    return Fold("holdout", h0 - pd.DateOffset(months=cfg.gap_months), h0, last_origin)


def is_origin_month(months: pd.Series, cfg: EvaluationConfig) -> pd.Series:
    """Return a mask of rows whose month is a modelling origin on/after ``first_origin``."""
    return months.dt.month.isin(cfg.origin_months) & (months >= pd.Timestamp(cfg.first_origin))


def split_label(months: pd.Series, cfg: EvaluationConfig) -> pd.Series:
    """Label each origin as ``backtest`` (before the holdout) or ``holdout``."""
    return pd.Series(
        ["holdout" if m >= pd.Timestamp(cfg.holdout_start) else "backtest" for m in months],
        index=months.index,
        dtype="string",
    )
