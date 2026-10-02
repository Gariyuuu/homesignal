"""Momentum / target features computed on the wide ZHVI matrix (rows = ZIP, columns = months).

Every feature at column ``t`` uses only columns ``<= t``; the target uses column ``t + horizon``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _log(wide: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.log(wide.to_numpy(dtype=float)), index=wide.index, columns=wide.columns)


def pct_change(wide: pd.DataFrame, months: int) -> pd.DataFrame:
    """Percent change over ``months`` columns, in percentage points (uses columns ``<= t``)."""
    return (wide / wide.shift(months, axis=1) - 1.0) * 100.0


def forward_target(wide: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Percent change from ``t`` to ``t + horizon`` (uses the *future* column; target only)."""
    return (wide.shift(-horizon, axis=1) / wide - 1.0) * 100.0


def volatility(wide: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing standard deviation of monthly log returns (pct points) over ``window`` months."""
    logret = _log(wide).diff(axis=1)
    return logret.T.rolling(window, min_periods=window).std().T * 100.0


def drawdown_from_peak(wide: pd.DataFrame, window: int) -> pd.DataFrame:
    """Distance from the trailing ``window``-month peak, in pct points (<= 0)."""
    peak = wide.T.rolling(window, min_periods=1).max().T
    return (wide / peak - 1.0) * 100.0


def momentum_features(
    wide: pd.DataFrame, horizons: list[int], vol_window: int, peak_window: int
) -> dict[str, pd.DataFrame]:
    """Return ``{feature_name: wide_frame}`` for all momentum features."""
    out: dict[str, pd.DataFrame] = {f"growth_{h}m": pct_change(wide, h) for h in horizons}
    out[f"volatility_{vol_window}m"] = volatility(wide, vol_window)
    out[f"drawdown_{peak_window}m"] = drawdown_from_peak(wide, peak_window)
    out["log_zhvi"] = _log(wide)
    return out


def momentum_feature_names(horizons: list[int], vol_window: int, peak_window: int) -> list[str]:
    """Names produced by :func:`momentum_features`, in order."""
    return [f"growth_{h}m" for h in horizons] + [
        f"volatility_{vol_window}m",
        f"drawdown_{peak_window}m",
        "log_zhvi",
    ]
