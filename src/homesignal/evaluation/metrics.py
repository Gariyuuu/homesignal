"""Regression metrics for 12-month growth predictions (percentage points)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import spearmanr

METRIC_NAMES = ["mae", "rmse", "r2", "spearman", "cs_spearman", "directional_accuracy", "n"]


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """MAE, RMSE, R², Spearman rank correlation, directional accuracy and sample size.

    Rows with a NaN prediction or target are dropped. Directional accuracy is the share of rows
    where ``sign(y_pred) == sign(y_true)`` (zero counts as positive).
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    n = int(len(yt))
    if n == 0:
        return {m: float("nan") for m in METRIC_NAMES} | {"n": 0.0}
    err = yp - yt
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rho = float(spearmanr(yt, yp).statistic) if n > 2 and np.std(yp) > 0 else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": r2,
        "spearman": rho,
        "cs_spearman": float("nan"),
        "directional_accuracy": float(np.mean((yp >= 0) == (yt >= 0))),
        "n": float(n),
    }


def cross_sectional_spearman(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Mean Spearman correlation computed *within* each group (e.g. each origin month).

    This isolates ranking skill across geographies from the (hard to predict) market-wide
    level of growth that dominates the pooled correlation.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    g = np.asarray(groups)
    rhos = []
    for key in np.unique(g):
        m = (g == key) & np.isfinite(yt) & np.isfinite(yp)
        if m.sum() > 2 and np.std(yp[m]) > 0 and np.std(yt[m]) > 0:
            rhos.append(float(spearmanr(yt[m], yp[m]).statistic))
    return float(np.mean(rhos)) if rhos else float("nan")


def regression_metrics_by_origin(
    y_true: ArrayLike, y_pred: ArrayLike, origins: ArrayLike
) -> dict[str, float]:
    """Pooled metrics plus ``cs_spearman`` (mean within-origin Spearman)."""
    m = regression_metrics(y_true, y_pred)
    m["cs_spearman"] = cross_sectional_spearman(y_true, y_pred, origins)
    return m


def summarise_folds(per_fold: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation of each metric across folds."""
    out: dict[str, float] = {}
    for m in METRIC_NAMES:
        vals = np.array([f[m] for f in per_fold], dtype=float)
        out[f"{m}_mean"] = float(np.nanmean(vals))
        out[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return out
