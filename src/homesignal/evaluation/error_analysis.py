"""Where does the model fail? Error breakdowns, residual plots and decile calibration.

Uses the out-of-sample backtest predictions of the LightGBM model (all folds) plus the
holdout predictions of the final model, so every number is out-of-sample.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from homesignal.config import ROOT, DataConfig, EvaluationConfig, FeatureConfig, ModelConfig
from homesignal.evaluation.backtest import PRED_DIR
from homesignal.evaluation.metrics import regression_metrics
from homesignal.features.build import TARGET, load_panel
from homesignal.models.train import HOLDOUT_PRED_PATH

log = logging.getLogger(__name__)

FIG_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "results"
PRICE_TIERS = ["bottom 20%", "20-40%", "40-60%", "60-80%", "top 20%"]


def _breakdown(df: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True):
        m = regression_metrics(g[TARGET], g["pred"])
        m["bias"] = float((g["pred"] - g[TARGET]).mean())
        rows.append({by: str(key), **m})
    return pd.DataFrame(rows)


def _attach_context(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    ctx = panel[["zip", "month", "zhvi", "region", "state", "metro_size_bucket"]]
    df = preds.merge(ctx, on=["zip", "month"], how="left")
    df["year"] = df["month"].dt.year
    df["price_tier"] = df.groupby("month")["zhvi"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=PRICE_TIERS)
    )
    df["residual"] = df["pred"] - df[TARGET]
    return df


def calibration_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs. mean actual growth by predicted decile."""
    d = df.dropna(subset=["pred", TARGET]).copy()
    d["decile"] = pd.qcut(d["pred"].rank(method="first"), n_bins, labels=range(1, n_bins + 1))
    out = d.groupby("decile", observed=True).agg(
        pred_mean=("pred", "mean"), actual_mean=(TARGET, "mean"), n=("pred", "size")
    )
    return out.reset_index()


def _plot_residuals(df: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sample = df.sample(min(len(df), 50_000), random_state=0)
    axes[0].scatter(sample["pred"], sample[TARGET], s=2, alpha=0.2)
    lim = [-30, 50]
    axes[0].plot(lim, lim, color="black", lw=1)
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("predicted 12m growth (pp)")
    axes[0].set_ylabel("actual 12m growth (pp)")
    axes[0].set_title("predicted vs actual")
    grp = df.groupby("month")["residual"]
    by_month = pd.DataFrame(
        {"mean": grp.mean(), "p10": grp.quantile(0.1), "p90": grp.quantile(0.9)}
    )
    axes[1].fill_between(
        by_month.index, by_month["p10"], by_month["p90"], alpha=0.25, label="p10–p90"
    )
    axes[1].plot(by_month.index, by_month["mean"], label="mean residual")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("prediction − actual (pp)")
    axes[1].set_title("residuals over time (origin month)")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_calibration(cal: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(cal["pred_mean"], cal["actual_mean"], marker="o")
    lo, hi = (
        float(min(cal["pred_mean"].min(), cal["actual_mean"].min())),
        float(max(cal["pred_mean"].max(), cal["actual_mean"].max())),
    )
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
    ax.set_xlabel("mean predicted growth in decile (pp)")
    ax.set_ylabel("mean actual growth (pp)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_breakdown(
    tbl: pd.DataFrame, by: str, path: Path, baseline: pd.DataFrame | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(tbl)), 4))
    x = np.arange(len(tbl))
    ax.bar(x - 0.2, tbl["mae"], width=0.4, label="LightGBM")
    if baseline is not None:
        ax.bar(x + 0.2, baseline["mae"], width=0.4, label="trailing-12m baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(tbl[by], rotation=45, ha="right")
    ax.set_ylabel("MAE (pp)")
    ax.set_title(f"MAE by {by}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_error_analysis(
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    eval_cfg: EvaluationConfig,
    model_cfg: ModelConfig,
    tag: str = "main",
) -> dict[str, Any]:
    """Write breakdown tables (JSON) and figures for the backtest and holdout predictions."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel(data_cfg.processed_dir)
    out: dict[str, Any] = {}
    sets = {"backtest": PRED_DIR / tag / "lightgbm.parquet", "holdout": HOLDOUT_PRED_PATH}
    base_path = PRED_DIR / tag / "baseline_trailing_12m.parquet"
    for name, path in sets.items():
        if not path.exists():
            log.warning("missing predictions %s", path)
            continue
        df = _attach_context(pd.read_parquet(path), panel)
        baseline = None
        if name == "backtest" and base_path.exists():
            baseline = _attach_context(pd.read_parquet(base_path), panel)
        elif name == "holdout" and "pred_baseline_trailing_12m" in df.columns:
            baseline = df.drop(columns="pred").rename(
                columns={"pred_baseline_trailing_12m": "pred"}
            )
        section: dict[str, Any] = {}
        for by in ["year", "region", "metro_size_bucket", "price_tier"]:
            tbl = _breakdown(df, by)
            section[f"by_{by}"] = tbl.to_dict(orient="records")
            btbl = _breakdown(baseline, by) if baseline is not None else None
            _plot_breakdown(tbl, by, FIG_DIR / f"{name}_mae_by_{by}.png", btbl)
        cal = calibration_table(df)
        section["calibration"] = cal.to_dict(orient="records")
        _plot_calibration(cal, FIG_DIR / f"{name}_calibration.png", f"Decile calibration ({name})")
        _plot_residuals(df, FIG_DIR / f"{name}_residuals.png", f"LightGBM residuals ({name})")
        worst = df.assign(abs_err=df["residual"].abs()).nlargest(10, "abs_err")[
            ["zip", "state", "month", TARGET, "pred"]
        ]
        section["worst_rows"] = worst.assign(month=worst["month"].dt.strftime("%Y-%m")).to_dict(
            orient="records"
        )
        out[name] = section
    (RESULTS_DIR / "error_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    log.info("error analysis written for %s", list(out))
    return out
