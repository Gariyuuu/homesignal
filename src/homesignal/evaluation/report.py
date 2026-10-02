"""Render ``reports/evaluation.md`` from saved results so every number in the docs is computed."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from homesignal.config import ROOT, DataConfig, EvaluationConfig, FeatureConfig, ModelConfig
from homesignal.evaluation.backtest import load_results
from homesignal.evaluation.splits import backtest_folds
from homesignal.models.train import METADATA_PATH

log = logging.getLogger(__name__)

REPORT_PATH = ROOT / "reports" / "evaluation.md"
RESULTS_DIR = ROOT / "reports" / "results"
MODEL_LABELS = {
    "baseline_zero": "Zero growth",
    "baseline_trailing_12m": "Trailing 12m continues",
    "baseline_national_mean": "Training mean (national avg)",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM (default)",
    "lightgbm_tuned": "LightGBM (tuned)",
    "mlp": "MLP",
}


def _f(x: float | None, nd: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:.{nd}f}"


def _summary_table(results: dict[str, Any]) -> str:
    lines = [
        "| Model | MAE (pp) | RMSE (pp) | R² | ρ pooled | ρ within origin | Dir. acc. |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, res in results["models"].items():
        s = res["summary"]
        lines.append(
            f"| {MODEL_LABELS.get(name, name)} | {_f(s['mae_mean'])} ± {_f(s['mae_std'])} | {_f(s['rmse_mean'])} ± {_f(s['rmse_std'])} "
            f"| {_f(s['r2_mean'], 3)} ± {_f(s['r2_std'], 3)} | {_f(s['spearman_mean'], 3)} ± {_f(s['spearman_std'], 3)} "
            f"| {_f(s['cs_spearman_mean'], 3)} ± {_f(s['cs_spearman_std'], 3)} "
            f"| {_f(s['directional_accuracy_mean'], 3)} ± {_f(s['directional_accuracy_std'], 3)} |"
        )
    return "\n".join(lines)


def _per_fold_table(results: dict[str, Any], metric: str, folds: list[str]) -> str:
    names = list(results["models"])
    header = "| Fold | " + " | ".join(MODEL_LABELS.get(n, n) for n in names) + " |"
    lines = [header, "|---|" + "---|" * len(names)]
    for fold in folds:
        row = [fold]
        for n in names:
            pf = next((f for f in results["models"][n]["per_fold"] if f["fold"] == fold), None)
            row.append(_f(pf[metric], 2 if metric in ("mae", "rmse") else 3) if pf else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _holdout_section(meta: dict[str, Any]) -> str:
    hm = meta["holdout_metrics"]
    lines = [
        f"Final model: **{meta['model']}** trained on origins {meta['training_origins'][0]} → {meta['training_origins'][1]} "
        f"({meta['training_rows']:,} rows), evaluated **once** on holdout origins {meta['holdout_origins'][0]} → "
        f"{meta['holdout_origins'][1]} ({meta['holdout_rows']:,} rows).",
        "",
        "| Model | MAE (pp) | RMSE (pp) | R² | ρ pooled | ρ within origin | Dir. acc. |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in hm.items():
        label = "**Final LightGBM**" if name == "final_model" else MODEL_LABELS.get(name, name)
        lines.append(
            f"| {label} | {_f(m['mae'])} | {_f(m['rmse'])} | {_f(m['r2'], 3)} | {_f(m['spearman'], 3)} | {_f(m['cs_spearman'], 3)} | {_f(m['directional_accuracy'], 3)} |"
        )
    lines += [
        "",
        "By holdout origin year (final model):",
        "",
        "| Year | MAE | RMSE | R² | ρ pooled | ρ within origin | Dir. acc. | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for y, m in meta["holdout_metrics_by_year"].items():
        lines.append(
            f"| {y} | {_f(m['mae'])} | {_f(m['rmse'])} | {_f(m['r2'], 3)} | {_f(m['spearman'], 3)} | {_f(m['cs_spearman'], 3)} | {_f(m['directional_accuracy'], 3)} | {int(m['n']):,} |"
        )
    return "\n".join(lines)


def _breakdown_table(rows: list[dict[str, Any]], key: str) -> str:
    lines = [
        f"| {key} | MAE | RMSE | R² | Spearman ρ | bias (pred−actual) | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r[key]} | {_f(r['mae'])} | {_f(r['rmse'])} | {_f(r['r2'], 3)} | {_f(r['spearman'], 3)} | {_f(r['bias'])} | {int(r['n']):,} |"
        )
    return "\n".join(lines)


def _calibration_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Predicted decile | mean predicted | mean actual | n |", "|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['decile']} | {_f(r['pred_mean'])} | {_f(r['actual_mean'])} | {int(r['n']):,} |"
        )
    return "\n".join(lines)


def _ablation_section(main: dict[str, Any], tags: dict[str, str]) -> str:
    lines = [
        "| Variant | MAE (pp) | RMSE (pp) | ρ pooled | ρ within origin | Dir. acc. |",
        "|---|---|---|---|---|---|",
    ]

    def row(label: str, s: dict[str, float]) -> str:
        return (
            f"| {label} | {_f(s['mae_mean'])} ± {_f(s['mae_std'])} | {_f(s['rmse_mean'])} "
            f"| {_f(s['spearman_mean'], 3)} | {_f(s['cs_spearman_mean'], 3)} ± {_f(s['cs_spearman_std'], 3)} "
            f"| {_f(s['directional_accuracy_mean'], 3)} |"
        )

    lines.append(row("All features", main["models"]["lightgbm"]["summary"]))
    for tag, label in tags.items():
        r = load_results(tag)
        if r and "lightgbm" in r["models"]:
            lines.append(row(label, r["models"]["lightgbm"]["summary"]))
    return "\n".join(lines)


def write_reports(
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    eval_cfg: EvaluationConfig,
    model_cfg: ModelConfig,
) -> Path:
    """Assemble the evaluation report from results JSON files and figures."""
    results = load_results("main")
    if results is None:
        raise RuntimeError("run `homesignal backtest` first")
    folds = [f.name for f in backtest_folds(eval_cfg)]
    meta = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else None
    err_path = RESULTS_DIR / "error_analysis.json"
    err = json.loads(err_path.read_text()) if err_path.exists() else {}
    shap_path = RESULTS_DIR / "shap.json"
    shp = json.loads(shap_path.read_text()) if shap_path.exists() else None
    fold_defs = backtest_folds(eval_cfg)

    parts = [
        "# Evaluation report",
        "",
        "All numbers below are produced by `make all` (`homesignal backtest`, `train-final`, `explain`, `report`) and are",
        "out-of-sample. Nothing is hand-edited. Target: 12-month-forward percent change in ZIP-level ZHVI (percentage points).",
        "",
        "## Evaluation design",
        "",
        f"- Prediction origins: month-ends in months {eval_cfg.origin_months} from {eval_cfg.first_origin} (quarterly, to keep the laptop budget).",
        f"- Expanding-window backtest, one fold per validation year, {eval_cfg.gap_months}-month gap between the last training origin and the first validation origin so no training target overlaps validation.",
        f"- Final holdout: origins from {eval_cfg.holdout_start} onward, touched exactly once by `train-final`.",
        "",
        "| Fold | training origins | validation origins |",
        "|---|---|---|",
        *[
            f"| {f.name} | ≤ {f.train_end.date()} | {f.val_start.date()} → {f.val_end.date()} |"
            for f in fold_defs
        ],
        "",
        "## Backtest: mean ± std across folds",
        "",
        f"Feature set: {len(results['features'])} features (see `docs/feature_dictionary.md`).",
        "",
        _summary_table(results),
        "",
        "## Backtest: per-fold MAE (pp)",
        "",
        _per_fold_table(results, "mae", folds),
        "",
        "## Backtest: per-fold Spearman ρ (pooled across ZIPs and origins)",
        "",
        _per_fold_table(results, "spearman", folds),
        "",
        "## Backtest: per-fold Spearman ρ (within origin month, averaged)",
        "",
        _per_fold_table(results, "cs_spearman", folds),
        "",
        "## Backtest: per-fold R²",
        "",
        _per_fold_table(results, "r2", folds),
        "",
        "## Geography ablation (LightGBM, backtest mean ± std)",
        "",
        _ablation_section(
            results,
            {
                "no_geo": "Without state/region/division/metro bucket",
                "no_acs": "Without ACS demographics",
            },
        ),
        "",
    ]
    if meta:
        parts += ["## Final holdout (touched once)", "", _holdout_section(meta), ""]
    if err:
        for name, label in (
            ("backtest", "Backtest (LightGBM, all folds)"),
            ("holdout", "Holdout (final model)"),
        ):
            if name not in err:
                continue
            e = err[name]
            parts += [f"## Error analysis — {label}", ""]
            for key, title in (
                ("year", "By origin year"),
                ("region", "By Census region"),
                ("metro_size_bucket", "By metro size"),
                ("price_tier", "By price tier (ZHVI quintile within month)"),
            ):
                parts += [
                    f"### {title}",
                    "",
                    _breakdown_table(e[f"by_{key}"], key),
                    "",
                    f"![MAE by {key}](figures/{name}_mae_by_{key}.png)",
                    "",
                ]
            parts += [
                "### Decile calibration",
                "",
                _calibration_table(e["calibration"]),
                "",
                f"![calibration](figures/{name}_calibration.png)",
                "",
                f"![residuals](figures/{name}_residuals.png)",
                "",
            ]
    if shp:
        top = list(shp["importance"].items())[:15]
        parts += [
            "## Explainability (SHAP, final model)",
            "",
            f"Computed on a seeded sample of {shp['sample_rows']:,} training rows ({shp['sample_origins'][0]} → {shp['sample_origins'][1]}).",
            "",
            "| Feature | mean \\|SHAP\\| (pp) |",
            "|---|---|",
            *[f"| {k} | {_f(v, 3)} |" for k, v in top],
            "",
            "![SHAP summary](figures/shap_summary.png)",
            "",
            *[
                f"![dependence {f}](figures/shap_dependence_{f}.png)"
                for f in shp["dependence_features"]
            ],
            "",
            "### Local explanations",
            "",
        ]
        for ex in shp["local_examples"]:
            contribs = ", ".join(
                f"{c['feature']} ({c['shap']:+.2f})" for c in ex["top_contributions"][:4]
            )
            actual = (
                "n/a (not yet observed)" if ex["actual"] != ex["actual"] else f"{ex['actual']:.2f}"
            )
            parts.append(
                f"- ZIP {ex['zip']} @ {ex['origin']}: predicted {ex['prediction_pct']:.2f} pp, actual {actual}. Top: {contribs}"
            )
        parts.append("")
    parts += [
        "## Discussion",
        "",
        "See the *Honest discussion* section appended by the author below (kept separate from generated tables).",
        "",
    ]
    discussion = ROOT / "reports" / "evaluation_discussion.md"
    if discussion.exists():
        parts.append(discussion.read_text())
    REPORT_PATH.write_text("\n".join(parts))
    log.info("wrote %s", REPORT_PATH)
    return REPORT_PATH
