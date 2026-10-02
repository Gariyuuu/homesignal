"""Modest Optuna search for LightGBM using the most recent backtest folds (time-based CV)."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import optuna

from homesignal.config import DataConfig, EvaluationConfig, FeatureConfig, ModelConfig
from homesignal.evaluation.backtest import fit_predict_fold
from homesignal.evaluation.metrics import regression_metrics
from homesignal.evaluation.splits import backtest_folds
from homesignal.models.dataset import load_modeling_frame
from homesignal.models.registry import TUNED_PARAMS_PATH
from homesignal.tracking import log_run

log = logging.getLogger(__name__)


def tune(
    data_cfg: DataConfig,
    feat_cfg: FeatureConfig,
    eval_cfg: EvaluationConfig,
    model_cfg: ModelConfig,
) -> dict[str, Any]:
    """Minimise mean validation MAE over the configured folds; write the best params to disk."""
    frame = load_modeling_frame(data_cfg, feat_cfg, eval_cfg, model_cfg)
    folds = [
        f for f in backtest_folds(eval_cfg) if int(f.name.split("_")[1]) in model_cfg.tuning.folds
    ]
    base = dict(model_cfg.lightgbm)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        trial_cfg = model_cfg.model_copy(update={"lightgbm": {**base, **params}})
        maes = []
        for fold in folds:
            out, _ = fit_predict_fold(frame, fold, "lightgbm", trial_cfg)
            maes.append(regression_metrics(out["target_growth_12m"], out["pred"])["mae"])
        score = float(np.mean(maes))
        log.info("trial %d: MAE=%.4f %s", trial.number, score, params)
        return score

    sampler = optuna.samplers.TPESampler(seed=model_cfg.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=model_cfg.tuning.n_trials)
    result = {
        "params": study.best_params,
        "best_mean_mae": study.best_value,
        "folds": [f.name for f in folds],
        "n_trials": model_cfg.tuning.n_trials,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params} for t in study.trials
        ],
    }
    TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PARAMS_PATH.write_text(json.dumps(result, indent=2))
    log_run(
        {
            "kind": "tune",
            "model": "lightgbm",
            "best_params": study.best_params,
            "best_mean_mae": study.best_value,
            "folds": result["folds"],
        }
    )
    log.info("best MAE %.4f with %s", study.best_value, study.best_params)
    return result
