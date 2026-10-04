# SPEC — HomeSignal (verbatim brief)

Predict 12-month forward home value growth (percent change) at the U.S. ZIP code level (or county level if ZIP-level
joins prove unreliable; document the decision), using only information available at prediction time.
Secondary target (optional): predict current typical rent from housing and demographic features.

## Data sources (verify all)
Zillow Research (ZHVI / ZORI), Census ACS 5-year, FHFA HPI, HUD FMR, FRED, BLS LAUS. Check documentation, URLs, terms
of use. Do not commit raw data if redistribution is restricted; download at build time with attribution. Record every
source in `docs/sources.md`. Never fabricate data, results, or metrics.

## Tech stack
Python 3.11+, pandas/numpy, scikit-learn, lightgbm, shap, matplotlib, pydantic, pytest, ruff, mypy. Experiment
tracking via mlflow or a structured JSON log (document the choice). Demo via gradio/streamlit deployable to Hugging
Face Spaces. Laptop CPU only.

## Phases
1. Discovery and planning — research sources, decide unit of analysis, write plan to PROGRESS.md.
2. Data pipeline — idempotent cached downloads with retries; official ZIP/ZCTA crosswalk approach documented;
   zero-padded FIPS; monthly panel; explicit missing-data handling and coverage report; never impute the target.
3. Target and leakage prevention — target = pct change ZHVI t → t+12. Features at t use only data published by t.
   Lagged value features from data up to t; ACS vintages assigned to the months they were actually available;
   macro series with publication lags. Unit tests for leakage. `docs/methodology.md` leakage section.
4. Features — momentum, valuation, demographics, macro, geography (no raw identifiers; test with/without).
   Document in `docs/feature_dictionary.md`.
5. Evaluation design — time-based expanding-window backtest with a 12-month gap; untouched final holdout;
   MAE, RMSE (pct points), R², Spearman, directional accuracy; per-fold metrics and fold std.
6. Modeling — naive baselines (zero, trailing-12m, national average), ridge/lasso, RF/LightGBM, modest tuning with
   time-based CV, optional MLP. Compare everything with baselines, honestly.
7. Explainability and error analysis — SHAP global/dependence/local; errors by region, metro size, price tier,
   period (incl. 2020–2022); residual plots; decile calibration.
8. Final model, artifacts, demo — retrain on pre-holdout data, evaluate once on holdout; metadata JSON;
   `predict` CLI; demo app with disclaimer, one-command local run, HF Spaces ready, respects redistribution limits.
9. Documentation — model card, evaluation report, README (headline results, Mermaid architecture, quickstart,
   repo map, attribution).
10. Testing and quality — unit tests (features, crosswalk, splits, leakage, metrics), offline integration test,
    ruff + mypy clean, GitHub Actions, fixed seeds with reproducibility check.

## Working style
Plan first; keep PROGRESS.md updated; commit often; make reasonable documented decisions; never report a metric not
computed in this run; treat suspiciously good results (R² > 0.9) as leakage until proven otherwise. Export the
modelling dataset as a single Parquet with schema + split definitions in `docs/benchmark_handoff.md`.

## Definition of Done
- [x] `make all` downloads, builds, trains, evaluates, writes reports with no manual steps.
- [x] Leakage handling documented and tested.
- [x] Time-based backtest with gap, plus untouched final holdout.
- [x] Baselines, linear, and tree models compared with per-fold metrics.
- [x] SHAP analysis and error analysis with figures.
- [x] Model card, evaluation report, feature dictionary, sources, methodology complete.
- [x] Demo app runs locally and is ready for Hugging Face Spaces.
- [x] `docs/benchmark_handoff.md` and exported modelling dataset ready.
- [x] Tests, ruff, mypy, CI all passing. Results reproducible with fixed seeds.

## Stretch
Quantile/conformal prediction intervals with backtested coverage; rent target; static map of predicted growth.
