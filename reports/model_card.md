# Model card — HomeSignal LightGBM (12-month ZIP home value growth)

## Model details

* **Developer:** Gary Wang (personal, educational project), October 2026.
* **Model type:** gradient-boosted regression trees (LightGBM 4.7.0), `lightgbm_tuned` with Optuna-selected parameters in `models/lightgbm_tuned_params.json`.
* **Input:** 46 features for a (ZIP, origin month) row — see `docs/feature_dictionary.md`.
* **Output:** point estimate of the percent change in the Zillow Home Value Index (mid-tier, raw) from
  the origin month to 12 months later, in percentage points.
* **Training data:** 1,009,637 rows at quarterly origins 2013-03 → 2022-12, ~26k ZIPs.
* **Artifacts:** `models/homesignal_lightgbm.joblib`, `models/homesignal_lightgbm.metadata.json`
  (features, params, training window, holdout metrics, library versions, data file hashes).

## Intended use

* Educational demonstration of a leakage-aware time-series ML workflow on public data.
* Exploring *relative* growth expectations across ZIP codes (ranking), with explanations.

## Out-of-scope uses

* **Not** for lending, underwriting, appraisal, valuation, pricing, insurance, tax assessment or any
  individual financial or investment decision.
* Not a forecast of the national housing market: the model has no demonstrated skill on the
  market-wide level of growth (see Evaluation).
* Not for ZIPs outside Zillow's coverage, for sub-ZIP properties, or for horizons other than 12 months.

## Training data

* Zillow ZHVI (raw) and ZORI, ZIP level, monthly; ACS 5-year vintages 2011–2024 (ZCTA, joined by
  same-numbered ZCTA); Freddie Mac PMMS; BLS CPI and national/state unemployment. Sources, terms and
  release-lag assumptions: `docs/sources.md`.
* Every feature uses only data public at the origin month; see `docs/methodology.md` for the per-input
  rules and the leakage tests.

## Evaluation

Protocol: expanding-window backtest with validation years 2016–2023 and a 12-month gap, then a single
evaluation on holdout origins 2024-01 → 2025-06 (157,605 rows) never used before `train-final`.

### Holdout (final model vs. naive baselines)

| Model | MAE (pp) | RMSE (pp) | R² | ρ pooled | ρ within origin | Dir. acc. |
|---|---|---|---|---|---|---|
| **Final LightGBM (tuned)** | 5.84 | 7.34 | -1.174 | 0.044 | 0.094 | 0.686 |
| Zero growth | 4.02 | 5.28 | -0.123 | — | — | 0.686 |
| Trailing 12m continues | 4.47 | 6.27 | -0.588 | 0.265 | 0.312 | 0.707 |
| Training mean | 5.77 | 7.15 | -1.061 | — | — | 0.686 |

### Backtest (LightGBM default params, mean ± std across 8 folds)

| Model | MAE (pp) | ρ within origin |
|---|---|---|
| Trailing 12m continues | 6.18 ± 2.64 | 0.257 ± 0.116 |
| Training mean | 5.33 ± 2.50 | — |
| LightGBM | 4.91 ± 2.34 | 0.312 ± 0.164 |

Per-fold tables, ablations (no geography, no demographics), error breakdowns by year/region/metro
size/price tier, decile calibration and SHAP figures: `reports/evaluation.md`.

### Reading these numbers

* Typical absolute error on the holdout is about 5.8 pp on a target whose standard deviation
  is roughly 5 pp in calm years and 7+ pp in 2020–2022.
* Pooled R² is -1.17; it is low or negative in regime years for every model because the national
  component of growth is not predictable from these inputs.
* **The final model is worse than the zero-growth and trailing-growth baselines on the holdout** (MAE 5.84 vs 4.02 and 4.47); its predictions are biased upward by ≈6 pp. Within-origin Spearman ρ of 0.09 on the holdout (vs. 0.31 for the
  trailing-growth baseline) is the evidence of ranking skill.

## Limitations

* **Index revisions.** Features use today's revised ZHVI/LAUS series, not the vintage visible at the
  time; a real-time user would see noisier inputs. Not fixable with public data; disclosed.
* **ZIP ≠ ZCTA.** Demographics are approximately local; a 2020 ZCTA redraw changes some areas.
* **Coverage gaps.** Rural and small ZIPs enter Zillow's file later or not at all; ZORI covers only
  8.4k ZIPs; ACS cells are suppressed in small ZCTAs. Errors are larger for small/non-metro and
  low-price-tier ZIPs (see the breakdowns in `reports/evaluation.md`).
* **Regime dependence.** Performance through 2020–2022 is markedly worse; the model learned from one
  such regime at most.
* **Quarterly origins.** Trained on March/June/September/December origins; predictions at other months
  use the same features but were not separately validated.
* **No uncertainty estimates.** Point predictions only (prediction intervals are a stretch goal).

## Potential biases and ethical considerations

* Demographic features (income, education, age structure, tenure, race is **not** used) describe places,
  not people, but a model that learns "high-income ZIPs appreciate more" can reinforce existing disparities
  if misused for allocation decisions. This is one reason the model is out of scope for lending or pricing.
* Geographic coverage is uneven: dense metros are over-represented; sparse rural data mean worse and
  potentially systematically biased predictions there.
* Home-value growth is a societal outcome with winners and losers; the project's value is methodological,
  not advisory. The demo carries a disclaimer and shows typical error alongside every prediction.

## Reproducibility

`make all` rebuilds everything from public sources; seeds are fixed (`configs/model.yaml: seed`),
LightGBM runs with `deterministic=True`, and the integration test asserts two fits give identical
predictions. A full-scale check (`reports/results/reproducibility.json`) refit the final model twice and compared to the saved artifact: all three prediction vectors are bit-identical (max |diff| = 0).
