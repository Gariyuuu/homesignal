# Benchmark handoff

`make export` (or `homesignal export`) writes a single Parquet file with the exact rows, features, target
and split labels used in this project, so a downstream benchmarking project can reuse them without
re-running the pipeline.

* **File:** `data/processed/homesignal_benchmark.parquet` (git-ignored; regenerate with `make all` or
  `make export` after `make build`).
* **Schema sidecar:** `data/processed/homesignal_benchmark.schema.json` — column dtypes, feature list,
  fold definitions and split counts, written at export time.
* **Full monthly panel** (all months, rows without a target included): `data/processed/panel.parquet`.

## Rows

One row per (ZIP, origin month) where the origin is a modelling origin (**March, June, September,
December** of each year from **2013-03**) and the 12-month-forward target is observed. ~1.27 M rows,
26k ZIPs, origins 2013-03 → 2025-06 at the time of writing (the last origin advances as Zillow publishes).

## Columns

| Column | dtype | Meaning |
|---|---|---|
| `zip` | string | 5-digit USPS ZIP (zero-padded) |
| `month` | datetime64 | origin month (first day of month) |
| `split` | string | `backtest` (origin < 2024-01) or `holdout` (origin ≥ 2024-01) |
| `fold_val_2016` … `fold_val_2023` | string | per backtest fold: `train`, `val`, or empty (row unused in that fold) |
| `target_growth_12m` | float32 | **target**, pp change in ZHVI from origin to origin + 12 months |
| `zhvi` | float32 | ZHVI level at origin (reference, not a feature) |
| `acs_vintage` | float64 | ACS vintage supplying the demographics (reference) |
| 46 feature columns | float32 / string | see `docs/feature_dictionary.md`; categorical features are `state`, `region`, `division`, `metro_size_bucket` |

## Split definitions (must be reproduced exactly)

* Expanding-window folds, validation year *V* ∈ {2016, …, 2023}:
  training = origins ≤ (first origin of *V*) − 12 months = **March of V−1**; validation = the four
  origins of year *V*. The 12-month gap guarantees no training target window overlaps the validation
  period.
* Holdout: origins ≥ 2024-01; the model is trained on origins ≤ 2023-01 (… holdout start − 12 months,
  i.e. the last usable origin is 2022-12) and scored once.
* The per-fold labels in the Parquet encode all of this; downstream projects should use them rather than
  re-deriving.

## Metrics used

MAE, RMSE (pp), R², Spearman ρ pooled, Spearman ρ averaged within origin month (`cs_spearman`),
directional accuracy (`src/homesignal/evaluation/metrics.py`). Report per-fold values and standard
deviations; compare against the three naive baselines (`zero`, `trailing_12m` = feature `growth_12m`,
`national_mean` = training mean).

## Reference numbers

See `reports/evaluation.md` for the numbers produced by this repo's models on exactly this dataset.

## Caveats carried into any benchmark

Index-revision leakage in features (not fixable with public data), ZIP≠ZCTA join, quarterly origins,
and the unpredictable market-wide component — all described in `docs/methodology.md` and
`docs/known_issues.md`.
