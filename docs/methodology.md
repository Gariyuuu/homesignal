# Methodology

## Unit of analysis: ZIP code

The target index (Zillow ZHVI) is published per **USPS ZIP code**; demographics (ACS) are published
per **ZCTA** (Census ZIP Code Tabulation Area). They are not the same thing: ZIPs are mail-delivery
routes, ZCTAs are areal approximations built from census blocks, some ZIPs (PO boxes, single
buildings) have no ZCTA, and ZCTA boundaries were redrawn for the 2020 census.

Decision: keep the ZIP as the unit, join to ZCTA by **direct 5-digit match**. Measured coverage (from
`data/processed/coverage.json`, written by `homesignal build-panel`):

* 26,268 Zillow ZIPs; 26,262 (99.98 %) have a same-numbered ZCTA in at least one ACS vintage.
* 83 % of ZIPs have a complete ZHVI history from 2012-01 onward; the remainder enter later (the panel
  simply starts when the index does — no back-filling).

Why not county? County would make the ZIP≠ZCTA issue vanish, but it discards most of the
cross-sectional signal (3,071 counties vs 26k ZIPs) and the ACS county tables are *not* in the
key-free US-level summary file folder. The code is geography-agnostic enough that a county variant is a
reasonable follow-up (see `docs/known_issues.md`).

The remaining risk of the direct match is that a same-numbered ZCTA can cover a somewhat different
area than the ZIP, so demographic features are "approximately local". Unmatched ZIPs get `NaN`
demographics; LightGBM handles missing values natively and the linear models impute the training
median. The target is never imputed.

## Target

`target_growth_12m = (ZHVI[t+12] / ZHVI[t] − 1) × 100`, in percentage points, where `t` is a month
(the "origin"). Rows whose `t+12` is not yet observed have a `NaN` target and are excluded from
training and evaluation (they are kept in the panel so the demo can predict at the latest origin).

We use the **raw** ZHVI (not the smoothed, seasonally adjusted series). The SA series uses a
two-sided seasonal filter, which means the value published for month *t* is later revised using
months after *t* — a quiet form of look-ahead. The 12-month horizon makes seasonality largely cancel
anyway because origin and target are the same calendar month.

## What "available at prediction time" means here

A row with origin *t* is a prediction made when **ZHVI for month *t* has just been published**
(Zillow publishes month *t* around the third week of *t+1*). Everything else must have been public
by then:

| Input | Rule | Where enforced |
|---|---|---|
| ZHVI momentum/valuation features | computed only from columns `≤ t` (`features/lags.py` uses `shift(h)` with `h ≥ 0`; the only negative shift is the target) | `tests/test_lags.py`, `tests/test_leakage.py` |
| ZORI | same, columns `≤ t` | same |
| State/metro relative value | medians over ZIPs at month `t` only | `tests/test_leakage.py` |
| National growth regime | median across ZIPs of `growth_12m` at `t` | same |
| ACS 5-year vintage *Y* | used only from its **availability month**: January of *Y+2* (Dec *Y+1* release + 1 month), overrides for 2020 (Apr 2022) and 2024 (Feb 2026); at origin *t* the newest available vintage is used | `features/demographics.py`; `tests/test_leakage.py::test_acs_vintage_assignment_respects_release_dates` |
| Mortgage rate (Freddie Mac, weekly, Thursdays) | month *t* average, lag 0 — all of month *t*'s surveys are public before ZHVI(t) | `features/macro.py` |
| CPI (BLS) | month *t−1* (CPI for *t* is released ~10th–15th of *t+1*, which can be after Zillow's release) | `publication_lags.cpi = 1` |
| National unemployment (BLS CES/CPS) | month *t−1* (released first Friday of *t+1*) | `publication_lags.national_unemployment = 1` |
| State unemployment (BLS LAUS) | month *t−2* (state data for *t* come out in the second half of *t+1*; we stay conservative) | `publication_lags.state_unemployment = 2` |

`tests/test_leakage.py::test_no_feature_depends_on_future_data` scrambles **every** input strictly
after a cutoff (ZHVI, ZORI, BLS, mortgage rates, and all ACS vintages not yet released) and asserts
that every feature at every origin `≤ cutoff` is bit-for-bit unchanged while the target does change.
`test_macro_publication_lag_respected` checks each lag individually.

## Leakage risks identified and how each is handled

1. **Target overlap between training and validation.** A row with origin Dec 2015 has a target that
   covers all of 2016. Training on it while validating on 2016 origins leaks 2016 outcomes.
   → Expanding-window folds with a **12-month gap**: training origins satisfy
   `origin + 12 months ≤ first validation origin` (`evaluation/splits.py`, `tests/test_splits.py`).
2. **ACS vintages describing years that are not yet published.** Vintage 2019 describes 2015–2019 but
   was published Dec 2020. → Availability-month assignment described above.
3. **Two-sided seasonal adjustment in ZHVI.** → Raw series.
4. **Macro series publication lags.** → Per-series lags in `configs/features.yaml`.
5. **Location memorisation.** A model that sees the ZIP id can memorise "Austin went up". →
   No raw identifiers (ZIP, metro, county, city) are features; only coarse categorical geography
   (state, Census region/division, metro-size bucket). The evaluation report includes an ablation
   that removes even these.
6. **Calendar memorisation.** Year/month are not features. The "national growth regime" feature
   (median trailing 12-month growth across ZIPs) is the only explicit time-varying national input and
   is computed from the past.
7. **Group-level statistics computed with validation rows.** State/metro medians are per-month
   cross-sectional statistics using only month *t*; the national-mean baseline uses the training mean.
8. **Pre-processing fit on validation data.** Winsorising, imputation and scaling for the linear models
   are fit inside the fold on training rows only (`sklearn.Pipeline`).

## Leakage we cannot remove with public data (disclosed, not fixed)

* **Index revisions.** Zillow revises ZHVI history when it retrains its valuation models; BLS revises
  LAUS annually; ACS does not revise. We use today's vintage of every series, not the vintage that was
  visible at origin *t* (no public archive of ZHVI vintages exists). Trailing-growth features are
  therefore slightly "cleaner" than they would have been in real time. This affects the *features*,
  not the holdout protocol, and it affects every model and baseline equally.
* **Survivorship.** ZIPs that Zillow stopped publishing are absent from today's file.

## Modelling origins

The monthly panel has ~5 M rows (2010-01 → 2026-08). Overlapping 12-month targets at adjacent months
are almost perfectly correlated, so training on every month multiplies compute without adding much
information. Models are trained and evaluated on **quarterly origins** (March, June, September,
December; `configs/evaluation.yaml: origin_months`), ≈1.27 M rows with a known target from 2013-03 to
2025-06. The exported benchmark Parquet (`docs/benchmark_handoff.md`) contains these rows; the full
monthly panel is also written to `data/processed/panel.parquet`.

## Evaluation protocol

* Backtest folds: validation years 2016 … 2023, each an expanding window with the 12-month gap.
* Holdout: origins 2024-03 → 2025-06 (targets observed up to 2026-06/08); the final model is trained on
  origins ≤ 2023-01 (… the holdout start minus the gap) and scored on the holdout **once**.
* Metrics (percentage points): MAE, RMSE, R², Spearman ρ pooled over all rows, **Spearman ρ within
  each origin month** (ranking skill independent of the market-wide level), directional accuracy.
  Per-fold values and fold standard deviations are always reported.

## Missing data policy

* Target: never imputed; rows without a target are excluded from fitting/scoring.
* Momentum features: `NaN` until enough history exists (e.g. `growth_36m` needs 36 months).
* ZORI-based features: `NaN` for ~91 % of rows (ZORI covers 8.4k ZIPs from 2015). Kept because they
  are informative where present and trees handle missingness.
* ACS: `NaN` before 2013-01 (no vintage available) and for unmatched ZIPs; cells suppressed by the
  Census (coded `-666666666` etc. in table-based files, `.` in legacy files) become `NaN`.
* Linear/MLP pipelines: winsorise numeric features at the training 0.5/99.5 percentiles, impute the
  training median, standardise; one-hot encode categoricals. Tree models: native missing handling and
  pandas categoricals with fixed level sets.
