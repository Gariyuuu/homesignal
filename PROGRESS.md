# PROGRESS

Working log for HomeSignal. Reread `SPEC.md` (the brief) and this file if context is lost.

## Phase checklist
- [x] Phase 1 — Discovery and planning
- [x] Phase 2 — Data pipeline
- [x] Phase 3 — Target and leakage prevention
- [x] Phase 4 — Features
- [x] Phase 5 — Evaluation design
- [x] Phase 6 — Modeling
- [x] Phase 7 — Explainability and error analysis
- [x] Phase 8 — Final model, artifacts, demo
- [x] Phase 9 — Documentation
- [x] Phase 10 — Testing and quality

## Log

### 2026-10-02 — Phase 1: source discovery (all probed live with curl)
Findings:
- **Zillow** ZHVI ZIP (26,268 ZIPs, 2000-01 → 2026-08) and ZORI ZIP (8,459 ZIPs, 2015-01 →) download without a key.
  A raw (not smoothed, not seasonally adjusted) ZHVI variant exists; the SA variant uses a two-sided filter,
  so the raw series is used for target and momentum features.
- **Census API** now returns a "Missing Key" page for keyless requests. Rather than require a signup, ACS 5-year
  data is pulled from the key-free Census FTP: legacy sequence files in the `UnitedStates` folder (contains all
  32,989 ZCTAs for vintages 2011–2020; 2009/2010 have no ZCTA rows there) and table-based files for 2021–2024.
  Counties are *not* in that folder (they are per-state) — one reason ZIP/ZCTA is the unit of analysis.
- **FRED** connects but never responds from this network → not used. Mortgage rates come from **Freddie Mac PMMS**
  (weekly CSV, 1971 →); CPI and unemployment (national + state) come from the **BLS Public API v1** (no key,
  25 queries/day, 25 series/query, 10 years/query; ~10 queries per full build). BLS flat files return 403 to scripts.
- **FHFA** HPI "csv" downloads are actually xlsx; **HUD FMR** needs a browser UA. Both deferred (ZHVI is the index,
  ZORI/ACS rent give price-to-rent). Documented in docs/sources.md.
- Census ZCTA↔county relationship file is available (used only for documentation of ZIP≠ZCTA caveat).
Decisions:
- Unit of analysis: **ZIP code** (Zillow ZIP series) joined to ACS **ZCTA** by direct 5-digit match. Coverage reported in build.
- Modelling origins: quarterly (Mar/Jun/Sep/Dec) to keep ~1M rows on a laptop; monthly panel is still built and exported.
- Experiment tracking: JSON-lines log (`models/experiments.jsonl`); mlflow judged too heavy.
Next: data pipeline.

### 2026-10-02 — Phases 2–5 built and verified
- Data layer (`src/homesignal/data/`): cached downloader with retries + SHA-256 sidecars; Zillow wide loaders;
  ACS legacy-sequence parser (lookup → sequence/start column, geo file → LOGRECNO→ZCTA) and table-based parser
  (ZCTA GEO_ID prefixes `8600000US` and `860Z200US`; sentinel values ≤ −222222222 → NaN); BLS API v1 client
  (10 queries per fresh build, "-" → NaN); Freddie Mac weekly → monthly mean.
  Live-verified: ZIP 94110 vintage-2019 median income $134,592 matches published ACS.
- Panel (`features/build.py::assemble_panel`, pure function so tests run offline): 4.98 M rows × 51 cols from
  2010-01, 18 s, ~7 GB peak RAM. Hand-check: `growth_12m` and target for 94110 @ 2019-06 equal values recomputed
  from the raw CSV; ACS vintage in use is 2017 (released Dec 2018); macro lags as configured.
- Coverage (`data/processed/coverage.json`): 26,268 ZIPs; 99.98 % have a same-numbered ZCTA; 83 % complete ZHVI
  since 2012; ZORI features missing on 91 % of rows; ACS missing on 16 % (pre-2013 months).
- Leakage tests (`tests/test_leakage.py`): scrambling every input after a cutoff leaves all features at origins
  ≤ cutoff bit-identical; per-series lag tests; ACS availability test; target-never-imputed test.
- Splits: 8 expanding folds (val 2016–2023) with 12-month gap; holdout origins ≥ 2024-01. Quarterly origins
  → 1,272,314 modelling rows (2013-03 → 2025-06), 46 features.
- First backtest (pre-fix): ridge exploded on fold 2016 (MAE 23.9) because of extreme ratio features in
  validation → added a training-percentile winsoriser to the linear/MLP pipelines. Added within-origin Spearman
  (`cs_spearman`) because the pooled numbers are dominated by the unpredictable national level.
- Tooling: ruff + mypy --strict clean, 33 offline tests, Makefile, GitHub Actions CI.
- Pipeline chain (tune → train-final → explain → export → report, then backtests main/no_geo/no_acs) running.

### 2026-10-03 — Phases 6–10: results, artifacts, demo, docs
- Backtest (`reports/results/backtest_main.json`, 8 folds): LightGBM MAE 4.91 ± 2.34 vs training-mean 5.33, trailing
  6.18, zero 7.81; RF 4.81; lasso 5.12; ridge 6.86 (fold 2016 still bad even after winsorising). Within-origin Spearman:
  LightGBM 0.31 ± 0.16 (0.49 in 2016–17, 0.06–0.08 in 2022–23), trailing baseline 0.26 ± 0.12.
- Ablations: no geography → MAE 5.05 (worse); no ACS → 4.91 (no change); no macro → see evaluation.md (run last).
- Tuning: 25 Optuna trials on folds 2021–23, best mean MAE 4.82 (num_leaves 18, 200 trees, lr 0.045, min_child 620).
- **Holdout (touched once, origins 2024-03 → 2025-06, 157,605 rows): final model MAE 5.84, R² −1.17, within-origin
  ρ 0.09; zero-growth baseline MAE 4.02, trailing 4.47.** Predictions biased ≈ +6 pp: mean prediction 6–9 % vs actual
  1–3 %. SHAP: `mortgage_rate_30y` is the top feature; low 2020–21 rates carry +8 pp contributions → the model learned
  the boom regime and extrapolated. Ranking skill collapsed in 2022–23 folds already — visible before the holdout.
- Decision: the holdout number is reported as the headline, unchanged; the no-macro variant is a backtest ablation and
  a *labelled post-hoc* diagnostic only. No re-tuning on the holdout.
- Artifacts: `models/homesignal_lightgbm.joblib` + metadata JSON (features, params, windows, metrics, versions, source
  hashes); `predict` CLI with SHAP top features; Gradio app bundle (51 MB) smoke-tested; benchmark Parquet exported.
- Reproducibility (`reports/results/reproducibility.json`): two full refits and the saved artifact are bit-identical.
- Docs: sources, methodology, feature dictionary, known issues, benchmark handoff, model card, evaluation report
  (generated), README.
