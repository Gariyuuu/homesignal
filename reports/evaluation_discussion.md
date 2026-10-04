### Honest discussion

**The headline result is negative.** On the untouched holdout (origins 2024-03 → 2025-06) the final
LightGBM has MAE 5.84 pp, worse than predicting zero growth (4.02) and worse than "trailing 12-month
growth continues" (4.47). Its predictions are biased upward by about 6 pp: the mean prediction per origin
was 5.7–9.2 % while realised growth was 0.8–2.9 %. Within-origin Spearman is 0.09 — almost no ranking
skill in that period.

**Why.** Three things line up:

1. *Regime extrapolation on macro features.* `mortgage_rate_30y` is the most important feature by mean
   |SHAP|, followed by `mortgage_rate_chg_12m` and `national_unemployment_chg_12m`. The SHAP summary shows
   the pattern the model learned: very low rates (2020–21) → +8 pp. Training ends at origin 2022-12, so the
   model saw exactly one rate cycle, and the holdout sits in a rate regime (6.5–7 %) it had barely seen.
   Tree models cannot extrapolate; they map the holdout onto the nearest leaves of a different regime.
2. *The national component dominates and is unpredictable here.* Year-to-year, realised growth swings
   between −4 % (2010) and +15 % (2020) on average across ZIPs. No model in the backtest beats the
   training-mean baseline by much on MAE in calm years, and all models miss 2020 by ~10 pp. R² is negative
   in most folds for every model because the market-wide level is most of the variance.
3. *Ranking skill decayed before the holdout.* LightGBM's within-origin Spearman was 0.49 / 0.50 / 0.30 in
   2016–18, 0.36 / 0.34 in 2020–21, then 0.06 and 0.08 in 2022–23. The trailing-growth baseline fell
   less (0.08, 0.28). The post-2021 market is less momentum-driven and more rate-driven, and the model
   had no way to learn that.

**What does work.** Over the 8 backtest folds LightGBM and the random forest beat every naive baseline on
MAE (4.9 and 4.8 vs 5.3 for the training mean, 6.2 for trailing growth) and rank ZIPs with a within-origin
Spearman of ≈0.31 on average — real but modest cross-sectional skill, concentrated in 2016–2021.
Momentum (`growth_12m`, `growth_1m`), relative value (`zhvi_rel_metro`) and state are the stable
contributors. The geography ablation hurts (MAE 5.05 without state/region/metro bucket); dropping ACS
demographics changes nothing (4.91), i.e. the slow-moving demographic features add essentially no
12-month signal beyond what momentum and location already carry.

**The no-macro ablation.** Dropping all nine macro series (37 features left) makes the backtest *worse* on average: MAE 5.86 ± 3.27 vs 4.91 ± 2.34 with them, within-origin Spearman 0.30 vs 0.31. The difference is concentrated in 2022 (MAE 11.1 without vs 5.0 with): the rising-rate features let the model anticipate the 2022 slowdown that momentum alone could not. In 2023 the no-macro model ranks slightly better (0.19 vs 0.08). So the macro features carry real signal in-sample, but it is a signal learned from a single cycle, and on the holdout the same features drove the +6 pp bias. That is the central lesson of this project: with ~12 years of history there is one rate cycle to learn from, and any feature that encodes it will look good in a backtest that spans it and fail when the regime moves on.

**Calibration.** On the holdout the decile calibration is flat-to-inverted: the top predicted decile
(mean 10.7 %) realised 1.3 %, the bottom decile (3.6 %) realised 0.2 %. On the backtest folds calibration
is monotone (higher predicted deciles do realise higher growth) but the model under-predicts by 1–3 pp in
every decile — the opposite sign of the holdout bias, which is further evidence that the level is regime noise. Predictions should be read as rankings, not as magnitudes.

**Where the errors are.** Holdout MAE is 4.2 in the Midwest and 4.8 in the Northeast but 7.1 in the South
and 7.3 in the West — the Sun Belt/West markets that boomed in 2020–21 and then stalled are exactly where
the regime-extrapolation bias is largest (+6.6 and +7.1 pp bias). Non-metro ZIPs (MAE 6.2 backtest / 6.6 holdout) and bottom-quintile ZIPs (6.2 / 7.7) have the highest
errors in both periods (see the breakdown tables).

**What I would do next** (not done, to keep the holdout clean):

* Remove or de-emphasise level macro features and keep only cross-sectional ones, or predict
  *relative* growth (ZIP minus national) and leave the national level to a separate, explicitly
  uncertain component.
* Train with sample weights favouring recent origins, or with a rolling (not expanding) window, so the
  2020–21 regime does not dominate.
* Quantile regression / conformal intervals so the demo shows the (wide) uncertainty.
* A longer history (ZHVI goes back to 2000; ACS limits the start to 2013) to see more than one rate cycle —
  possible by letting demographics be missing pre-2013.

**Protocol note.** The holdout was evaluated exactly once with the pre-registered model. Nothing above was
used to change that model. The ablations are backtest-only experiments.
