# Known issues and limitations

Honest list of what is weak, approximate or unverified. See also the limitations section of
`reports/model_card.md`.

## Data

1. **Index revisions are not reconstructed.** ZHVI, ZORI and BLS LAUS are revised after first
   publication; the pipeline uses today's vintage of each series everywhere. Trailing-growth features are
   therefore somewhat cleaner than a real-time user would have seen. There is no public ZHVI vintage
   archive, so this cannot be fixed with free data. It affects features, not the time-split protocol.
2. **ZIP ≠ ZCTA.** Demographics are joined by same-numbered ZCTA (99.98 % of ZIPs match on the number),
   but a ZCTA can cover a different area than its ZIP, and 2020-geography ZCTAs (vintages 2020+) differ
   from 2010-geography ZCTAs (vintages 2011–2019). HUD's USPS crosswalk would be better but requires a
   token.
3. **ACS 2009/2010 vintages are not used** (the key-free US-level summary files do not include ZCTAs for
   them), so demographic features start in 2013-01 and the first modelling origin is 2013-03.
4. **ACS vintage-over-vintage "growth" features** compare overlapping five-year windows; they are
   trend proxies, not annual rates.
5. **ZORI covers only ~8.4k ZIPs from 2015**, so `price_to_rent_zori` and `rent_growth_12m_zori` are
   missing for ~91 % of rows.
6. **Metro size** is proxied by the number of Zillow ZIPs in the metro (population would be better but
   requires a metro-level join we did not build).
7. **FHFA HPI and HUD FMR are downloaded-able but unused**; FHFA could serve as a cross-check of
   ZHVI-based targets (deferred).
8. **Survivorship.** ZIPs Zillow stopped publishing are missing from the current file.
9. **Nominal values.** Incomes, rents and values are nominal; `cpi_yoy` is the only inflation input.

## Modelling

10. **Quarterly origins only.** Models are trained on March/June/September/December origins to keep the
    laptop budget (~1.3 M rows). The full monthly panel is built and exported; the demo bundles quarterly
    rows.
11. **The market-wide level of growth is not predictable from these features.** Most of the target
    variance in any year is the national component (2020–2022 especially); every model, including the
    baselines, misses it. The useful skill is cross-sectional ranking (within-origin Spearman). Treat MAE
    and R² in the report accordingly.
12. **Hyper-parameter tuning is modest** (25 Optuna trials on the last three folds) and tuned on the same
    folds that are reported in the backtest, so the "tuned" backtest numbers are slightly optimistic; the
    holdout is the clean number.
13. **No prediction intervals yet** (stretch goal).
14. **LightGBM `deterministic=True` with `n_jobs=-1`**: results are reproducible on the same machine
    and library versions; across CPUs/library versions small floating-point differences can appear.

## Engineering

15. **BLS API v1 quota.** 25 queries/day per IP; a fresh build uses 10. Cached JSON responses mean
    rebuilds use none. Set `BLS_API_KEY` for v2 limits.
16. **Raw download size** is ~2.5 GB (table-based ACS files are large because they contain every
    geography; only ZCTA rows are kept).
17. **Peak memory** of `build-panel` is ~7 GB (wide 26k × 367 frames and the 5 M-row panel).
18. **The Hugging Face Space bundle** includes the per-ZIP ZHVI history (needed for the chart). If
    Zillow's terms change, switch `app/prepare_app_data.py` to download the history at startup instead.
19. The Census API path (with `CENSUS_API_KEY`) is intentionally **not** implemented so the project has no
    untested code paths; the FTP summary files cover every vintage.
