# PROGRESS

Working log for HomeSignal. Reread `SPEC.md` (the brief) and this file if context is lost.

## Phase checklist
- [x] Phase 1 — Discovery and planning
- [ ] Phase 2 — Data pipeline
- [ ] Phase 3 — Target and leakage prevention
- [ ] Phase 4 — Features
- [ ] Phase 5 — Evaluation design
- [ ] Phase 6 — Modeling
- [ ] Phase 7 — Explainability and error analysis
- [ ] Phase 8 — Final model, artifacts, demo
- [ ] Phase 9 — Documentation
- [ ] Phase 10 — Testing and quality

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
