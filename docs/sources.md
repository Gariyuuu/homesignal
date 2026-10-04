# Data sources

Every source below was probed live on 2026-10-02 from this project's download code; nothing in the
repo is hand-copied. **No API keys are required.** Raw files are downloaded at build time into
`data/raw/` (git-ignored) with a `.meta.json` sidecar recording URL, timestamp, size and SHA-256.

| Source | What we use | URL | Geography | Frequency / coverage | Terms (as read on the provider's site) |
|---|---|---|---|---|---|
| **Zillow Research — ZHVI** | Zillow Home Value Index, mid-tier (33rd–67th percentile), all homes (SFR + condo), **raw** (not smoothed, not seasonally adjusted) | `https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv` | ZIP (26,268 ZIPs) | Monthly, 1996-02 → 2026-08 (most ZIPs start 2000) | Zillow makes its research data freely available for download; Zillow's Terms of Use govern it and ask for attribution ("Data: Zillow Research, zillow.com/research/data"). We do not commit or redistribute the raw CSV; the demo bundles only the per-ZIP index history needed for its chart, with attribution. |
| **Zillow Research — ZORI** | Zillow Observed Rent Index, all homes + multifamily, smoothed | `.../zori/Zip_zori_uc_sfrcondomfr_sm_month.csv` | ZIP (8,459 ZIPs) | Monthly, 2015-01 → 2026-08 | Same as above. |
| **U.S. Census Bureau — ACS 5-year estimates** | Tables B01001 (sex by age), B01002 (median age), B01003 (population), B15002/B15003 (education), B19013 (median household income), B23025 (employment status, 2012+), B25001 (housing units), B25002 (occupancy), B25003 (tenure), B25064 (median gross rent), B25077 (median home value) | Legacy sequence files: `https://www2.census.gov/programs-surveys/acs/summary_file/{Y}/data/5_year_seq_by_state/UnitedStates/All_Geographies_Not_Tracts_Block_Groups/` (vintages 2011–2020). Table-based files: `.../summary_file/{Y}/table-based-SF/data/5YRData/acsdt5y{Y}-{table}.dat` (2021–2024) | ZCTA (32,989 in 2010-vintage geography; 33,772 in 2020 geography) | One vintage per year; vintage *Y* covers *Y−4 … Y* | Public domain (U.S. government work). The Census API now requires a key for all requests, so the FTP summary files are used instead. |
| **Freddie Mac — PMMS** | 30-year fixed mortgage rate, weekly survey, averaged to monthly | `https://www.freddiemac.com/pmms/docs/PMMS_history.csv` | National | Weekly, 1971-04 → present | "Information from this document may be used with proper attribution. Alteration of this document or its content is strictly prohibited." We use the values unaltered and attribute Freddie Mac. |
| **BLS — Public Data API v1** | `LNS14000000` national unemployment rate (SA); `CUUR0000SA0` CPI-U all items (NSA); `LASST{fips}0000000000003` state unemployment rate (SA), 51 states/DC | `https://api.bls.gov/publicAPI/v1/timeseries/data/` | National, state | Monthly, 2000 → present | Public domain. v1 limits: 25 queries/day, 25 series/query, 10 years/query; a full build uses 10 queries and the JSON responses are cached. Set `BLS_API_KEY` to use v2 (optional). |

## Sources considered and not used

| Source | Status | Reason |
|---|---|---|
| Census API (`api.census.gov`) | Returns a "Missing Key" page for keyless requests (verified 2026-10-02) | Replaced by the key-free FTP summary files so `make all` needs no signup. |
| FRED (`fred.stlouisfed.org/graph/fredgraph.csv`) | TCP connects but never responds from the build machine (both plain and browser user agents) | Mortgage rates come from Freddie Mac directly (FRED's `MORTGAGE30US` *is* the PMMS); CPI/unemployment from the BLS API. Supporting FRED would add a second, untested path. |
| BLS flat files (`download.bls.gov/pub/time.series/…`) | HTTP 403 for scripted clients | The BLS API serves the same series. |
| FHFA House Price Index (ZIP5 / county, annual) | Downloadable (note: the `.csv` links serve xlsx) | Annual frequency and a second index would duplicate ZHVI's role; deferred, listed in `docs/known_issues.md` as a possible cross-check. |
| HUD Fair Market Rents | Downloadable with a browser user agent (xlsx per fiscal year, format varies by year) | Price-to-rent is covered by ZORI (2015+) and ACS median gross rent (all years). Deferred. |
| HUD USPS ZIP–ZCTA crosswalk | Requires a HUD API token | We join ZIP to ZCTA by direct 5-digit match (99.98 % of Zillow ZIPs exist as ZCTAs) and document the caveat; see `docs/methodology.md`. |
| Census ZCTA–county relationship file (2020) | Downloadable | Not needed at ZIP level; kept as a reference for a county-level variant. |

## ACS release dates used for availability

| Vintage | Released | Usable as a feature from |
|---|---|---|
| 2011–2019 | December of *Y+1* | January of *Y+2* (rule: `default_availability_lag_months: 13` after January of *Y+1*) |
| 2020 | 2022-03-17 (delayed by COVID data-quality work) | 2022-04 (override) |
| 2021 | 2022-12-08 | 2023-01 |
| 2022 | 2023-12-07 | 2024-01 |
| 2023 | 2024-12-12 | 2025-01 |
| 2024 | 2026-01-08 (per census.gov press kit) | 2026-02 (override) |

Dates for 2011–2023 follow the Census Bureau's regular December release cadence; the two overrides
are the known exceptions. These assumptions live in `configs/data.yaml`.
