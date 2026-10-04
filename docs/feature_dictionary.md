# Feature dictionary

All features are defined for a row (ZIP, origin month *t*). "Lag" is the publication lag applied so the
value is one that was public when ZHVI(t) was released. Units: pp = percentage points.
`NaN` is allowed wherever noted; the target is never imputed.

## Target

| Name | Definition | Source | Unit |
|---|---|---|---|
| `target_growth_12m` | `(ZHVI[t+12] / ZHVI[t] − 1) × 100` | Zillow ZHVI (raw) | pp |

## Momentum (`features/lags.py`) — from ZHVI columns ≤ *t*, lag 0

| Name | Definition | Unit / notes |
|---|---|---|
| `growth_1m`, `growth_3m`, `growth_6m`, `growth_12m`, `growth_24m`, `growth_36m` | `(ZHVI[t] / ZHVI[t−h] − 1) × 100` | pp; `NaN` if `t−h` missing |
| `volatility_12m` | std of monthly log returns over the trailing 12 months × 100 | pp; needs 12 months |
| `drawdown_36m` | `(ZHVI[t] / max(ZHVI[t−35 … t]) − 1) × 100` | pp, ≤ 0 |
| `log_zhvi` | `ln(ZHVI[t])` | level of home values |

## Valuation (`features/build.py`)

| Name | Definition | Source | Notes |
|---|---|---|---|
| `price_to_income` | `ZHVI[t] / acs_median_income` | Zillow + ACS B19013 | ACS as-of vintage |
| `price_to_rent_zori` | `ZHVI[t] / (12 × ZORI[t])` | Zillow ZORI | `NaN` where ZORI absent (~91 % of rows) |
| `price_to_rent_acs` | `ZHVI[t] / (12 × acs_median_gross_rent)` | ACS B25064 | |
| `zhvi_to_acs_value` | `ZHVI[t] / acs_median_home_value` | ACS B25077 | Zillow estimate vs. owner-reported value |
| `zhvi_rel_state` | `(ZHVI[t] / median over ZIPs in the same state at t − 1) × 100` | Zillow | pp |
| `zhvi_rel_metro` | same, within Zillow metro | Zillow | `NaN` outside metros |
| `rent_growth_12m_zori` | `(ZORI[t] / ZORI[t−12] − 1) × 100` | ZORI | pp |

## Demographics (`features/demographics.py`) — ACS 5-year, as-of vintage (see methodology)

| Name | Definition | ACS cells |
|---|---|---|
| `acs_population` | total population | B01003_001 |
| `acs_median_age` | median age | B01002_001 |
| `acs_share_under18` | population under 18 / total | B01001 lines 3–6 (M), 27–30 (F) / 001 |
| `acs_share_25_34` | population 25–34 / total | B01001 11–12, 35–36 |
| `acs_share_65plus` | population 65+ / total | B01001 20–25, 44–49 |
| `acs_bachelor_share` | bachelor's or higher / population 25+ | B15003 22–25 / 001 (2012+); B15002 15–18 + 32–35 / 001 (2011) |
| `acs_median_income` | median household income (USD, nominal) | B19013_001 |
| `acs_housing_units` | total housing units | B25001_001 |
| `acs_vacancy_rate` | vacant / total units | B25002_003 / 001 |
| `acs_renter_share` | renter-occupied / occupied units | B25003_003 / 001 |
| `acs_median_gross_rent` | median gross rent (USD/month) | B25064_001 |
| `acs_median_home_value` | median owner-estimated value (USD) | B25077_001 |
| `acs_unemployment_rate` | unemployed / civilian labour force | B23025_005 / 003 (`NaN` for vintage 2011) |
| `acs_lfpr` | labour force / population 16+ | B23025_002 / 001 |
| `acs_pop_growth` | `(pop[v] / pop[v−1] − 1) × 100` between consecutive vintages | derived; `NaN` if v−1 missing |
| `acs_income_growth` | same for median income | derived |
| `acs_housing_unit_growth` | same for housing units | derived |

Vintage-over-vintage growth compares two overlapping five-year windows offset by one year; treat it as
a slow-moving trend indicator, not an annual growth rate.

## Macro (`features/macro.py`)

| Name | Definition | Source | Lag |
|---|---|---|---|
| `mortgage_rate_30y` | mean of weekly PMMS 30-year rates in month *t* | Freddie Mac | 0 |
| `mortgage_rate_chg_3m`, `mortgage_rate_chg_12m` | change vs. 3 / 12 months earlier (pp) | Freddie Mac | 0 |
| `national_unemployment` | unemployment rate, SA | BLS LNS14000000 | 1 month |
| `national_unemployment_chg_12m` | change vs. 12 months earlier (pp) | BLS | 1 month |
| `cpi_yoy` | `(CPI[t−1] / CPI[t−13] − 1) × 100` | BLS CUUR0000SA0 | 1 month |
| `state_unemployment` | state unemployment rate, SA | BLS LASST… | 2 months |
| `state_unemployment_chg_12m` | change vs. 12 months earlier (pp) | BLS | 2 months |
| `national_growth_12m` | median across all ZIPs of `growth_12m` at *t* | Zillow | 0 |

## Geography (`features/geography.py`) — categorical, static

| Name | Definition | Levels |
|---|---|---|
| `state` | USPS state | 51 |
| `region` | Census region of the state | Northeast, Midwest, South, West |
| `division` | Census division | 9 |
| `metro_size_bucket` | number of Zillow ZIPs in the ZIP's metro: `non_metro`, `<10 small`, `<50 mid`, `<200 large`, `≥200 mega` | 5 |

Raw identifiers (ZIP, metro name, county, city) are deliberately **not** features.

## Panel columns that are not features

`zip`, `month` (origin, month start), `zhvi` (level at *t*, for reference), `acs_vintage` (which vintage
supplied the demographics).
