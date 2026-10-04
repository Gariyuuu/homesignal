export type Contribution = [feature: string, value: number | string | null, shap: number];

export interface Pred {
  origin: string;
  zhvi: number;
  pred: number;
  actual: number | null;
  growth_12m: number | null;
  shap: Contribution[];
}

export interface ZipRecord {
  zip: string;
  city: string;
  state: string;
  metro: string;
  county: string;
  history: [month: string, zhvi: number][];
  preds: Pred[];
}

export type IndexRow = [zip: string, city: string, state: string, metro: string, pred: number | null, zhvi: number | null];

export interface Meta {
  latest_origin: string;
  origins: string[];
  model: string;
  training_origins: [string, string];
  training_rows: number;
  holdout_origins: [string, string];
  holdout_rows: number;
  holdout: Record<string, Record<string, number | null>>;
  n_zips: number;
  by_origin: { origin: string; pred_mean: number; actual_mean: number | null; n: number }[];
  national_median_zhvi: [string, number][];
  top: { zip: string; city: string; state: string; pred: number; zhvi: number }[];
  bottom: { zip: string; city: string; state: string; pred: number; zhvi: number }[];
  importance: [string, number][];
}

const cache = new Map<string, Promise<unknown>>();

function getJSON<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(path).then((r) => {
        if (!r.ok) throw new Error(`${path}: ${r.status}`);
        return r.json();
      }),
    );
  }
  return cache.get(path) as Promise<T>;
}

export const loadMeta = () => getJSON<Meta>("/data/meta.json");
export const loadIndex = () => getJSON<IndexRow[]>("/data/index.json");

export async function loadZip(zip: string): Promise<ZipRecord | null> {
  const shard = await getJSON<Record<string, ZipRecord>>(`/data/shards/${zip.slice(0, 3)}.json`).catch(() => null);
  return shard?.[zip] ?? null;
}

export const FRIENDLY: Record<string, string> = {
  growth_1m: "last month's growth",
  growth_3m: "trailing 3-month growth",
  growth_6m: "trailing 6-month growth",
  growth_12m: "trailing 12-month growth",
  growth_24m: "trailing 24-month growth",
  growth_36m: "trailing 36-month growth",
  volatility_12m: "12-month volatility",
  drawdown_36m: "distance from 3-year peak",
  log_zhvi: "home value level",
  price_to_income: "price-to-income ratio",
  price_to_rent_zori: "price-to-rent (ZORI)",
  price_to_rent_acs: "price-to-rent (ACS)",
  zhvi_to_acs_value: "Zillow value vs owner-reported value",
  zhvi_rel_state: "value vs state median",
  zhvi_rel_metro: "value vs metro median",
  rent_growth_12m_zori: "12-month rent growth",
  mortgage_rate_30y: "30-year mortgage rate",
  mortgage_rate_chg_3m: "3-month change in mortgage rate",
  mortgage_rate_chg_12m: "12-month change in mortgage rate",
  national_unemployment: "national unemployment",
  national_unemployment_chg_12m: "12-month change in national unemployment",
  cpi_yoy: "inflation (CPI y/y)",
  state_unemployment: "state unemployment",
  state_unemployment_chg_12m: "12-month change in state unemployment",
  national_growth_12m: "national 12-month growth",
  state: "state",
  region: "Census region",
  division: "Census division",
  metro_size_bucket: "metro size",
};

export const friendly = (f: string) => FRIENDLY[f] ?? f.replace(/^acs_/, "ACS ").replace(/_/g, " ");

export function fmtValue(feature: string, v: number | string | null): string {
  if (v === null || v === undefined) return "n/a";
  if (typeof v === "string") return v;
  if (feature === "log_zhvi") return `$${Math.round(Math.exp(v)).toLocaleString()}`;
  if (feature.startsWith("acs_share") || feature.includes("_share") || feature.includes("_rate") && feature.startsWith("acs") || feature === "acs_lfpr")
    return `${(v * 100).toFixed(1)}%`;
  if (feature.startsWith("acs_median") || feature === "acs_population" || feature === "acs_housing_units") return Math.round(v).toLocaleString();
  if (feature.includes("mortgage") || feature.includes("unemployment") || feature.includes("growth") || feature.includes("cpi") || feature.includes("rel_") || feature.includes("drawdown") || feature.includes("volatility"))
    return `${v.toFixed(2)}${feature.includes("rate_30y") || feature.includes("unemployment") && !feature.includes("chg") ? "%" : " pp"}`;
  return v.toFixed(2);
}

export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
