"""Export static JSON for the Vercel frontend (run after `make train`).

Writes to web/public/data/:
  meta.json            model summary, origins, holdout metrics, national series, top/bottom lists
  index.json           one compact row per ZIP for search: [zip, city, state, metro, pred_latest]
  shards/<zip3>.json   per ZIP: ZHVI history (monthly since 2010), predictions at the bundled origins
                       with actual outcome where known, and the top-6 SHAP contributions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from homesignal.config import load_data_config, load_evaluation_config
from homesignal.evaluation.splits import is_origin_month
from homesignal.features.build import TARGET, load_panel
from homesignal.models.dataset import encode_categoricals
from homesignal.models.train import load_final_model, load_metadata

OUT = Path(__file__).resolve().parents[1] / "public" / "data"
N_ORIGINS = 8
TOP_K = 6


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shards").mkdir(exist_ok=True)
    data_cfg, eval_cfg = load_data_config(), load_evaluation_config()
    bundle, meta = load_final_model(), load_metadata()
    model, features = bundle["model"], bundle["features"]
    panel = load_panel(data_cfg.processed_dir)
    origins = sorted(panel.loc[is_origin_month(panel["month"], eval_cfg), "month"].unique())[-N_ORIGINS:]
    rows = panel[panel["month"].isin(origins)].reset_index(drop=True)
    X = encode_categoricals(rows, bundle["categorical"])[features]
    rows["pred"] = model.predict(X)
    print("explaining", len(rows), "rows")
    sv = np.asarray(shap.TreeExplainer(model).shap_values(X))
    order = np.argsort(-np.abs(sv), axis=1)[:, :TOP_K]

    zmeta = pd.read_csv(
        data_cfg.raw_dir / "zillow" / "zhvi_zip.csv", dtype={"RegionName": str},
        usecols=["RegionName", "State", "City", "Metro", "CountyName"],
    ).rename(columns={"RegionName": "zip", "State": "state", "City": "city", "Metro": "metro", "CountyName": "county"})
    zmeta["zip"] = zmeta["zip"].str.zfill(5)
    zmeta = zmeta.drop_duplicates("zip").set_index("zip").fillna("")

    hist = panel.loc[panel["month"] >= "2010-01-01", ["zip", "month", "zhvi"]]
    hist_by_zip = {z: g for z, g in hist.groupby("zip")}
    latest = origins[-1]

    def val(v):  # JSON-safe
        if isinstance(v, str):
            return v
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
            return None
        return round(float(v), 4)

    shards: dict[str, dict] = {}
    index = []
    for i, r in enumerate(rows.itertuples(index=False)):
        z = r.zip
        rec = shards.setdefault(z[:3], {}).setdefault(z, {"zip": z, "preds": []})
        contribs = [[features[j], val(X.iat[i, j]), round(float(sv[i, j]), 3)] for j in order[i]]
        rec["preds"].append({
            "origin": str(r.month.date())[:7], "zhvi": int(r.zhvi), "pred": round(float(r.pred), 2),
            "actual": None if pd.isna(getattr(r, TARGET)) else round(float(getattr(r, TARGET)), 2),
            "growth_12m": val(r.growth_12m), "shap": contribs,
        })
    for z3, d in shards.items():
        for z, rec in d.items():
            m = zmeta.loc[z] if z in zmeta.index else None
            rec["city"], rec["state"], rec["metro"], rec["county"] = (
                (m["city"], m["state"], m["metro"], m["county"]) if m is not None else ("", "", "", "")
            )
            h = hist_by_zip.get(z)
            rec["history"] = [[str(mo.date())[:7], int(v)] for mo, v in zip(h["month"], h["zhvi"])] if h is not None else []
            rec["preds"].sort(key=lambda p: p["origin"])
            lp = next((p for p in rec["preds"] if p["origin"] == str(latest.date())[:7]), None)
            index.append([z, rec["city"], rec["state"], rec["metro"], lp["pred"] if lp else None, lp["zhvi"] if lp else None])
        (OUT / "shards" / f"{z3}.json").write_text(json.dumps(d, separators=(",", ":")))
    index.sort()
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))

    lat = rows[rows["month"] == latest].sort_values("pred")
    def lst(df):
        return [{"zip": r.zip, "city": zmeta.loc[r.zip, "city"] if r.zip in zmeta.index else "", "state": zmeta.loc[r.zip, "state"] if r.zip in zmeta.index else "", "pred": round(float(r.pred), 2), "zhvi": int(r.zhvi)} for r in df.itertuples(index=False)]
    by_origin = rows.groupby("month").agg(pred_mean=("pred", "mean"), actual_mean=(TARGET, "mean"), n=("pred", "size"))
    nat = panel.groupby("month")["zhvi"].median()
    nat = nat[nat.index >= "2010-01-01"]
    hm = meta["holdout_metrics"]
    out = {
        "generated_from_commit": None,
        "latest_origin": str(latest.date())[:7],
        "origins": [str(o.date())[:7] for o in origins],
        "model": meta["model"], "training_origins": meta["training_origins"], "training_rows": meta["training_rows"],
        "holdout_origins": meta["holdout_origins"], "holdout_rows": meta["holdout_rows"],
        "holdout": {k: {m: val(v[m]) for m in ("mae", "rmse", "r2", "spearman", "cs_spearman", "directional_accuracy")} for k, v in hm.items()},
        "n_zips": int(rows["zip"].nunique()),
        "by_origin": [{"origin": str(i.date())[:7], "pred_mean": round(float(r.pred_mean), 2), "actual_mean": val(r.actual_mean), "n": int(r.n)} for i, r in by_origin.iterrows()],
        "national_median_zhvi": [[str(i.date())[:7], int(v)] for i, v in nat.items()],
        "top": lst(lat.tail(15).iloc[::-1]), "bottom": lst(lat.head(15)),
        "importance": list(json.loads((Path(__file__).resolve().parents[2] / "reports" / "results" / "shap.json").read_text())["importance"].items())[:12],
    }
    (OUT / "meta.json").write_text(json.dumps(out, separators=(",", ":")))
    size = sum(p.stat().st_size for p in OUT.rglob("*.json")) / 1e6
    print(f"wrote {len(index)} zips, {len(shards)} shards, {size:.1f} MB")


if __name__ == "__main__":
    main()
