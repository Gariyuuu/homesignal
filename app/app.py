"""HomeSignal demo (Gradio). Runs locally with `make app`; deployable to Hugging Face Spaces as-is.

The app is self-contained: it reads the bundle in ``app/data`` (model + derived features +
ZHVI history) and imports only ``homesignal.models.dataset`` for the categorical encoding.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from homesignal.models.dataset import encode_categoricals

DATA = Path(__file__).resolve().parent / "data"
DISCLAIMER = (
    "**Educational project — not financial, investment, lending or appraisal advice.** "
    "Predictions are a statistical model's point estimate with large errors (see the model card); "
    "they are not forecasts you should act on. Home value data © Zillow Research; demographics from "
    "the U.S. Census Bureau ACS; mortgage rates from Freddie Mac PMMS; labour data from BLS."
)

bundle = joblib.load(DATA / "homesignal_lightgbm.joblib")
META = json.loads((DATA / "homesignal_lightgbm.metadata.json").read_text())
FEATURES = pd.read_parquet(DATA / "features.parquet")
HISTORY = pd.read_parquet(DATA / "history.parquet")
ZIP_META = pd.read_parquet(DATA / "zip_meta.parquet").set_index("zip")
ORIGINS = sorted(FEATURES["month"].unique())
EXPLAINER = shap.TreeExplainer(bundle["model"])
HOLDOUT_MAE = META["holdout_metrics"]["final_model"]["mae"]
FRIENDLY = {
    "growth_12m": "trailing 12-month growth",
    "growth_36m": "trailing 36-month growth",
    "growth_24m": "trailing 24-month growth",
    "growth_6m": "trailing 6-month growth",
    "growth_3m": "trailing 3-month growth",
    "growth_1m": "last month's growth",
    "national_growth_12m": "national 12-month growth (median ZIP)",
    "mortgage_rate_30y": "30-year mortgage rate",
    "mortgage_rate_chg_12m": "12-month change in mortgage rate",
    "zhvi_rel_state": "value vs. state median",
    "zhvi_rel_metro": "value vs. metro median",
    "price_to_income": "price-to-income ratio",
    "log_zhvi": "home value level (log)",
}


def _label(f: str) -> str:
    return FRIENDLY.get(f, f.replace("_", " "))


def predict(zip_code: str, origin: str):
    zip_code = (zip_code or "").strip().zfill(5)
    origin_ts = pd.Timestamp(origin)
    row = FEATURES[(FEATURES["zip"] == zip_code) & (FEATURES["month"] == origin_ts)]
    if row.empty:
        return f"No data for ZIP {zip_code} at {origin}.", None, None, ""
    X = encode_categoricals(row, bundle["categorical"])[bundle["features"]]
    pred = float(bundle["model"].predict(X)[0])
    sv = np.asarray(EXPLAINER.shap_values(X))[0]
    base = float(np.asarray(EXPLAINER.expected_value).ravel()[0])
    order = np.argsort(-np.abs(sv))[:8]
    contrib = pd.DataFrame(
        {
            "feature": [_label(bundle["features"][i]) for i in order],
            "value": [X.iloc[0, i] for i in order],
            "contribution (pp)": [round(float(sv[i]), 2) for i in order],
        }
    )
    m = ZIP_META.loc[zip_code] if zip_code in ZIP_META.index else None
    place = f"{m['city']}, {m['state']} ({m['metro']})" if m is not None else zip_code
    actual = row["target_growth_12m"].iloc[0]
    actual_txt = (
        f"Actual 12-month growth after {origin[:7]}: **{actual:+.1f}%**"
        if pd.notna(actual)
        else "Actual outcome not yet observed."
    )
    summary = (
        f"### {zip_code} — {place}\n"
        f"ZHVI at origin ({origin[:7]}): **${row['zhvi'].iloc[0]:,.0f}**\n\n"
        f"Model prediction for the next 12 months: **{pred:+.1f}%** "
        f"(typical error on the holdout ≈ ±{HOLDOUT_MAE:.1f} pp; model average = {base:+.1f}%)\n\n{actual_txt}"
    )
    h = HISTORY[HISTORY["zip"] == zip_code].sort_values("month")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(h["month"], h["zhvi"], lw=1.6, label="ZHVI (raw, mid-tier)")
    ax.axvline(origin_ts, color="grey", ls="--", lw=1, label="prediction origin")
    z0 = float(row["zhvi"].iloc[0])
    ax.plot(
        [origin_ts, origin_ts + pd.DateOffset(months=12)],
        [z0, z0 * (1 + pred / 100)],
        color="tab:red",
        lw=2,
        label=f"predicted 12m path ({pred:+.1f}%)",
    )
    ax.set_title(f"{zip_code} home value history")
    ax.set_ylabel("USD")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig2, ax2 = plt.subplots(figsize=(8, 3.6))
    colors = ["tab:green" if c > 0 else "tab:red" for c in contrib["contribution (pp)"]]
    ax2.barh(contrib["feature"][::-1], contrib["contribution (pp)"][::-1], color=colors[::-1])
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("SHAP contribution to predicted growth (pp)")
    ax2.set_title("Why this prediction? (top 8 features)")
    fig2.tight_layout()
    return summary, fig, fig2, contrib


with gr.Blocks(title="HomeSignal") as demo:
    gr.Markdown("# HomeSignal — 12-month ZIP-level home value growth (educational demo)")
    gr.Markdown(DISCLAIMER)
    with gr.Row():
        zip_in = gr.Textbox(label="ZIP code", value="78702")
        origin_in = gr.Dropdown(
            label="Prediction origin (month)",
            choices=[str(o.date()) for o in ORIGINS],
            value=str(ORIGINS[-1].date()),
        )
        btn = gr.Button("Predict", variant="primary")
    summary_out = gr.Markdown()
    with gr.Row():
        hist_plot = gr.Plot(label="History")
        shap_plot = gr.Plot(label="Explanation")
    table_out = gr.Dataframe(label="Top contributing features")
    btn.click(predict, [zip_in, origin_in], [summary_out, hist_plot, shap_plot, table_out])
    gr.Markdown(
        f"Model: {META['model']} trained on origins {META['training_origins'][0]} → {META['training_origins'][1]}; "
        f"holdout MAE {HOLDOUT_MAE:.2f} pp, within-origin Spearman {META['holdout_metrics']['final_model']['cs_spearman']:.2f}. "
        "Source and model card: github.com/Gariyuuu/homesignal"
    )

if __name__ == "__main__":
    demo.launch()
