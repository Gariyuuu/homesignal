"""Bundle what the demo needs into ``app/data`` (run after `make train`).

What is bundled (derived artifacts only, no raw source files):
* the final model + metadata,
* feature rows for every ZIP at the last 8 quarterly origins (so a user can pick an origin
  whose outcome is already known and compare prediction vs. actual),
* a compact per-ZIP ZHVI history (monthly index values since 2010) used for the history chart.

ZHVI values are Zillow Research data (see docs/sources.md). The Hugging Face Space ships the
bundle produced here; if you deploy your own copy, keep the Zillow attribution in the UI.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from homesignal.config import load_data_config, load_evaluation_config
from homesignal.evaluation.splits import is_origin_month
from homesignal.features.build import load_panel
from homesignal.models.train import METADATA_PATH, MODEL_PATH

APP_DATA = Path(__file__).resolve().parent / "data"
N_ORIGINS = 8


def main() -> None:
    """Write the app bundle."""
    APP_DATA.mkdir(exist_ok=True)
    data_cfg = load_data_config()
    eval_cfg = load_evaluation_config()
    panel = load_panel(data_cfg.processed_dir)
    origins = sorted(panel.loc[is_origin_month(panel["month"], eval_cfg), "month"].unique())[
        -N_ORIGINS:
    ]
    rows = panel[panel["month"].isin(origins)].reset_index(drop=True)
    rows.to_parquet(APP_DATA / "features.parquet", index=False)
    history = panel.loc[panel["month"] >= "2010-01-01", ["zip", "month", "zhvi"]]
    history.to_parquet(APP_DATA / "history.parquet", index=False)
    meta = pd.read_csv(
        data_cfg.raw_dir / "zillow" / "zhvi_zip.csv",
        dtype={"RegionName": str},
        usecols=["RegionName", "State", "City", "Metro", "CountyName"],
    )
    meta = meta.rename(
        columns={
            "RegionName": "zip",
            "State": "state",
            "City": "city",
            "Metro": "metro",
            "CountyName": "county",
        }
    )
    meta["zip"] = meta["zip"].str.zfill(5)
    meta.to_parquet(APP_DATA / "zip_meta.parquet", index=False)
    shutil.copy(MODEL_PATH, APP_DATA / MODEL_PATH.name)
    shutil.copy(METADATA_PATH, APP_DATA / METADATA_PATH.name)
    print(
        f"bundled {len(rows):,} feature rows at origins {[str(o.date()) for o in origins]} and {len(history):,} history rows"
    )


if __name__ == "__main__":
    main()
