"""Coarse geography features: state, Census region/division, metro size bucket.

Raw identifiers (ZIP, metro name, county) are deliberately *not* features so the model cannot
memorise locations; the geography ablation in the evaluation report drops even these.
"""

from __future__ import annotations

import pandas as pd

from homesignal.data.crosswalks import STATE_TO_DIVISION, STATE_TO_REGION

GEO_FEATURES = ["state", "region", "division", "metro_size_bucket"]
BUCKET_LABELS = ["non_metro", "small", "mid", "large", "mega"]


def metro_size_bucket(meta: pd.DataFrame, thresholds: list[int]) -> pd.Series:
    """Bucket metros by the number of Zillow ZIPs they contain (a size proxy, documented)."""
    counts = meta["metro"].value_counts()
    labels = BUCKET_LABELS[1:]
    bins = [-1, *thresholds, float("inf")]
    bucket_by_metro = pd.cut(counts, bins=bins, labels=labels).astype("object")
    out = meta["metro"].map(bucket_by_metro)
    return out.fillna("non_metro").astype("string")


def geography_table(meta: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """One row per ZIP with the geography features."""
    return pd.DataFrame(
        {
            "state": meta["state"].astype("string"),
            "region": meta["state"].map(STATE_TO_REGION).astype("string"),
            "division": meta["state"].map(STATE_TO_DIVISION).astype("string"),
            "metro_size_bucket": metro_size_bucket(meta, thresholds),
        },
        index=meta.index,
    )
