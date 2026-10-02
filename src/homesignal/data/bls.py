"""BLS Public Data API client (v1: no key, 25 queries/day, 25 series and 10 years per query).

Responses are cached as JSON in ``data/raw/bls`` so a rebuild makes no new requests. If the
optional ``BLS_API_KEY`` environment variable is set, the v2 endpoint (50 series, 20 years,
500 queries/day) is used instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date
from pathlib import Path

import pandas as pd

from homesignal.config import DataConfig
from homesignal.data.crosswalks import STATE_FIPS_TO_ABBR
from homesignal.data.download import fetch

log = logging.getLogger(__name__)


def _chunks(items: list[str], n: int) -> list[list[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _fetch_series(
    cfg: DataConfig, series: list[str], start: int, end: int
) -> list[dict[str, object]]:
    key = os.environ.get("BLS_API_KEY")
    url = cfg.bls.api_url.replace("/v1/", "/v2/") if key else cfg.bls.api_url
    body: dict[str, object] = {"seriesid": series, "startyear": str(start), "endyear": str(end)}
    if key:
        body["registrationkey"] = key
    digest = hashlib.sha1(json.dumps([series, start, end]).encode()).hexdigest()[:12]
    dest = cfg.raw_dir / "bls" / f"bls_{start}_{end}_{digest}.json"
    fetch(url, dest, user_agent=cfg.user_agent, method="POST", json_body=body, retries=3)
    payload = json.loads(dest.read_text())
    if payload.get("status") != "REQUEST_SUCCEEDED":
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"BLS request failed: {payload.get('message')}")
    result: list[dict[str, object]] = payload["Results"]["series"]
    return result


def _parse(series_payload: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for s in series_payload:
        sid = str(s["seriesID"])
        for d in s["data"]:  # type: ignore[attr-defined]
            period = str(d["period"])
            if not period.startswith("M") or period == "M13":
                continue
            try:
                value = float(str(d["value"]))
            except ValueError:  # BLS uses "-" for suppressed/missing values
                value = float("nan")
            rows.append(
                (sid, pd.Timestamp(year=int(d["year"]), month=int(period[1:]), day=1), value)
            )
    return pd.DataFrame(rows, columns=["series_id", "month", "value"])


def load_bls(cfg: DataConfig) -> pd.DataFrame:
    """Return a long frame ``series_id, month, value`` for all configured series."""
    cache = cfg.interim_dir / "bls.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    series = list(cfg.bls.national_series)
    if cfg.bls.state_unemployment:
        series += [f"LASST{fips}0000000000003" for fips in STATE_FIPS_TO_ABBR]
    per_query, span = (50, 20) if os.environ.get("BLS_API_KEY") else (25, 10)
    this_year = date.today().year
    windows = [
        (y, min(y + span - 1, this_year)) for y in range(cfg.bls.start_year, this_year + 1, span)
    ]
    frames = []
    for start, end in windows:
        for chunk in _chunks(series, per_query):
            frames.append(_parse(_fetch_series(cfg, chunk, start, end)))
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["series_id", "month"])
        .sort_values(["series_id", "month"])
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    log.info(
        "BLS: %d series, %s → %s",
        df["series_id"].nunique(),
        df["month"].min().date(),
        df["month"].max().date(),
    )
    return df


def bls_cache_path(cfg: DataConfig) -> Path:
    """Interim parquet path."""
    return cfg.interim_dir / "bls.parquet"
