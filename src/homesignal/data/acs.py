"""Census ACS 5-year estimates for ZCTAs, pulled from the key-free Census FTP summary files.

Two file formats are supported:

* **Legacy sequence files** (vintages 2011–2020): the ``UnitedStates`` folder of
  ``5_year_seq_by_state`` contains every ZCTA. A lookup file maps each table to a sequence
  number and a start column; the geography file maps ``LOGRECNO`` to a GEOID.
* **Table-based files** (vintages 2021+): one pipe-delimited file per table keyed by GEOID.

Each vintage is cached as ``data/interim/acs/acs_<vintage>.parquet`` with columns named
``<TABLE>_<LINE>`` (e.g. ``B19013_001``) plus ``zcta`` and ``vintage``.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from pathlib import Path

import pandas as pd

from homesignal.config import DataConfig
from homesignal.data.download import fetch

log = logging.getLogger(__name__)

# Cells (line numbers) needed from each table. Documented in docs/feature_dictionary.md.
TABLE_CELLS: dict[str, list[int]] = {
    "B01001": [
        1,
        3,
        4,
        5,
        6,
        11,
        12,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        29,
        30,
        35,
        36,
        44,
        45,
        46,
        47,
        48,
        49,
    ],
    "B01002": [1],
    "B01003": [1],
    "B15002": [1, 15, 16, 17, 18, 32, 33, 34, 35],
    "B15003": [1, 22, 23, 24, 25],
    "B19013": [1],
    "B23025": [1, 2, 3, 5],
    "B25001": [1],
    "B25002": [1, 2, 3],
    "B25003": [1, 2, 3],
    "B25064": [1],
    "B25077": [1],
}

ACS_SENTINEL_THRESHOLD = -200_000_000

LEGACY_LOOKUP_PATHS = [
    "documentation/5_year/user_tools/Sequence_Number_and_Table_Number_Lookup.txt",
    "documentation/user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt",
]


def cell_name(table: str, line: int) -> str:
    """Column name for a table cell, e.g. ``B19013_001``."""
    return f"{table}_{line:03d}"


# ----------------------------------------------------------------------------- legacy format


def parse_lookup(text: str, tables: list[str]) -> dict[str, tuple[int, int]]:
    """Parse a legacy lookup file into ``{table: (sequence, start_position)}``.

    The format changed slightly over the years (quoted fields, ``.`` for missing), so the
    parser only relies on column order: file id, table id, sequence, line, start position.
    """
    rdr = csv.reader(io.StringIO(text))
    next(rdr)
    out: dict[str, tuple[int, int]] = {}
    for row in rdr:
        if len(row) < 5:
            continue
        table, seq, line, start = (c.strip() for c in row[1:5])
        if (
            table in tables
            and line in ("", ".")
            and re.fullmatch(r"\d+", start)
            and table not in out
        ):
            out[table] = (int(seq), int(start))
    missing = [t for t in tables if t not in out]
    if missing:
        log.info("lookup: tables not present in this vintage: %s", missing)
    return out


def _legacy_lookup(cfg: DataConfig, vintage: int) -> dict[str, tuple[int, int]]:
    dest = cfg.raw_dir / "acs" / str(vintage) / "lookup.txt"
    if not dest.exists():
        last: Exception | None = None
        for rel in LEGACY_LOOKUP_PATHS:
            try:
                fetch(
                    f"{cfg.acs.base_url}/{vintage}/{rel}",
                    dest,
                    user_agent=cfg.user_agent,
                    retries=1,
                )
                break
            except RuntimeError as err:
                last = err
        else:
            raise RuntimeError(f"no lookup file for ACS {vintage}") from last
    return parse_lookup(dest.read_text(encoding="latin-1"), cfg.acs.tables)


def _legacy_geo(cfg: DataConfig, vintage: int) -> pd.DataFrame:
    """Return ``LOGRECNO -> zcta`` for the ZCTA (summary level 860) rows of a vintage."""
    url = (
        f"{cfg.acs.base_url}/{vintage}/data/5_year_seq_by_state/UnitedStates/"
        f"All_Geographies_Not_Tracts_Block_Groups/g{vintage}5us.csv"
    )
    dest = fetch(
        url, cfg.raw_dir / "acs" / str(vintage) / f"g{vintage}5us.csv", user_agent=cfg.user_agent
    )
    rows = list(csv.reader(dest.open(encoding="latin-1")))
    first = rows[0]
    geoid_col = next(i for i, v in enumerate(first) if re.fullmatch(r"\d{5}US\w*", v))
    zcta_rows = [(r[4], r[geoid_col]) for r in rows if r[2] == "860"]
    geo = pd.DataFrame(zcta_rows, columns=["logrecno", "geoid"])
    if not geo["geoid"].str.fullmatch(r"86000US\d{5}").all():
        raise ValueError(f"unexpected ZCTA GEOID format in ACS {vintage}")
    geo["zcta"] = geo["geoid"].str[-5:]
    return geo[["logrecno", "zcta"]]


def _load_legacy_vintage(cfg: DataConfig, vintage: int) -> pd.DataFrame:
    lookup = _legacy_lookup(cfg, vintage)
    geo = _legacy_geo(cfg, vintage)
    by_seq: dict[int, list[tuple[str, int]]] = {}
    for table, (seq, start) in lookup.items():
        by_seq.setdefault(seq, []).append((table, start))
    frames = [geo.set_index("logrecno")]
    for seq, specs in sorted(by_seq.items()):
        name = f"{vintage}5us{seq:04d}000"
        url = (
            f"{cfg.acs.base_url}/{vintage}/data/5_year_seq_by_state/UnitedStates/"
            f"All_Geographies_Not_Tracts_Block_Groups/{name}.zip"
        )
        dest = fetch(
            url, cfg.raw_dir / "acs" / str(vintage) / f"{name}.zip", user_agent=cfg.user_agent
        )
        cols: dict[int, str] = {5: "logrecno"}
        for table, start in specs:
            for line in TABLE_CELLS[table]:
                cols[start + line - 2] = cell_name(table, line)
        with zipfile.ZipFile(dest) as zf, zf.open(f"e{name}.txt") as fh:
            part = pd.read_csv(
                fh,
                header=None,
                usecols=sorted(cols),
                dtype=str,
                na_values=[".", ""],
                keep_default_na=False,
            )
        part = part.rename(columns=cols).set_index("logrecno")
        frames.append(part.loc[part.index.intersection(frames[0].index)])
    df = pd.concat(frames, axis=1).reindex(frames[0].index).reset_index(drop=True)
    return df


# ------------------------------------------------------------------------ table-based format


def _load_table_based_vintage(cfg: DataConfig, vintage: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for table in cfg.acs.tables:
        url = (
            f"{cfg.acs.base_url}/{vintage}/table-based-SF/data/5YRData/"
            f"acsdt5y{vintage}-{table.lower()}.dat"
        )
        dest = cfg.raw_dir / "acs" / str(vintage) / f"{table}.dat"
        try:
            fetch(url, dest, user_agent=cfg.user_agent, retries=2)
        except RuntimeError:
            log.info(
                "ACS %d: table %s not available in table-based format, skipping", vintage, table
            )
            continue
        want = {f"{table}_E{line:03d}": cell_name(table, line) for line in TABLE_CELLS[table]}
        part = pd.read_csv(
            dest,
            sep="|",
            dtype=str,
            usecols=["GEO_ID", *want],
            na_values=[".", ""],
            keep_default_na=False,
        )
        # ZCTA GEOIDs are "8600000US<zcta>" (2010 geography) or "860Z200US<zcta>" (2020 geography).
        part = part[part["GEO_ID"].str.match(r"^860(0000|Z200)US\d{5}$")].rename(columns=want)
        part["zcta"] = part["GEO_ID"].str[-5:]
        frames.append(part.drop(columns="GEO_ID").set_index("zcta"))
    return pd.concat(frames, axis=1).reset_index()


# -------------------------------------------------------------------------------- public API


def load_acs_vintage(cfg: DataConfig, vintage: int) -> pd.DataFrame:
    """Load one vintage (cached as parquet). Values are floats; ``zcta`` is a 5-char string."""
    cache = cfg.interim_dir / "acs" / f"acs_{vintage}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if vintage in cfg.acs.table_based_vintages:
        df = _load_table_based_vintage(cfg, vintage)
    elif vintage in cfg.acs.legacy_vintages:
        df = _load_legacy_vintage(cfg, vintage)
    else:
        raise ValueError(f"vintage {vintage} not configured")
    value_cols = [c for c in df.columns if c != "zcta"]
    for c in value_cols:
        numeric = pd.to_numeric(df[c], errors="coerce")
        # Table-based files encode suppressed/not-applicable cells as -222222222 … -999999999.
        df[c] = numeric.where(numeric > ACS_SENTINEL_THRESHOLD)
    df["vintage"] = vintage
    df = df[["zcta", "vintage", *sorted(value_cols)]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    log.info("ACS %d: %d ZCTAs, %d cells", vintage, len(df), len(value_cols))
    return df


def load_acs(cfg: DataConfig) -> pd.DataFrame:
    """Load all configured vintages stacked into one long frame."""
    return pd.concat([load_acs_vintage(cfg, v) for v in cfg.acs.vintages], ignore_index=True)


def vintage_available_from(cfg: DataConfig, vintage: int) -> pd.Timestamp:
    """First month in which ``vintage`` may be used as a feature (release month + 1)."""
    override = cfg.acs.availability_overrides.get(vintage)
    if override:
        return pd.Timestamp(override)
    return pd.Timestamp(year=vintage, month=1, day=1) + pd.DateOffset(
        months=12 + cfg.acs.default_availability_lag_months - 1
    )


def derive_acs_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw table cells into named demographic variables (one row per zcta × vintage)."""
    g = raw.get

    def s(cols: list[str]) -> pd.Series:
        present = [c for c in cols if c in raw.columns]
        if not present:
            return pd.Series(float("nan"), index=raw.index)
        return raw[present].sum(axis=1, min_count=len(present))

    pop = raw["B01001_001"]
    under18 = s([cell_name("B01001", i) for i in (3, 4, 5, 6, 27, 28, 29, 30)])
    age_25_34 = s([cell_name("B01001", i) for i in (11, 12, 35, 36)])
    age_65p = s([cell_name("B01001", i) for i in (20, 21, 22, 23, 24, 25, 44, 45, 46, 47, 48, 49)])
    if "B15003_001" in raw.columns:
        bach_total = raw["B15003_001"]
        bach = s([cell_name("B15003", i) for i in (22, 23, 24, 25)])
    else:
        bach_total = raw["B15002_001"]
        bach = s([cell_name("B15002", i) for i in (15, 16, 17, 18, 32, 33, 34, 35)])
    out = pd.DataFrame(
        {
            "zcta": raw["zcta"],
            "vintage": raw["vintage"],
            "acs_population": raw["B01003_001"],
            "acs_median_age": raw["B01002_001"],
            "acs_share_under18": under18 / pop,
            "acs_share_25_34": age_25_34 / pop,
            "acs_share_65plus": age_65p / pop,
            "acs_bachelor_share": bach / bach_total,
            "acs_median_income": raw["B19013_001"],
            "acs_housing_units": raw["B25001_001"],
            "acs_vacancy_rate": raw["B25002_003"] / raw["B25002_001"],
            "acs_renter_share": raw["B25003_003"] / raw["B25003_001"],
            "acs_median_gross_rent": raw["B25064_001"],
            "acs_median_home_value": raw["B25077_001"],
        }
    )
    if g("B23025_001") is not None:
        out["acs_unemployment_rate"] = raw["B23025_005"] / raw["B23025_003"]
        out["acs_lfpr"] = raw["B23025_002"] / raw["B23025_001"]
    else:
        out["acs_unemployment_rate"] = float("nan")
        out["acs_lfpr"] = float("nan")
    return out


def acs_cache_paths(cfg: DataConfig) -> list[Path]:
    """Interim parquet paths for all vintages (for provenance reporting)."""
    return [cfg.interim_dir / "acs" / f"acs_{v}.parquet" for v in cfg.acs.vintages]
