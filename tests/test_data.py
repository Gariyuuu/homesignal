import numpy as np
import pandas as pd

from homesignal.data.acs import derive_acs_features, parse_lookup, vintage_available_from
from homesignal.data.crosswalks import (
    STATE_FIPS_TO_ABBR,
    STATE_TO_DIVISION,
    STATE_TO_REGION,
    zip_to_zcta,
)
from homesignal.features.demographics import acs_feature_table, vintage_as_of
from homesignal.features.geography import geography_table
from homesignal.features.macro import _lagged
from tests.conftest import make_acs_raw, make_meta

LOOKUP_2019 = """File ID,Table ID,Sequence Number,Line Number,Start Position,Total Cells in Table,Total Cells in Sequence,Table Title,Subject Area
ACSSF,B01001,0002,,7,49 CELLS,,SEX BY AGE,Age-Sex
ACSSF,B01001,0002,,,,,Universe:  Total population,
ACSSF,B01001,0002,1,,,,Total:,
ACSSF,B19013,0058,,177,1 CELL,,MEDIAN HOUSEHOLD INCOME,Income
"""
LOOKUP_2011 = """"fileid","Table ID","seq","Line Number Decimal M Lines","position","cells","total","Long Table Title","subject_area"
"ACSSF","B01001","0002",".","7","49 CELLS",".","SEX BY AGE","Age-Sex"
"ACSSF","B01001","0002",".",".","",".","Universe:  Total population",""
"ACSSF","B19013","0058",".","177","1 CELL",".","MEDIAN HOUSEHOLD INCOME","Income"
"""


def test_parse_lookup_both_formats():
    for text in (LOOKUP_2019, LOOKUP_2011):
        out = parse_lookup(text, ["B01001", "B19013", "B99999"])
        assert out == {"B01001": (2, 7), "B19013": (58, 177)}


def test_vintage_availability_rule(data_cfg):
    assert vintage_available_from(data_cfg, 2017) == pd.Timestamp("2019-01-01")
    assert vintage_available_from(data_cfg, 2020) == pd.Timestamp(
        "2022-04-01"
    )  # COVID delay override
    assert vintage_available_from(data_cfg, 2024) == pd.Timestamp("2026-02-01")


def test_vintage_as_of_picks_newest_available(data_cfg):
    months = pd.DatetimeIndex(
        ["2012-12-01", "2013-01-01", "2022-03-01", "2022-04-01", "2026-03-01"]
    )
    v = vintage_as_of(data_cfg, months)
    assert np.isnan(v.iloc[0])
    assert v.iloc[1] == 2011
    assert v.iloc[2] == 2019  # 2020 vintage not yet released
    assert v.iloc[3] == 2020
    assert v.iloc[4] == 2024


def test_derive_acs_features_ratios():
    raw = make_acs_raw([2019]).iloc[:1].copy()
    raw["B25003_001"], raw["B25003_003"] = 100.0, 40.0
    raw["B25002_001"], raw["B25002_003"] = 200.0, 20.0
    f = derive_acs_features(raw)
    assert np.isclose(f["acs_renter_share"].iloc[0], 0.4)
    assert np.isclose(f["acs_vacancy_rate"].iloc[0], 0.1)
    assert 0 <= f["acs_bachelor_share"].iloc[0] or np.isnan(f["acs_bachelor_share"].iloc[0])


def test_acs_growth_only_between_consecutive_vintages():
    raw = make_acs_raw([2015, 2016, 2018])
    f = acs_feature_table(raw).set_index(["zcta", "vintage"])
    z = f.index.get_level_values(0)[0]
    assert np.isnan(f.loc[(z, 2015), "acs_income_growth"])
    assert np.isfinite(f.loc[(z, 2016), "acs_income_growth"])
    assert np.isnan(f.loc[(z, 2018), "acs_income_growth"])  # 2017 missing → no growth


def test_zip_to_zcta_direct_match_and_coverage():
    mapped, stats = zip_to_zcta(pd.Index(["94110", "00000", "78702"]), {"94110", "78702"})
    assert mapped["94110"] == "94110" and mapped["00000"] is None
    assert stats["n_matched"] == 2 and np.isclose(stats["match_rate"], 2 / 3)


def test_state_tables_complete():
    assert len(STATE_FIPS_TO_ABBR) == 51
    assert set(STATE_FIPS_TO_ABBR.values()) == set(STATE_TO_REGION) == set(STATE_TO_DIVISION)
    assert all(len(k) == 2 for k in STATE_FIPS_TO_ABBR)  # zero-padded FIPS strings


def test_geography_table_buckets():
    meta = make_meta()
    meta.loc[meta.index[:3], "metro"] = "Big metro"
    meta.loc[meta.index[3], "metro"] = None
    geo = geography_table(meta, [2, 5, 10])
    assert geo.loc[meta.index[0], "metro_size_bucket"] == "mid"
    assert geo.loc[meta.index[3], "metro_size_bucket"] == "non_metro"
    assert geo.loc[meta.index[4], "metro_size_bucket"] == "small"
    assert geo["region"].notna().all()


def test_lagged_shifts_forward():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="MS"))
    out = _lagged(s, 2)
    assert out.index[0] == pd.Timestamp("2020-03-01") and out.iloc[0] == 1.0
