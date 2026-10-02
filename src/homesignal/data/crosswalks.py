"""Geography lookups: state FIPS/abbreviations, Census regions, and the ZIP → ZCTA join.

ZIP codes (USPS delivery routes) and ZCTAs (Census areal approximations) are not identical.
There is no official USPS-ZIP → ZCTA crosswalk that is free of registration (HUD's requires
an API token), so this project joins Zillow ZIPs to ACS ZCTAs by **direct 5-digit match** and
reports the match rate. Unmatched ZIPs keep ``NaN`` demographics (never dropped, never imputed
with a neighbour's value). See docs/methodology.md for the limitations of this choice.
"""

from __future__ import annotations

import pandas as pd

STATE_FIPS_TO_ABBR: dict[str, str] = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}  # fmt: skip

STATE_ABBR_TO_FIPS: dict[str, str] = {v: k for k, v in STATE_FIPS_TO_ABBR.items()}

# Census Bureau regions and divisions.
_DIVISIONS: dict[str, list[str]] = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Middle Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
}
_REGIONS: dict[str, list[str]] = {
    "Northeast": ["New England", "Middle Atlantic"],
    "Midwest": ["East North Central", "West North Central"],
    "South": ["South Atlantic", "East South Central", "West South Central"],
    "West": ["Mountain", "Pacific"],
}
STATE_TO_DIVISION: dict[str, str] = {s: d for d, states in _DIVISIONS.items() for s in states}
STATE_TO_REGION: dict[str, str] = {
    s: r for r, divs in _REGIONS.items() for d in divs for s in _DIVISIONS[d]
}


def zip_to_zcta(zips: pd.Index, zctas: set[str]) -> tuple[pd.Series, dict[str, float]]:
    """Map ZIPs to ZCTAs by direct match.

    Returns a Series (index = zip, value = zcta or NaN) and coverage statistics.
    """
    mapped = pd.Series(
        [z if z in zctas else None for z in zips], index=zips, dtype="object", name="zcta"
    )
    n = len(zips)
    matched = int(mapped.notna().sum())
    stats = {
        "n_zips": float(n),
        "n_matched": float(matched),
        "match_rate": matched / n if n else 0.0,
    }
    return mapped, stats
