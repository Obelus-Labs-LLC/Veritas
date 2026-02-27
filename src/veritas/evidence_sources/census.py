"""US Census Bureau — verify demographic and population claims.

Uses the free Census API (no key required for basic queries, ~500/day limit).
Covers: population, demographics, income, poverty, education attainment.

Docs: https://www.census.gov/data/developers.html
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from .base import rate_limited_get


_BASE_URL = "https://api.census.gov/data"

# Map keywords to Census API datasets and variables
_CENSUS_QUERIES = {
    "population": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B01003_001E",  # Total population
        "description": "Total Population (ACS 1-Year Estimates)",
    },
    "median income": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B19013_001E",  # Median household income
        "description": "Median Household Income (ACS 1-Year Estimates)",
    },
    "household income": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B19013_001E",
        "description": "Median Household Income (ACS 1-Year Estimates)",
    },
    "poverty": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B17001_002E",  # Below poverty level
        "description": "Population Below Poverty Level (ACS 1-Year Estimates)",
    },
    "poverty rate": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B17001_002E",
        "description": "Population Below Poverty Level (ACS 1-Year Estimates)",
    },
    "education": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B15003_022E",  # Bachelor's degree
        "description": "Bachelor's Degree or Higher (ACS 1-Year Estimates)",
    },
    "bachelor": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B15003_022E",
        "description": "Bachelor's Degree Attainment (ACS 1-Year Estimates)",
    },
    "college": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B15003_022E",
        "description": "Educational Attainment (ACS 1-Year Estimates)",
    },
    "uninsured": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B27010_001E",  # Health insurance
        "description": "Health Insurance Coverage Status (ACS 1-Year Estimates)",
    },
    "health insurance": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B27010_001E",
        "description": "Health Insurance Coverage Status (ACS 1-Year Estimates)",
    },
    "homeownership": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B25003_002E",  # Owner occupied
        "description": "Owner-Occupied Housing Units (ACS 1-Year Estimates)",
    },
    "rent": {
        "dataset": "2022/acs/acs1",
        "variables": "NAME,B25064_001E",  # Median gross rent
        "description": "Median Gross Rent (ACS 1-Year Estimates)",
    },
}


def _match_query(claim_text: str) -> Optional[Dict[str, str]]:
    """Match claim text to a Census API query."""
    lower = claim_text.lower()
    for term in sorted(_CENSUS_QUERIES.keys(), key=len, reverse=True):
        if term in lower:
            return _CENSUS_QUERIES[term]
    return None


def search_census(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Census Bureau for demographic data matching a claim.

    Standard evidence source signature.
    """
    query_config = _match_query(claim_text)
    if not query_config:
        return []

    dataset = query_config["dataset"]
    variables = query_config["variables"]
    description = query_config["description"]

    # Fetch national-level data (for=us:*)
    resp = rate_limited_get(
        f"{_BASE_URL}/{dataset}",
        source_name="census",
        params={
            "get": variables,
            "for": "us:*",
        },
        timeout=15.0,
    )

    snippet = f"{description}. Source: U.S. Census Bureau."

    if resp is not None:
        try:
            data = resp.json()
            # Census API returns array of arrays: [headers, values...]
            if isinstance(data, list) and len(data) >= 2:
                headers = data[0]
                values = data[1]
                # Build human-readable snippet
                pairs = []
                for h, v in zip(headers, values):
                    if h != "us" and v and v != "null":
                        try:
                            num = int(v)
                            pairs.append(f"{h}: {num:,}")
                        except ValueError:
                            pairs.append(f"{h}: {v}")
                if pairs:
                    snippet += " National totals: " + "; ".join(pairs) + "."
        except Exception:
            pass

    # Also try state-level if claim mentions a state
    state_data_snippet = ""
    state_match = _extract_state(claim_text)
    if state_match:
        resp2 = rate_limited_get(
            f"{_BASE_URL}/{dataset}",
            source_name="census",
            params={
                "get": variables,
                "for": f"state:{state_match}",
            },
            timeout=10.0,
        )
        if resp2 is not None:
            try:
                sdata = resp2.json()
                if isinstance(sdata, list) and len(sdata) >= 2:
                    headers = sdata[0]
                    values = sdata[1]
                    pairs = []
                    for h, v in zip(headers, values):
                        if h not in ("state",) and v and v != "null":
                            try:
                                num = int(v)
                                pairs.append(f"{h}: {num:,}")
                            except ValueError:
                                pairs.append(f"{h}: {v}")
                    if pairs:
                        state_data_snippet = " State data: " + "; ".join(pairs) + "."
            except Exception:
                pass

    results = [{
        "url": f"https://data.census.gov/table?q={description.replace(' ', '+')}",
        "title": f"Census Bureau: {description}",
        "source_name": "census",
        "evidence_type": "gov",
        "snippet": (snippet + state_data_snippet)[:2000],
    }]

    return results[:max_results]


# Simple state FIPS code lookup
_STATE_FIPS = {
    "alabama": "01", "alaska": "02", "arizona": "04", "arkansas": "05",
    "california": "06", "colorado": "08", "connecticut": "09",
    "delaware": "10", "florida": "12", "georgia": "13", "hawaii": "15",
    "idaho": "16", "illinois": "17", "indiana": "18", "iowa": "19",
    "kansas": "20", "kentucky": "21", "louisiana": "22", "maine": "23",
    "maryland": "24", "massachusetts": "25", "michigan": "26",
    "minnesota": "27", "mississippi": "28", "missouri": "29",
    "montana": "30", "nebraska": "31", "nevada": "32",
    "new hampshire": "33", "new jersey": "34", "new mexico": "35",
    "new york": "36", "north carolina": "37", "north dakota": "38",
    "ohio": "39", "oklahoma": "40", "oregon": "41", "pennsylvania": "42",
    "rhode island": "44", "south carolina": "45", "south dakota": "46",
    "tennessee": "47", "texas": "48", "utah": "49", "vermont": "50",
    "virginia": "51", "washington": "53", "west virginia": "54",
    "wisconsin": "55", "wyoming": "56",
}


def _extract_state(claim_text: str) -> Optional[str]:
    """Extract a US state FIPS code from claim text."""
    lower = claim_text.lower()
    for state_name, fips in _STATE_FIPS.items():
        if state_name in lower:
            return fips
    return None
