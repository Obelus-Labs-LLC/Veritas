"""Bureau of Labor Statistics (BLS) — verify labor/economic claims.

Uses BLS Public Data API v1 (no key required).
Covers: unemployment, CPI, wages, employment by sector.
Rate limit: 25 series/request, daily quota unspecified.

Docs: https://www.bls.gov/developers/home.htm
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from .base import rate_limited_get


_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"

# Well-known BLS series IDs mapped to keywords
_SERIES_MAP: Dict[str, tuple[str, str]] = {
    # (series_id, description)
    "unemployment rate": ("LNS14000000", "Unemployment Rate, Seasonally Adjusted"),
    "unemployment": ("LNS14000000", "Unemployment Rate, Seasonally Adjusted"),
    "nonfarm payroll": ("CES0000000001", "Total Nonfarm Employment, Seasonally Adjusted"),
    "payrolls": ("CES0000000001", "Total Nonfarm Employment, Seasonally Adjusted"),
    "jobs": ("CES0000000001", "Total Nonfarm Employment, Seasonally Adjusted"),
    "employment": ("CES0000000001", "Total Nonfarm Employment, Seasonally Adjusted"),
    "cpi": ("CUUR0000SA0", "Consumer Price Index, All Urban Consumers, U.S. City Average"),
    "consumer price": ("CUUR0000SA0", "Consumer Price Index, All Urban Consumers"),
    "inflation": ("CUUR0000SA0", "Consumer Price Index, All Urban Consumers"),
    "wages": ("CES0500000003", "Average Hourly Earnings, Private Sector"),
    "hourly earnings": ("CES0500000003", "Average Hourly Earnings, Private Sector"),
    "average wage": ("CES0500000003", "Average Hourly Earnings, Private Sector"),
    "labor force": ("LNS11000000", "Civilian Labor Force Level"),
    "participation rate": ("LNS11300000", "Labor Force Participation Rate"),
    "job openings": ("JTS000000000000000JOL", "Job Openings, Total Nonfarm"),
    "quit rate": ("JTS000000000000000QUR", "Quits Rate, Total Nonfarm"),
    "producer price": ("WPUFD4", "Producer Price Index, Final Demand"),
    "ppi": ("WPUFD4", "Producer Price Index, Final Demand"),
}


def _match_series(claim_text: str) -> Optional[tuple[str, str]]:
    """Match claim text to a known BLS series."""
    lower = claim_text.lower()
    for term in sorted(_SERIES_MAP.keys(), key=len, reverse=True):
        if term in lower:
            return _SERIES_MAP[term]
    return None


def search_bls(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search BLS for labor/economic data matching a claim.

    Standard evidence source signature.
    """
    match = _match_series(claim_text)
    if not match:
        return []

    series_id, description = match

    # Fetch recent data from BLS API (v1, no key)
    resp = rate_limited_get(
        f"{_BASE_URL}/{series_id}",
        source_name="bls",
        timeout=15.0,
    )

    snippet = f"{description}. Source: Bureau of Labor Statistics. Series: {series_id}."

    if resp is not None:
        try:
            data = resp.json()
            status = data.get("status", "")
            if status == "REQUEST_SUCCEEDED":
                series_data = data.get("Results", {}).get("series", [])
                if series_data:
                    obs = series_data[0].get("data", [])
                    # Build snippet with recent values
                    values = []
                    for o in obs[:8]:  # Last 8 periods
                        year = o.get("year", "")
                        period = o.get("periodName", "")
                        value = o.get("value", "")
                        if year and value:
                            values.append(f"{period} {year}: {value}")
                    if values:
                        snippet += " Recent values: " + "; ".join(values) + "."
        except Exception:
            pass

    results = [{
        "url": f"https://data.bls.gov/timeseries/{series_id}",
        "title": f"BLS: {description}",
        "source_name": "bls",
        "evidence_type": "gov",
        "snippet": snippet[:2000],
    }]

    return results[:max_results]
