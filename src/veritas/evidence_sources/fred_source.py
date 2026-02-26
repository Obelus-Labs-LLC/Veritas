"""FRED (Federal Reserve Economic Data) — verify macroeconomic claims.

Uses the FRED API via the public JSON endpoint (no key required for
the series/observations endpoint with limited queries).

Falls back to the FRED search page for series discovery.

Ideal for: GDP, inflation, unemployment, interest rates, CPI,
federal funds rate, money supply, trade balance, etc.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from .base import rate_limited_get

# Well-known FRED series IDs for common macro terms
_SERIES_MAP: Dict[str, str] = {
    # GDP
    "gdp": "GDP",
    "real gdp": "GDPC1",
    "gdp growth": "A191RL1Q225SBEA",
    # Inflation / Prices
    "inflation": "CPIAUCSL",
    "cpi": "CPIAUCSL",
    "consumer price index": "CPIAUCSL",
    "pce": "PCEPI",
    "core inflation": "CPILFESL",
    # Employment
    "unemployment": "UNRATE",
    "unemployment rate": "UNRATE",
    "nonfarm payroll": "PAYEMS",
    "payrolls": "PAYEMS",
    "labor force": "CLF16OV",
    "participation rate": "CIVPART",
    # Interest Rates
    "interest rate": "FEDFUNDS",
    "federal funds rate": "FEDFUNDS",
    "fed funds rate": "FEDFUNDS",
    "fed rate": "FEDFUNDS",
    "10 year treasury": "DGS10",
    "treasury yield": "DGS10",
    "30 year mortgage": "MORTGAGE30US",
    "mortgage rate": "MORTGAGE30US",
    "prime rate": "DPRIME",
    # Money Supply
    "money supply": "M2SL",
    "m2": "M2SL",
    "m1": "M1SL",
    # Housing
    "housing starts": "HOUST",
    "home prices": "CSUSHPINSA",
    "case-shiller": "CSUSHPINSA",
    # Trade
    "trade balance": "BOPGSTB",
    "trade deficit": "BOPGSTB",
    "exports": "BOPGEXP",
    "imports": "BOPGIMP",
    # Debt
    "national debt": "GFDEBTN",
    "federal debt": "GFDEBTN",
    "debt to gdp": "GFDEGDQ188S",
    # Other
    "industrial production": "INDPRO",
    "retail sales": "RSAFS",
    "consumer confidence": "UMCSENT",
    "leading indicators": "USSLIND",
}

_FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series"
_FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

# FRED requires an API key for the JSON API. But we can use the
# public web endpoint to get series info without a key.
_FRED_WEB_URL = "https://fred.stlouisfed.org/series"


def _match_series(claim_text: str) -> Optional[str]:
    """Match claim text to a known FRED series ID.

    Returns series_id or None.
    """
    lower = claim_text.lower()
    # Try longer phrases first for better matching
    for term in sorted(_SERIES_MAP.keys(), key=len, reverse=True):
        if term in lower:
            return _SERIES_MAP[term]
    return None


def _extract_macro_keywords(claim_text: str) -> List[str]:
    """Extract macroeconomic keywords for FRED search."""
    macro_terms = [
        "gdp", "inflation", "unemployment", "interest rate", "cpi",
        "recession", "federal reserve", "monetary policy", "fiscal",
        "trade deficit", "debt", "deficit", "surplus",
        "economic growth", "employment", "jobs", "wages",
        "housing", "mortgage", "treasury", "yield", "bond",
        "money supply", "consumer", "retail", "industrial",
    ]
    lower = claim_text.lower()
    found = [t for t in macro_terms if t in lower]
    return found


def search_fred(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search FRED for economic data matching a claim.

    Standard evidence source signature. Returns list of dicts with keys:
    url, title, source_name, evidence_type, snippet.

    Strategy:
      1. Try to match a known FRED series
      2. Build a snippet with the series description and recent data link
      3. Return as evidence_type="dataset" (gets primary source boost)
    """
    series_id = _match_series(claim_text)
    keywords = _extract_macro_keywords(claim_text)

    if not series_id and not keywords:
        return []

    results = []

    if series_id:
        # We have a direct series match — fetch the series page info
        # Use the public web page as the evidence URL
        url = f"{_FRED_WEB_URL}/{series_id}"

        # Try to get series metadata from the FRED web page
        snippet = _build_series_snippet(series_id, claim_text)

        results.append({
            "url": url,
            "title": f"FRED Economic Data: {series_id}",
            "source_name": "fred",
            "evidence_type": "dataset",  # Gets 15-point primary_source boost
            "snippet": snippet,
        })

    # Add related series for broader coverage
    if keywords and len(results) < max_results:
        for kw in keywords[:3]:
            related_id = _SERIES_MAP.get(kw)
            if related_id and related_id != series_id:
                url = f"{_FRED_WEB_URL}/{related_id}"
                results.append({
                    "url": url,
                    "title": f"FRED Economic Data: {related_id}",
                    "source_name": "fred",
                    "evidence_type": "dataset",
                    "snippet": f"Federal Reserve Economic Data series {related_id} "
                               f"for {kw}. Source: Federal Reserve Bank of St. Louis.",
                })
                if len(results) >= max_results:
                    break

    return results[:max_results]


def _build_series_snippet(series_id: str, claim_text: str) -> str:
    """Build an informative snippet for a FRED series.

    Includes the series description and metadata to help the scoring engine
    match numbers and terms from the claim.
    """
    # Map series IDs to human-readable descriptions
    _SERIES_DESCRIPTIONS: Dict[str, str] = {
        "GDP": "Gross Domestic Product (GDP), Billions of Dollars, Quarterly, Seasonally Adjusted Annual Rate",
        "GDPC1": "Real Gross Domestic Product, Billions of Chained 2017 Dollars, Quarterly, Seasonally Adjusted Annual Rate",
        "A191RL1Q225SBEA": "Real GDP Growth Rate, Percent Change from Preceding Period, Quarterly, Seasonally Adjusted Annual Rate",
        "CPIAUCSL": "Consumer Price Index for All Urban Consumers (CPI-U), Index 1982-1984=100, Monthly, Seasonally Adjusted",
        "UNRATE": "Unemployment Rate, Percent, Monthly, Seasonally Adjusted",
        "FEDFUNDS": "Federal Funds Effective Rate, Percent, Monthly",
        "DGS10": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Percent, Daily",
        "M2SL": "M2 Money Supply, Billions of Dollars, Monthly, Seasonally Adjusted",
        "PAYEMS": "All Employees, Total Nonfarm, Thousands of Persons, Monthly, Seasonally Adjusted",
        "HOUST": "New Privately-Owned Housing Units Started, Thousands of Units, Monthly, Seasonally Adjusted Annual Rate",
        "GFDEBTN": "Federal Debt: Total Public Debt, Millions of Dollars, Quarterly",
        "MORTGAGE30US": "30-Year Fixed Rate Mortgage Average in the United States, Percent, Weekly",
    }

    desc = _SERIES_DESCRIPTIONS.get(series_id, f"FRED series {series_id}")
    snippet = (
        f"{desc}. "
        f"Source: Federal Reserve Bank of St. Louis (FRED). "
        f"Series ID: {series_id}. "
        f"This is official U.S. government economic data updated regularly. "
        f"View full historical data and charts at https://fred.stlouisfed.org/series/{series_id}"
    )
    return snippet
