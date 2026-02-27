"""World Bank — verify international economic claims.

Uses the free World Bank Indicators API (no key required).
Covers: GDP, GNI, population, trade, debt, development indicators.

Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from .base import rate_limited_get


_BASE_URL = "https://api.worldbank.org/v2"

# Map keywords to World Bank indicators
_INDICATOR_MAP: Dict[str, tuple[str, str]] = {
    # (indicator_code, description)
    "gdp": ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    "gross domestic product": ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    "gdp per capita": ("NY.GDP.PCAP.CD", "GDP per capita (current US$)"),
    "gdp growth": ("NY.GDP.MKTP.KD.ZG", "GDP growth (annual %)"),
    "gni": ("NY.GNP.MKTP.CD", "GNI (current US$)"),
    "gni per capita": ("NY.GNP.PCAP.CD", "GNI per capita (current US$)"),
    "population": ("SP.POP.TOTL", "Population, total"),
    "life expectancy": ("SP.DYN.LE00.IN", "Life expectancy at birth (years)"),
    "infant mortality": ("SP.DYN.IMRT.IN", "Mortality rate, infant (per 1,000 live births)"),
    "co2 emissions": ("EN.ATM.CO2E.KT", "CO2 emissions (kt)"),
    "carbon emissions": ("EN.ATM.CO2E.KT", "CO2 emissions (kt)"),
    "renewable energy": ("EG.FEC.RNEW.ZS", "Renewable energy consumption (% of total)"),
    "electricity": ("EG.ELC.ACCS.ZS", "Access to electricity (% of population)"),
    "trade": ("NE.TRD.GNFS.ZS", "Trade (% of GDP)"),
    "exports": ("NE.EXP.GNFS.ZS", "Exports of goods and services (% of GDP)"),
    "imports": ("NE.IMP.GNFS.ZS", "Imports of goods and services (% of GDP)"),
    "foreign aid": ("DT.ODA.ALLD.CD", "Net official development assistance received (current US$)"),
    "external debt": ("DT.DOD.DECT.CD", "External debt stocks, total (DOD, current US$)"),
    "debt": ("GC.DOD.TOTL.GD.ZS", "Central government debt, total (% of GDP)"),
    "poverty": ("SI.POV.DDAY", "Poverty headcount ratio at $2.15/day (% of population)"),
    "inequality": ("SI.POV.GINI", "Gini index"),
    "gini": ("SI.POV.GINI", "Gini index"),
    "education": ("SE.XPD.TOTL.GD.ZS", "Government expenditure on education (% of GDP)"),
    "literacy": ("SE.ADT.LITR.ZS", "Literacy rate, adult total (% of people ages 15+)"),
    "unemployment": ("SL.UEM.TOTL.ZS", "Unemployment, total (% of total labor force)"),
    "inflation": ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
    "oil": ("NY.GDP.PETR.RT.ZS", "Oil rents (% of GDP)"),
}

# Country name to ISO 3166-1 alpha-2 code
_COUNTRY_CODES: Dict[str, str] = {
    "united states": "US", "america": "US", "usa": "US", "u.s.": "US",
    "china": "CN", "chinese": "CN",
    "india": "IN", "indian": "IN",
    "japan": "JP", "japanese": "JP",
    "germany": "DE", "german": "DE",
    "united kingdom": "GB", "uk": "GB", "britain": "GB", "british": "GB",
    "france": "FR", "french": "FR",
    "brazil": "BR", "brazilian": "BR",
    "canada": "CA", "canadian": "CA",
    "russia": "RU", "russian": "RU",
    "australia": "AU", "australian": "AU",
    "mexico": "MX", "mexican": "MX",
    "south korea": "KR", "korea": "KR",
    "italy": "IT", "italian": "IT",
    "spain": "ES", "spanish": "ES",
    "nigeria": "NG",
    "south africa": "ZA",
    "indonesia": "ID",
    "turkey": "TR",
    "saudi arabia": "SA",
    "argentina": "AR",
    "world": "WLD",
    "global": "WLD",
}


def _match_indicator(claim_text: str) -> Optional[tuple[str, str]]:
    lower = claim_text.lower()
    for term in sorted(_INDICATOR_MAP.keys(), key=len, reverse=True):
        if term in lower:
            return _INDICATOR_MAP[term]
    return None


def _extract_country(claim_text: str) -> str:
    lower = claim_text.lower()
    for name in sorted(_COUNTRY_CODES.keys(), key=len, reverse=True):
        if name in lower:
            return _COUNTRY_CODES[name]
    return "WLD"  # Default to world aggregate


def search_worldbank(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search World Bank for international economic data matching a claim.

    Standard evidence source signature.
    """
    indicator_match = _match_indicator(claim_text)
    if not indicator_match:
        return []

    indicator_code, indicator_desc = indicator_match
    country = _extract_country(claim_text)

    # Fetch indicator data for country
    resp = rate_limited_get(
        f"{_BASE_URL}/country/{country}/indicator/{indicator_code}",
        source_name="worldbank",
        params={
            "format": "json",
            "date": "2015:2024",
            "per_page": "10",
        },
        timeout=15.0,
    )

    snippet = f"{indicator_desc}. Source: World Bank. Country: {country}."

    if resp is not None:
        try:
            data = resp.json()
            # World Bank returns [metadata, data_array]
            if isinstance(data, list) and len(data) >= 2:
                records = data[1]
                if records:
                    values = []
                    for rec in records:
                        if rec and rec.get("value") is not None:
                            year = rec.get("date", "")
                            val = rec["value"]
                            country_name = rec.get("country", {}).get("value", country)
                            if isinstance(val, (int, float)):
                                if abs(val) >= 1e9:
                                    values.append(f"{year}: ${val/1e9:.1f}B")
                                elif abs(val) >= 1e6:
                                    values.append(f"{year}: ${val/1e6:.1f}M")
                                elif abs(val) < 100:
                                    values.append(f"{year}: {val:.1f}%")
                                else:
                                    values.append(f"{year}: {val:,.0f}")
                    if values:
                        snippet = (
                            f"{indicator_desc} - {country_name if records else country}. "
                            f"Source: World Bank. "
                            f"Data: {'; '.join(values[:8])}."
                        )
        except Exception:
            pass

    results = [{
        "url": f"https://data.worldbank.org/indicator/{indicator_code}?locations={country}",
        "title": f"World Bank: {indicator_desc} ({country})",
        "source_name": "worldbank",
        "evidence_type": "dataset",
        "snippet": snippet[:2000],
    }]

    return results[:max_results]
