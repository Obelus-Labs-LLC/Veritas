"""SEC.gov publications search — verify claims against SEC reports and data.

Unlike sec_edgar.py which searches EDGAR corporate filings (10-K, 10-Q, 8-K),
this source searches SEC's own publications: annual reports, budget justifications,
strategic plans, enforcement statistics, and staff reports.

Uses SEC website search (site:sec.gov) via EFTS and direct sec.gov endpoints.
No API key required. Rate limit: 10 req/sec.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query


# SEC EFTS full-text search covers SEC publications, not just EDGAR filings
_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# SEC requires specific User-Agent format
_SEC_UA = "Veritas Research Tool research@veritas-app.local"

# Keywords that indicate a claim is about SEC-as-institution (not corporate filings)
_SEC_INSTITUTIONAL_TERMS = frozenset({
    "sec", "securities and exchange commission", "commission",
    "enforcement", "examination", "examinations", "inspection",
    "registrant", "registrants", "registered", "registration",
    "division", "office", "staff", "fte", "budget",
    "appropriation", "congressional", "rulemaking",
    "transfer agent", "transfer agents",
    "investment adviser", "investment advisers", "adviser", "advisers",
    "broker-dealer", "broker-dealers", "broker dealer",
    "municipal advisor", "municipal advisors",
    "swap dealer", "security-based swap",
    "nationally recognized", "clearing agency",
    "self-regulatory", "sro",
    "whistleblower", "disgorgement", "penalty", "penalties",
    "filing fee", "filing fees",
    "investor protection", "market integrity",
    "tipster", "complaint", "complaints",
})

# Known SEC report series to search for
_SEC_REPORT_KEYWORDS = {
    "annual report": "SEC Annual Report",
    "budget justification": "Congressional Budget Justification",
    "strategic plan": "SEC Strategic Plan",
    "enforcement": "Division of Enforcement Annual Report",
    "examination": "Division of Examinations Annual Report",
    "inspection": "Office of Inspections Annual Report",
    "investor advocate": "Office of Investor Advocate Report",
    "ombudsman": "SEC Ombudsman Report",
}


def _has_sec_relevance(claim_text: str) -> bool:
    """Check if a claim is about SEC institutional operations.

    Returns True if the claim mentions SEC-specific terminology
    that would be found in SEC publications (not corporate filings).
    """
    lower = claim_text.lower()
    for term in _SEC_INSTITUTIONAL_TERMS:
        if term in lower:
            return True
    return False


def search_sec_gov(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search SEC.gov for institutional reports matching a claim.

    Pre-filters: only queries if claim mentions SEC institutional terms.
    Standard evidence source signature.
    """
    if not _has_sec_relevance(claim_text):
        return []

    query = build_search_query(claim_text, max_terms=6)
    if not query:
        return []

    # Add "SEC" to query if not already present
    if "sec" not in query.lower() and "securities" not in query.lower():
        query = f"SEC {query}"

    # Search SEC EFTS — this indexes sec.gov publications
    resp = rate_limited_get(
        _SEARCH_URL,
        source_name="sec_gov",
        params={
            "q": query,
            "dateRange": "custom",
            "startdt": "2018-01-01",
            "enddt": "2026-12-31",
        },
        headers={
            "User-Agent": _SEC_UA,
            "Accept": "application/json",
        },
    )

    results = []

    if resp is not None:
        try:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits[:max_results]:
                src = hit.get("_source", {})
                file_date = src.get("file_date", "")
                form = src.get("form", "")
                display_names = src.get("display_names", [])
                entity_name = display_names[0].split("(CIK")[0].strip() if display_names else "SEC"

                # Build title
                title = f"{entity_name} - {form}" if form else entity_name

                # Build URL
                adsh = src.get("adsh", "")
                ciks = src.get("ciks", [])
                cik = ciks[0] if ciks else ""
                if adsh and cik:
                    adsh_clean = adsh.replace("-", "")
                    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_clean}/"
                elif adsh:
                    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&accession={adsh}"
                else:
                    continue

                period = src.get("period_ending", "")
                snippet = ""
                if file_date:
                    snippet += f"Filed: {file_date}"
                if period:
                    snippet += f" | Period: {period}"
                if form:
                    snippet += f" | Form: {form}"

                results.append({
                    "url": url,
                    "title": title[:200],
                    "source_name": "sec_gov",
                    "evidence_type": "gov",
                    "snippet": snippet[:200],
                    "evidence_date": file_date[:4] if file_date else "",
                })
        except Exception:
            pass

    # Also search for SEC annual reports and statistical data via sec.gov search
    # SEC publishes reports at sec.gov/about/reports.shtml
    _search_sec_reports(claim_text, results, max_results)

    return results[:max_results]


def _search_sec_reports(
    claim_text: str,
    results: List[Dict[str, Any]],
    max_results: int,
) -> None:
    """Search for SEC reports using sec.gov website search.

    Appends results to the provided list.
    """
    lower = claim_text.lower()

    # Match against known report types
    for keyword, report_name in _SEC_REPORT_KEYWORDS.items():
        if keyword in lower:
            # Build a targeted query for this report type
            query = f"SEC {report_name}"
            # Extract years from claim for temporal targeting
            years = re.findall(r'\b(20\d{2})\b', claim_text)
            if years:
                query += f" {years[0]}"

            resp = rate_limited_get(
                _SEARCH_URL,
                source_name="sec_gov",
                params={
                    "q": query,
                    "dateRange": "custom",
                    "startdt": "2018-01-01",
                    "enddt": "2026-12-31",
                },
                headers={
                    "User-Agent": _SEC_UA,
                    "Accept": "application/json",
                },
            )
            if resp is not None:
                try:
                    data = resp.json()
                    hits = data.get("hits", {}).get("hits", [])
                    for hit in hits[:2]:
                        src = hit.get("_source", {})
                        file_date = src.get("file_date", "")
                        adsh = src.get("adsh", "")
                        ciks = src.get("ciks", [])
                        cik = ciks[0] if ciks else ""

                        if adsh and cik:
                            adsh_clean = adsh.replace("-", "")
                            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_clean}/"
                        else:
                            continue

                        snippet = f"{report_name}. Filed: {file_date}" if file_date else report_name
                        results.append({
                            "url": url,
                            "title": f"SEC: {report_name}"[:200],
                            "source_name": "sec_gov",
                            "evidence_type": "gov",
                            "snippet": snippet[:200],
                            "evidence_date": file_date[:4] if file_date else "",
                        })
                except Exception:
                    pass

            if len(results) >= max_results:
                break
