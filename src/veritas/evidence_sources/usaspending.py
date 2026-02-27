"""USASpending — verify federal spending and contract claims.

Uses the free USASpending API (no key required).
Covers: federal contracts, grants, loans, spending by agency.

Docs: https://api.usaspending.gov/
"""

from __future__ import annotations
import json
import re
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query

import requests
import time

_BASE_URL = "https://api.usaspending.gov/api/v2"
_LAST_REQUEST: float = 0.0
_MIN_INTERVAL = 1.0


def _rate_limit():
    global _LAST_REQUEST
    elapsed = time.time() - _LAST_REQUEST
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_REQUEST = time.time()


# Keywords indicating spending/contract claims
_SPENDING_TERMS = frozenset({
    "spending", "budget", "contract", "contracts", "grant", "grants",
    "federal spending", "government spending", "appropriation",
    "billion", "million", "trillion", "allocated", "funded",
    "agency", "department", "pentagon", "defense spending",
    "infrastructure", "stimulus", "bailout",
})


def _has_spending_relevance(claim_text: str) -> bool:
    lower = claim_text.lower()
    return sum(1 for t in _SPENDING_TERMS if t in lower) >= 1


def search_usaspending(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search USASpending for federal spending data matching a claim.

    Standard evidence source signature.
    """
    if not _has_spending_relevance(claim_text):
        return []

    query = build_search_query(claim_text, max_terms=4)
    if not query:
        return []

    results = []

    # Strategy 1: Search spending by award keyword
    _rate_limit()
    try:
        resp = requests.post(
            f"{_BASE_URL}/search/spending_by_award/",
            json={
                "filters": {
                    "keywords": query.replace('"', '').split()[:3],
                    "time_period": [{"start_date": "2018-01-01", "end_date": "2026-12-31"}],
                },
                "fields": [
                    "Award ID", "Recipient Name", "Award Amount",
                    "Awarding Agency", "Description",
                ],
                "limit": min(max_results, 5),
                "page": 1,
                "sort": "Award Amount",
                "order": "desc",
            },
            headers={"Content-Type": "application/json"},
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        awards = data.get("results", [])
        for award in awards[:max_results]:
            recipient = award.get("Recipient Name", "")
            amount = award.get("Award Amount", 0)
            agency = award.get("Awarding Agency", "")
            desc = award.get("Description", "")
            award_id = award.get("Award ID", "")

            title = f"Federal Award: {recipient[:60]}"
            snippet = (
                f"Recipient: {recipient}. "
                f"Amount: ${amount:,.0f}. " if isinstance(amount, (int, float)) else
                f"Recipient: {recipient}. "
            )
            snippet += f"Agency: {agency}. " if agency else ""
            snippet += f"Description: {desc[:300]}." if desc else ""

            results.append({
                "url": f"https://www.usaspending.gov/award/{award_id}" if award_id else "https://www.usaspending.gov",
                "title": title[:200],
                "source_name": "usaspending",
                "evidence_type": "gov",
                "snippet": snippet[:2000],
            })
    except Exception:
        pass

    # Strategy 2: Agency spending totals (if claim mentions agencies)
    if not results:
        _rate_limit()
        try:
            resp2 = requests.get(
                f"{_BASE_URL}/references/agency/",
                timeout=10.0,
            )
            if resp2.status_code == 200:
                agencies = resp2.json().get("results", [])
                # Match agency name from claim
                lower = claim_text.lower()
                for ag in agencies[:50]:
                    name = ag.get("agency_name", "")
                    if name and name.lower()[:10] in lower:
                        results.append({
                            "url": f"https://www.usaspending.gov/agency/{ag.get('agency_slug', '')}",
                            "title": f"USASpending: {name}",
                            "source_name": "usaspending",
                            "evidence_type": "gov",
                            "snippet": f"Federal agency: {name}. View spending data at USASpending.gov.",
                        })
                        break
        except Exception:
            pass

    return results[:max_results]
