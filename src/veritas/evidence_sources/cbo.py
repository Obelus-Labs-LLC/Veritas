"""Congressional Budget Office (CBO) — reference source for budget/fiscal claims.

CBO does not have a public JSON API. This source provides reference links
to CBO reports and publications via their website search, using the
GovInfo API (free, api.data.gov key optional with DEMO_KEY fallback).

Ideal for: budget projections, deficit, national debt, fiscal analysis,
cost estimates, entitlement spending, Social Security, Medicare.

Docs: https://api.govinfo.gov/docs/
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

from .base import rate_limited_get, build_search_query


_GOVINFO_URL = "https://api.govinfo.gov/search"
_API_KEY = "DEMO_KEY"  # Free demo key for govinfo.gov

# CBO topic keywords
_CBO_TERMS = frozenset({
    "budget", "deficit", "surplus", "national debt", "federal debt",
    "cbo", "congressional budget", "cost estimate",
    "social security", "medicare", "medicaid", "entitlement",
    "fiscal", "appropriations", "discretionary spending",
    "mandatory spending", "revenue projection", "baseline",
    "debt ceiling", "debt limit", "sequestration",
    "federal spending", "government spending",
})


def _has_cbo_relevance(claim_text: str) -> bool:
    """Check if claim is relevant to CBO data."""
    lower = claim_text.lower()
    return sum(1 for t in _CBO_TERMS if t in lower) >= 1


def search_cbo(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search GovInfo for CBO publications matching a claim.

    Standard evidence source signature.
    """
    if not _has_cbo_relevance(claim_text):
        return []

    query = build_search_query(claim_text, max_terms=5)
    if not query:
        return []

    # Search GovInfo for CBO documents
    resp = rate_limited_get(
        _GOVINFO_URL,
        source_name="cbo",
        params={
            "query": f"collection:BUDGET {query}",
            "pageSize": str(min(max_results, 5)),
            "offsetMark": "*",
            "api_key": _API_KEY,
        },
        timeout=15.0,
    )

    results = []

    if resp is not None:
        try:
            data = resp.json()
            packages = data.get("results", [])
            for pkg in packages[:max_results]:
                title = pkg.get("title", "")
                url = pkg.get("packageLink", "")
                date = pkg.get("dateIssued", "")

                if not title:
                    continue

                snippet = f"CBO/Budget Publication: {title}."
                if date:
                    snippet += f" Published: {date}."

                # Extract more metadata if available
                doc_class = pkg.get("docClass", "")
                if doc_class:
                    snippet += f" Type: {doc_class}."

                results.append({
                    "url": url or f"https://www.cbo.gov/search?query={query.replace(' ', '+')}",
                    "title": f"CBO: {title}"[:200],
                    "source_name": "cbo",
                    "evidence_type": "gov",
                    "snippet": snippet[:2000],
                })
        except Exception:
            pass

    # Fallback: if GovInfo returned nothing, provide a CBO search link
    if not results:
        results.append({
            "url": f"https://www.cbo.gov/search/results?query={query.replace(' ', '+')}",
            "title": f"CBO Search: {query[:80]}",
            "source_name": "cbo",
            "evidence_type": "gov",
            "snippet": (
                f"Congressional Budget Office search for: {query}. "
                f"CBO provides nonpartisan analysis of budgetary and economic issues. "
                f"Source: Congressional Budget Office (cbo.gov)."
            ),
        })

    return results[:max_results]
