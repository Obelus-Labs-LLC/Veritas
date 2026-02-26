"""Crossref API — search academic works by query.

Docs: https://api.crossref.org/swagger-ui/index.html
No API key required.  Rate limit: ~50 req/sec with polite pool.
"""

from __future__ import annotations
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query

_BASE_URL = "https://api.crossref.org/works"


def search_crossref(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Crossref for academic works matching a claim.

    Returns list of dicts with keys: url, title, source_name, evidence_type, snippet.
    """
    query = build_search_query(claim_text)
    if not query:
        return []

    resp = rate_limited_get(
        _BASE_URL,
        source_name="crossref",
        params={
            "query": query,
            "rows": str(max_results),
            "select": "DOI,title,abstract,type,published-print,published-online",
        },
    )
    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    items = data.get("message", {}).get("items", [])
    results = []
    for item in items:
        doi = item.get("DOI", "")
        titles = item.get("title", [])
        title = titles[0] if titles else ""
        abstract = item.get("abstract", "")
        # Clean HTML tags from abstract
        if abstract:
            import re
            abstract = re.sub(r"<[^>]+>", "", abstract)[:300]

        work_type = item.get("type", "")
        evidence_type = "paper"
        if work_type in ("dataset",):
            evidence_type = "dataset"

        results.append({
            "url": f"https://doi.org/{doi}" if doi else "",
            "title": title[:200],
            "source_name": "crossref",
            "evidence_type": evidence_type,
            "snippet": abstract[:200] if abstract else "",
        })

    return results
