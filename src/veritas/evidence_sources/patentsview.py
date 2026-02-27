"""USPTO PatentsView — verify technology/innovation claims.

PatentsView requires a free API key (X-Api-Key header).
Falls back to metadata-only reference links if no key is set.

Covers: patent counts, assignees, technology classifications.

Docs: https://search.patentsview.org/docs/
"""

from __future__ import annotations
import os
import re
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query

_BASE_URL = "https://search.patentsview.org/api/v1"
_API_KEY = os.environ.get("PATENTSVIEW_API_KEY", "")

# Tech/innovation keywords
_PATENT_TERMS = frozenset({
    "patent", "patents", "patented", "innovation", "invention",
    "intellectual property", "ip", "patent filing",
    "patent application", "utility patent", "design patent",
    "trademark", "r&d", "research and development",
})


def _has_patent_relevance(claim_text: str) -> bool:
    lower = claim_text.lower()
    return sum(1 for t in _PATENT_TERMS if t in lower) >= 1


def _extract_company_for_patent(claim_text: str) -> str:
    """Extract company name for patent assignee search."""
    # Look for capitalized proper nouns (company names)
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', claim_text)
    skip = {"The", "How", "Why", "What", "This", "That", "New", "Patent", "Innovation"}
    entities = [e for e in entities if e.split()[0] not in skip]
    return entities[0] if entities else ""


def search_patentsview(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search PatentsView for patent data matching a claim.

    Standard evidence source signature.
    Falls back to reference links if API key is not available.
    """
    if not _has_patent_relevance(claim_text):
        return []

    query = build_search_query(claim_text, max_terms=4)
    if not query:
        return []

    results = []

    # If we have an API key, use the actual API
    if _API_KEY:
        company = _extract_company_for_patent(claim_text)
        search_q = company if company else query.replace('"', '').split()[0]

        resp = rate_limited_get(
            f"{_BASE_URL}/patent/",
            source_name="patentsview",
            params={
                "q": f'{{"_text_any":{{"patent_abstract":"{search_q}"}}}}',
                "f": '["patent_id","patent_title","patent_date","patent_abstract"]',
                "per_page": str(min(max_results, 5)),
            },
            headers={
                "X-Api-Key": _API_KEY,
            },
            timeout=15.0,
        )

        if resp is not None:
            try:
                data = resp.json()
                patents = data.get("patents", [])
                for p in patents[:max_results]:
                    patent_id = p.get("patent_id", "")
                    title = p.get("patent_title", "")
                    date = p.get("patent_date", "")
                    abstract = p.get("patent_abstract", "")

                    results.append({
                        "url": f"https://patents.google.com/patent/US{patent_id}",
                        "title": f"Patent US{patent_id}: {title}"[:200],
                        "source_name": "patentsview",
                        "evidence_type": "gov",
                        "snippet": f"Filed: {date}. {abstract[:500]}"[:2000],
                    })
            except Exception:
                pass

    # Fallback: reference link to PatentsView search
    if not results:
        company = _extract_company_for_patent(claim_text)
        search_term = company if company else query[:50]
        results.append({
            "url": f"https://patentsview.org/search/{search_term.replace(' ', '+')}",
            "title": f"PatentsView: {search_term} Patents",
            "source_name": "patentsview",
            "evidence_type": "gov",
            "snippet": (
                f"USPTO patent search for: {search_term}. "
                f"PatentsView provides patent data from the US Patent and Trademark Office. "
                f"Source: USPTO/PatentsView."
            ),
        })

    return results[:max_results]
