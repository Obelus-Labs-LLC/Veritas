"""Crossref API — search academic works by query.

Docs: https://api.crossref.org/swagger-ui/index.html
No API key required.  Rate limit: ~50 req/sec with polite pool.
"""

from __future__ import annotations
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query

_BASE_URL = "https://api.crossref.org/works"


_ACADEMIC_INDICATORS = frozenset({
    "study", "studies", "research", "researchers", "published", "paper",
    "journal", "peer-reviewed", "findings", "experiment", "experiments",
    "hypothesis", "methodology", "statistical", "sample size",
    "correlation", "causation", "meta-analysis", "systematic review",
    "university", "professor", "phd", "thesis", "citation",
    "author", "authors", "scholar", "academic", "literature",
    "doi", "preprint", "manuscript", "proceedings", "conference",
    "evidence", "observed", "measured", "demonstrated", "analyzed",
    "theoretical", "empirical", "framework", "paradigm",
    "clinical", "trial", "trials", "randomized", "placebo",
    "peer reviewed", "double-blind", "cohort", "longitudinal",
})


def _has_academic_relevance(claim_text: str) -> bool:
    """Check if a claim is likely to match academic literature.

    Returns True if the claim contains academic language, specific
    scientific terms, or named entities that suggest an academic source
    would be relevant. Returns False for generic claims, personal
    opinions, or topics better served by government/financial sources.
    """
    lower = claim_text.lower()

    # Direct academic language
    for term in _ACADEMIC_INDICATORS:
        if term in lower:
            return True

    # Named multi-word entities (researchers, institutions, etc.)
    import re
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim_text)
    if len(entities) >= 2:
        return True

    return False


def search_crossref(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Crossref for academic works matching a claim.

    Pre-filters: only queries Crossref if claim has academic relevance.
    Returns list of dicts with keys: url, title, source_name, evidence_type, snippet.
    """
    # Pre-filter: skip claims without academic relevance
    if not _has_academic_relevance(claim_text):
        return []

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
