"""PubMed / NCBI E-utilities — search biomedical literature.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
No API key required (but limited to 3 req/sec without one).
"""

from __future__ import annotations
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query

_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def search_pubmed(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search PubMed for biomedical articles matching a claim.

    Returns list of dicts with keys: url, title, source_name, evidence_type, snippet.
    """
    query = build_search_query(claim_text)
    if not query:
        return []

    # Step 1: search for matching PMIDs
    resp = rate_limited_get(
        _SEARCH_URL,
        source_name="pubmed",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        },
    )
    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    id_list = data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    # Step 2: get summaries for those PMIDs
    resp2 = rate_limited_get(
        _SUMMARY_URL,
        source_name="pubmed",
        params={
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json",
        },
    )
    if resp2 is None:
        return []

    try:
        summary_data = resp2.json()
    except Exception:
        return []

    result_map = summary_data.get("result", {})
    results = []
    for pmid in id_list:
        info = result_map.get(pmid, {})
        title = info.get("title", "")
        source = info.get("source", "")  # journal name

        if title:
            results.append({
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "title": title[:200],
                "source_name": "pubmed",
                "evidence_type": "paper",
                "snippet": f"Published in: {source}" if source else "",
            })

    return results
