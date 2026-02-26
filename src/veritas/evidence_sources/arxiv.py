"""arXiv API — search preprints by query.

Docs: https://info.arxiv.org/help/api/index.html
No API key required.  Rate limit: 1 req/3 sec recommended.
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query

_BASE_URL = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search arXiv for preprints matching a claim.

    Returns list of dicts with keys: url, title, source_name, evidence_type, snippet.
    """
    query = build_search_query(claim_text)
    if not query:
        return []

    resp = rate_limited_get(
        _BASE_URL,
        source_name="arxiv",
        params={
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(max_results),
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        timeout=20.0,
    )
    if resp is None:
        return []

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    results = []
    for entry in root.findall("atom:entry", _NS):
        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)
        link_el = entry.find("atom:id", _NS)

        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
        summary = summary_el.text.strip().replace("\n", " ")[:300] if summary_el is not None and summary_el.text else ""
        url = link_el.text.strip() if link_el is not None and link_el.text else ""

        if url and title:
            results.append({
                "url": url,
                "title": title[:200],
                "source_name": "arxiv",
                "evidence_type": "paper",
                "snippet": summary[:200],
            })

    return results
