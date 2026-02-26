"""Wikipedia API — verify entity-level facts against Wikipedia.

Uses the MediaWiki Action API (free, no key required):
  - search: find relevant articles
  - extracts: get article summaries with actual facts/numbers

Ideal for: company facts (founding dates, HQ, key people),
historical events, person bios, geographic data.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query

_SEARCH_URL = "https://en.wikipedia.org/w/api.php"


def search_wikipedia(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Wikipedia for articles matching a claim.

    Standard evidence source signature. Returns list of dicts with keys:
    url, title, source_name, evidence_type, snippet.
    """
    query = build_search_query(claim_text)
    if not query:
        return []

    # Step 1: Search for matching articles
    resp = rate_limited_get(
        _SEARCH_URL,
        source_name="wikipedia",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(min(max_results, 5)),
            "format": "json",
            "utf8": "1",
        },
    )
    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    search_results = data.get("query", {}).get("search", [])
    if not search_results:
        return []

    # Step 2: Get extracts (summaries) for the matching pages
    page_ids = [str(r["pageid"]) for r in search_results]

    extract_resp = rate_limited_get(
        _SEARCH_URL,
        source_name="wikipedia",
        params={
            "action": "query",
            "pageids": "|".join(page_ids),
            "prop": "extracts|info",
            "exintro": "1",        # only intro section
            "explaintext": "1",    # plain text, no HTML
            "exlimit": str(len(page_ids)),
            "inprop": "url",
            "format": "json",
            "utf8": "1",
        },
    )
    if extract_resp is None:
        # Fall back to search snippets only
        return _build_results_from_search(search_results, max_results)

    try:
        extract_data = extract_resp.json()
    except Exception:
        return _build_results_from_search(search_results, max_results)

    pages = extract_data.get("query", {}).get("pages", {})
    results = []

    for sr in search_results:
        pid = str(sr["pageid"])
        page = pages.get(pid, {})
        title = page.get("title", sr.get("title", ""))
        url = page.get("fullurl", f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}")
        extract = page.get("extract", "")

        # Clean up the extract for the scoring engine
        snippet = _clean_extract(extract, claim_text)

        results.append({
            "url": url,
            "title": f"{title} - Wikipedia",
            "source_name": "wikipedia",
            "evidence_type": "secondary",  # Wikipedia is secondary source
            "snippet": snippet[:2000],
        })

        if len(results) >= max_results:
            break

    return results


def _clean_extract(extract: str, claim_text: str) -> str:
    """Clean and trim Wikipedia extract for scoring engine.

    Keeps the most relevant paragraphs that overlap with claim text.
    """
    if not extract:
        return ""

    # Split into paragraphs
    paragraphs = [p.strip() for p in extract.split("\n") if p.strip()]
    if not paragraphs:
        return extract[:500]

    # Score paragraphs by relevance to claim
    claim_words = set(claim_text.lower().split())
    scored = []
    for p in paragraphs:
        p_words = set(p.lower().split())
        overlap = len(claim_words & p_words)
        scored.append((overlap, p))

    # Sort by relevance, take top 3 paragraphs
    scored.sort(key=lambda x: -x[0])
    best = scored[:3]

    return " ".join(p for _, p in best)


def _build_results_from_search(
    search_results: List[Dict[str, Any]],
    max_results: int,
) -> List[Dict[str, Any]]:
    """Fallback: build results from search snippets when extracts fail."""
    results = []
    for sr in search_results[:max_results]:
        title = sr.get("title", "")
        # Search snippets have HTML tags — strip them
        snippet = re.sub(r'<[^>]+>', '', sr.get("snippet", ""))
        results.append({
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "title": f"{title} - Wikipedia",
            "source_name": "wikipedia",
            "evidence_type": "secondary",
            "snippet": snippet[:500],
        })
    return results
