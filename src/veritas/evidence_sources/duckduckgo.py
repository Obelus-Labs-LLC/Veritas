"""DuckDuckGo Instant Answers — general knowledge verification.

Uses the DuckDuckGo Instant Answer API (free, no key required).
Returns structured answers from 600+ sources including Wikipedia
infoboxes, Wolfram Alpha, StackOverflow, IMDb, etc.

Ideal for: general knowledge facts, definitions, entity lookups,
and any claim that specialised APIs don't cover.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

from .base import rate_limited_get

_API_URL = "https://api.duckduckgo.com/"


def _extract_ddg_queries(claim_text: str) -> List[str]:
    """Extract candidate queries for DDG Instant Answers, best-first.

    DDG's Instant Answer API works best with short entity names or
    topic phrases (like Wikipedia article titles), NOT complex boolean
    search strings. Returns multiple candidates to try in order.
    """
    queries: List[str] = []

    # 1. Multi-word proper nouns (e.g. "Goldman Sachs", "Albert Einstein")
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim_text)
    # Sort by position in text (earlier = more likely the subject)
    for ent in entities:
        if ent not in queries:
            queries.append(ent)

    # 2. Acronyms (e.g. SEC, GDP, NASA, NVIDIA)
    skip = {"I", "A", "THE", "AND", "BUT", "FOR", "NOT", "WAS", "HAS", "CEO",
            "CFO", "CTO", "COO", "IPO", "Q1", "Q2", "Q3", "Q4", "FY"}
    acronyms = [a for a in re.findall(r'\b[A-Z]{2,6}\b', claim_text) if a not in skip]
    for acr in acronyms:
        if acr not in queries:
            queries.append(acr)

    # 3. Sentence-start capitalized word (often the subject: "Apple was founded")
    words = claim_text.split()
    if words:
        first = words[0].strip(".,!?;:\"'()[]")
        if first and first[0].isupper() and first.isalpha() and len(first) > 2:
            common_starts = {"The", "This", "That", "These", "Those", "There",
                             "They", "Their", "What", "Which", "Where", "When",
                             "How", "Who", "Why", "Our", "His", "Her", "Its",
                             "Some", "Many", "Most", "All", "Each", "Every",
                             "And", "But", "Also", "Just", "Very", "More",
                             "Then", "Now", "Well", "Here"}
            if first not in common_starts and first not in queries:
                queries.append(first)

    # 4. Single capitalized word mid-sentence
    for i, w in enumerate(words):
        if i == 0:
            continue
        clean = w.strip(".,!?;:\"'()[]")
        if clean and clean[0].isupper() and clean.isalpha() and len(clean) > 2:
            if clean not in queries:
                queries.append(clean)

    # 5. Fallback: strip filler words and return a short phrase
    if not queries:
        stop_words = frozenset([
            "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
            "be", "been", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "to", "of", "in", "for", "on", "at", "by",
            "with", "from", "as", "and", "but", "or", "so", "if", "than", "that",
            "this", "it", "its", "not", "no", "just", "very", "really", "also",
            "about", "we", "our", "they", "their", "he", "she", "his", "her",
            "you", "your", "there", "here", "been", "being", "which", "what",
        ])
        key = [w.strip(".,!?;:\"'()[]").lower() for w in words
               if w.strip(".,!?;:\"'()[]").lower() not in stop_words
               and len(w.strip(".,!?;:\"'()[]")) > 2]
        fallback = " ".join(key[:4])
        if fallback:
            queries.append(fallback)

    return queries[:4]  # Max 4 candidates to try


def _fetch_ddg(query: str) -> Dict[str, Any]:
    """Make a single DDG Instant Answer API call. Returns parsed JSON or {}."""
    resp = rate_limited_get(
        _API_URL,
        source_name="duckduckgo",
        params={
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        },
    )
    if resp is None:
        return {}
    try:
        return resp.json()
    except (ValueError, Exception):
        return {}


def _parse_ddg_response(data: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
    """Parse DDG Instant Answer response into evidence results."""
    results: List[Dict[str, Any]] = []

    # 1. Abstract (main Wikipedia-like summary)
    abstract = (data.get("Abstract") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    abstract_source = (data.get("AbstractSource") or "").strip()
    if abstract and abstract_url:
        results.append({
            "url": abstract_url,
            "title": f"{abstract_source}: {data.get('Heading', '')}".strip(": "),
            "source_name": "duckduckgo",
            "evidence_type": "secondary",
            "snippet": abstract[:2000],
        })

    # 2. Answer (direct factual answer, e.g. calculations, conversions)
    answer = (data.get("Answer") or "").strip()
    if answer and not abstract:
        results.append({
            "url": data.get("AbstractURL", "https://duckduckgo.com"),
            "title": f"DuckDuckGo Answer: {data.get('Heading', '')}".strip(": "),
            "source_name": "duckduckgo",
            "evidence_type": "secondary",
            "snippet": answer[:2000],
        })

    # 3. Definition
    definition = (data.get("Definition") or "").strip()
    definition_url = (data.get("DefinitionURL") or "").strip()
    if definition and definition_url and not abstract:
        results.append({
            "url": definition_url,
            "title": f"Definition: {data.get('Heading', '')}".strip(": "),
            "source_name": "duckduckgo",
            "evidence_type": "secondary",
            "snippet": definition[:2000],
        })

    # 4. Related Topics (often contain useful snippets)
    related = data.get("RelatedTopics", [])
    for topic in related:
        if len(results) >= max_results:
            break
        if "Topics" in topic:
            continue
        text = (topic.get("Text") or "").strip()
        url = (topic.get("FirstURL") or "").strip()
        if text and url:
            results.append({
                "url": url,
                "title": text[:120],
                "source_name": "duckduckgo",
                "evidence_type": "secondary",
                "snippet": text[:2000],
            })

    return results[:max_results]


def search_duckduckgo(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search DuckDuckGo Instant Answers for evidence matching a claim.

    Tries multiple query candidates (entity names, acronyms, topic phrases)
    and returns results from the first query that produces a meaningful answer.
    """
    queries = _extract_ddg_queries(claim_text)
    if not queries:
        return []

    seen_urls: set = set()
    all_results: List[Dict[str, Any]] = []

    for query in queries:
        if not query or len(query.strip()) < 2:
            continue

        data = _fetch_ddg(query)
        if not data:
            continue

        results = _parse_ddg_response(data, max_results)

        # If we got a strong result (Abstract or Answer), return immediately
        has_abstract = bool((data.get("Abstract") or "").strip())
        has_answer = bool((data.get("Answer") or "").strip())

        for r in results:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                all_results.append(r)

        if has_abstract or has_answer:
            break  # Good enough — don't burn more API calls

        # If we got related topics but no abstract, keep trying
        if len(all_results) >= max_results:
            break

    return all_results[:max_results]
