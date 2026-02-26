"""Shared utilities for evidence source modules."""

from __future__ import annotations
import time
import requests
from typing import List, Dict, Any, Optional

# Global rate limiter: minimum seconds between requests per source
_LAST_REQUEST: Dict[str, float] = {}
_MIN_INTERVAL = 1.0  # 1 second between API calls per source


def rate_limited_get(
    url: str,
    source_name: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 15.0,
) -> Optional[requests.Response]:
    """HTTP GET with rate limiting and error handling.

    Returns None on any failure (timeout, HTTP error, network error).
    """
    now = time.time()
    last = _LAST_REQUEST.get(source_name, 0)
    wait = _MIN_INTERVAL - (now - last)
    if wait > 0:
        time.sleep(wait)

    default_headers = {
        "User-Agent": "Veritas/1.0 (local research tool; mailto:noreply@local)",
    }
    if headers:
        default_headers.update(headers)

    try:
        resp = requests.get(url, params=params, headers=default_headers, timeout=timeout)
        _LAST_REQUEST[source_name] = time.time()
        resp.raise_for_status()
        return resp
    except (requests.RequestException, Exception):
        _LAST_REQUEST[source_name] = time.time()
        return None


def build_search_query(claim_text: str, max_terms: int = 8) -> str:
    """Extract key terms from a claim for API search queries.

    Strips common filler words and keeps the most informative tokens.
    Preserves multi-word proper nouns (e.g. "Goldman Sachs", "Framingham Study")
    as quoted phrases for better API search matching.
    """
    import re

    stop_words = frozenset([
        "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
        "be", "been", "being", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "at", "by", "with", "from", "as", "into", "about", "between",
        "through", "during", "before", "after", "and", "but", "or", "so",
        "if", "then", "than", "that", "this", "these", "those", "it", "its",
        "not", "no", "just", "very", "really", "also", "too", "more", "most",
        "some", "any", "all", "each", "every", "both", "few", "many",
        "much", "own", "other", "such", "only",
    ])

    # Step 1: Extract multi-word proper nouns (e.g. "Goldman Sachs", "Federal Reserve")
    # These get preserved as quoted phrases for better search precision
    proper_noun_re = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')
    proper_nouns = proper_noun_re.findall(claim_text)

    # Filter out very common multi-word phrases that aren't useful for search
    _COMMON_PHRASES = frozenset([
        "United States", "New York", "Last Year", "Next Year",
        "First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter",
    ])
    proper_nouns = [pn for pn in proper_nouns if pn not in _COMMON_PHRASES]

    # Build quoted phrases (count each as roughly 2 terms toward max)
    quoted_phrases = []
    terms_used = 0
    for pn in proper_nouns[:3]:  # max 3 proper noun phrases
        quoted_phrases.append(f'"{pn}"')
        terms_used += 2

    # Step 2: Extract single key terms (skip words already in proper noun phrases)
    proper_noun_words = set()
    for pn in proper_nouns:
        proper_noun_words.update(pn.split())

    words = claim_text.split()
    key_terms = []
    for w in words:
        cleaned = w.strip(".,!?;:\"'()[]")
        if not cleaned:
            continue
        # Skip words already captured in proper noun phrases
        if cleaned in proper_noun_words:
            continue
        lower = cleaned.lower()
        # Always keep numbers
        if any(c.isdigit() for c in cleaned):
            key_terms.append(cleaned)
        # Keep capitalized words (remaining proper nouns)
        elif cleaned[0].isupper() and lower not in stop_words:
            key_terms.append(cleaned)
        elif lower not in stop_words and len(lower) > 2:
            key_terms.append(lower)

    remaining_slots = max_terms - terms_used
    result_parts = quoted_phrases + key_terms[:max(0, remaining_slots)]
    return " ".join(result_parts)
