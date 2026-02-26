"""Google Fact Check Explorer — verify claims against global fact-checker database.

Uses the Fact Check Explorer frontend API (free, no key required).
Returns fact-checked claims from hundreds of verified publishers
(PolitiFact, Snopes, Full Fact, AFP, Reuters, etc.).

Each result includes the original claim, the fact-checker's rating
(e.g., "False", "Mostly True", "Misleading"), and a link to the
full fact-check article.

Ideal for: political claims, viral misinformation, public health claims,
economic statistics cited by politicians or media figures.
"""

from __future__ import annotations
import json
from typing import List, Dict, Any, Optional

from .base import rate_limited_get, build_search_query

_EXPLORER_URL = "https://toolbox.google.com/factcheck/api/search"


def search_google_factcheck(
    claim_text: str,
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    """Search Google Fact Check Explorer for matching fact-checked claims.

    Standard evidence source signature. Returns list of dicts with keys:
    url, title, source_name, evidence_type, snippet.
    """
    query = build_search_query(claim_text, max_terms=10)
    if not query:
        return []

    resp = rate_limited_get(
        _EXPLORER_URL,
        source_name="google_factcheck",
        params={
            "query": query,
            "num": str(min(max_results * 2, 10)),  # over-fetch, filter later
        },
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; Veritas/1.0; research tool)",
        },
    )
    if resp is None:
        return []

    # The Explorer API returns a prefixed JSON response: )]}'\n followed by JSON
    raw = resp.text
    if raw.startswith(")]}'"):
        raw = raw[raw.index("\n") + 1:]

    try:
        parsed = _parse_explorer_response(raw)
    except Exception:
        return []

    results = []
    for item in parsed:
        result = _format_result(item)
        if result:
            results.append(result)
            if len(results) >= max_results:
                break

    return results


def _parse_explorer_response(raw: str) -> List[Dict[str, Any]]:
    """Parse the nested array response from Fact Check Explorer.

    Response structure:
      [["claims_response", [entry1, entry2, ...], ...other_metadata...]]

    Each entry:
      [claim_array, thumbnail_url, relevance_score]

    Each claim_array:
      [claim_text, [claimant, claimant_id], timestamp, [[review1], [review2], ...], ...]

    Each review:
      [[pub_name, pub_site], url, timestamp_or_null, rating, null, [null, id], lang, null, title_snippet, ...]
    """
    try:
        outer = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []

    if not outer or not isinstance(outer, list):
        return []

    # outer[0] = ["claims_response", [entries...], ...]
    response_wrapper = outer[0] if outer else None
    if not response_wrapper or not isinstance(response_wrapper, list) or len(response_wrapper) < 2:
        return []

    entries = response_wrapper[1]
    if not isinstance(entries, list):
        return []

    parsed = []
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 1:
            continue

        # entry[0] is the claim_array with all the data
        claim_array = entry[0]
        if not isinstance(claim_array, list) or len(claim_array) < 4:
            continue

        claim_text = claim_array[0] if isinstance(claim_array[0], str) else ""

        # Extract claimant from [claimant_name, claimant_id]
        claimant = ""
        if isinstance(claim_array[1], list) and len(claim_array[1]) > 0:
            claimant = claim_array[1][0] if isinstance(claim_array[1][0], str) else ""

        # Extract reviews from claim_array[3]
        reviews_block = claim_array[3] if len(claim_array) > 3 and isinstance(claim_array[3], list) else []

        reviews = []
        for rev in reviews_block:
            if not isinstance(rev, list) or len(rev) < 4:
                continue

            # rev[0] = [publisher_name, publisher_site]
            publisher_info = rev[0] if isinstance(rev[0], list) else []
            publisher_name = publisher_info[0] if len(publisher_info) > 0 and isinstance(publisher_info[0], str) else ""
            publisher_site = publisher_info[1] if len(publisher_info) > 1 and isinstance(publisher_info[1], str) else ""

            url = rev[1] if len(rev) > 1 and isinstance(rev[1], str) else ""
            rating = rev[3] if len(rev) > 3 and isinstance(rev[3], str) else ""
            title_snippet = rev[8] if len(rev) > 8 and isinstance(rev[8], str) else ""

            reviews.append({
                "publisher_name": publisher_name,
                "publisher_site": publisher_site,
                "url": url,
                "rating": rating,
                "title_snippet": title_snippet,
            })

        if claim_text and reviews:
            parsed.append({
                "claim_text": claim_text,
                "claimant": claimant,
                "reviews": reviews,
            })

    return parsed


def _format_result(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Format a parsed fact-check item into the standard evidence source result.

    Builds a rich snippet that includes the fact-checker's rating for scoring.
    """
    claim_text = item.get("claim_text", "")
    claimant = item.get("claimant", "")
    reviews = item.get("reviews", [])

    if not reviews:
        return None

    # Use the first review (most relevant)
    best_review = reviews[0]
    url = best_review.get("url", "")
    publisher = best_review.get("publisher_name", "Unknown")
    rating = best_review.get("rating", "")
    title_snippet = best_review.get("title_snippet", "")

    if not url:
        return None

    # Build a title that includes the publisher and rating
    title = f"Fact Check by {publisher}"
    if rating:
        title += f": {rating}"

    # Build a rich snippet for the scoring engine
    snippet_parts = []
    snippet_parts.append(f"Claim: {claim_text}")
    if claimant:
        snippet_parts.append(f"Claimant: {claimant}")
    if rating:
        snippet_parts.append(f"Rating: {rating}")
    snippet_parts.append(f"Checked by: {publisher}")
    if title_snippet:
        snippet_parts.append(title_snippet)

    # Add additional reviews if available
    if len(reviews) > 1:
        other_ratings = []
        for rev in reviews[1:3]:  # up to 2 additional
            r_pub = rev.get("publisher_name", "")
            r_rating = rev.get("rating", "")
            if r_pub and r_rating:
                other_ratings.append(f"{r_pub}: {r_rating}")
        if other_ratings:
            snippet_parts.append(f"Also checked: {'; '.join(other_ratings)}")

    snippet = " | ".join(snippet_parts)

    return {
        "url": url,
        "title": title,
        "source_name": "google_factcheck",
        "evidence_type": "factcheck",
        "snippet": snippet[:2000],
    }
