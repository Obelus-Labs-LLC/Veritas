"""Wikidata REST API — verify entity-level structured facts.

Uses the Wikidata Action API (free, no key required):
  - wbsearchentities: find entities by name
  - wbgetentities: get structured claims/properties

Ideal for: founding dates, headquarters, key people, population
numbers, financial figures, geographic facts, historical dates.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any, Optional

from .base import rate_limited_get, build_search_query

_API_URL = "https://www.wikidata.org/w/api.php"

# Key Wikidata property IDs we look for during verification
_USEFUL_PROPERTIES: Dict[str, str] = {
    "P571": "inception/founding date",
    "P159": "headquarters location",
    "P1128": "number of employees",
    "P2139": "total revenue",
    "P169": "chief executive officer",
    "P112": "founded by",
    "P17": "country",
    "P1082": "population",
    "P569": "date of birth",
    "P570": "date of death",
    "P27": "country of citizenship",
    "P19": "place of birth",
    "P106": "occupation",
    "P108": "employer",
    "P576": "dissolved/abolished date",
    "P856": "official website",
    "P18": "image",
    "P1566": "GeoNames ID",
    "P625": "coordinate location",
    "P2142": "box office",
    "P577": "publication date",
    "P50": "author",
    "P136": "genre",
}


def _has_entity_relevance(claim_text: str) -> bool:
    """Check if claim contains named entities worth looking up in Wikidata."""
    # Multi-word proper nouns (e.g. "Goldman Sachs", "Albert Einstein")
    if re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim_text):
        return True
    # Acronyms (e.g. SEC, GDP, NASA, NVIDIA)
    skip = {"I", "A", "THE", "AND", "BUT", "FOR", "NOT", "WAS", "HAS"}
    acronyms = [a for a in re.findall(r'\b[A-Z]{2,6}\b', claim_text) if a not in skip]
    if acronyms:
        return True
    # Single capitalized words mid-sentence (not sentence-start)
    words = claim_text.split()
    for i, w in enumerate(words):
        if i > 0 and w[0:1].isupper() and w.isalpha() and len(w) > 2:
            return True
    return False


def _extract_entity_query(claim_text: str) -> str:
    """Extract the most likely entity name from claim text for Wikidata search."""
    # Prefer multi-word proper nouns
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim_text)
    if entities:
        return entities[0]
    # Fall back to acronyms
    skip = {"I", "A", "THE", "AND", "BUT", "FOR", "NOT", "WAS", "HAS"}
    acronyms = [a for a in re.findall(r'\b[A-Z]{2,6}\b', claim_text) if a not in skip]
    if acronyms:
        return acronyms[0]
    # Fall back to first capitalized word mid-sentence
    words = claim_text.split()
    for i, w in enumerate(words):
        if i > 0 and w[0:1].isupper() and w.isalpha() and len(w) > 2:
            return w
    return ""


def _format_value(snak: Dict) -> str:
    """Format a Wikidata snak value into a readable string."""
    dv = snak.get("datavalue", {})
    vtype = dv.get("type", "")
    val = dv.get("value", "")

    if vtype == "string":
        return str(val)
    elif vtype == "time":
        # Format: +1976-04-01T00:00:00Z → 1976-04-01
        t = val.get("time", "")
        return t.lstrip("+").split("T")[0] if t else ""
    elif vtype == "quantity":
        amount = val.get("amount", "")
        unit = val.get("unit", "").split("/")[-1]  # Q-id or "1"
        return f"{amount} ({unit})" if unit != "1" else str(amount)
    elif vtype == "monolingualtext":
        return val.get("text", "")
    elif vtype == "wikibase-entityid":
        return val.get("id", "")
    return str(val)[:100]


def search_wikidata(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Wikidata for structured facts relevant to a claim."""
    if not _has_entity_relevance(claim_text):
        return []

    entity_query = _extract_entity_query(claim_text)
    if not entity_query:
        return []

    # Step 1: Search for entities matching the query
    resp = rate_limited_get(
        _API_URL,
        source_name="wikidata",
        params={
            "action": "wbsearchentities",
            "search": entity_query,
            "language": "en",
            "limit": "3",
            "format": "json",
        },
    )
    if resp is None:
        return []

    try:
        search_data = resp.json()
    except (ValueError, Exception):
        return []

    search_results = search_data.get("search", [])
    if not search_results:
        return []

    # Step 2: Get structured properties for the top entity
    entity_id = search_results[0].get("id", "")
    entity_label = search_results[0].get("label", entity_query)
    entity_desc = search_results[0].get("description", "")

    if not entity_id:
        return []

    resp2 = rate_limited_get(
        _API_URL,
        source_name="wikidata",
        params={
            "action": "wbgetentities",
            "ids": entity_id,
            "languages": "en",
            "props": "claims|descriptions",
            "format": "json",
        },
    )
    if resp2 is None:
        return []

    try:
        entity_data = resp2.json()
    except (ValueError, Exception):
        return []

    entities = entity_data.get("entities", {})
    entity = entities.get(entity_id, {})
    claims_data = entity.get("claims", {})

    # Step 3: Extract useful properties and build snippet
    facts: List[str] = []
    if entity_desc:
        facts.append(f"{entity_label}: {entity_desc}")

    for prop_id, prop_label in _USEFUL_PROPERTIES.items():
        if prop_id in claims_data:
            prop_claims = claims_data[prop_id]
            for pc in prop_claims[:2]:  # Max 2 values per property
                mainsnak = pc.get("mainsnak", {})
                val = _format_value(mainsnak)
                if val:
                    facts.append(f"{prop_label}: {val}")

    if not facts:
        return []

    snippet = "; ".join(facts[:15])  # Cap at 15 facts
    wikidata_url = f"https://www.wikidata.org/wiki/{entity_id}"

    results: List[Dict[str, Any]] = [{
        "url": wikidata_url,
        "title": f"Wikidata: {entity_label}",
        "source_name": "wikidata",
        "evidence_type": "dataset",
        "snippet": snippet[:2000],
    }]

    # Add additional search results as secondary entries
    for sr in search_results[1:max_results]:
        sr_label = sr.get("label", "")
        sr_desc = sr.get("description", "")
        sr_id = sr.get("id", "")
        if sr_label and sr_id:
            results.append({
                "url": f"https://www.wikidata.org/wiki/{sr_id}",
                "title": f"Wikidata: {sr_label}",
                "source_name": "wikidata",
                "evidence_type": "dataset",
                "snippet": f"{sr_label}: {sr_desc}"[:2000],
            })

    return results[:max_results]
