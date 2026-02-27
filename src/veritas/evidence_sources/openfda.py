"""OpenFDA — verify health claims against FDA drug/device data.

Uses the free OpenFDA API (no key required).
Endpoints: drug adverse events, drug labeling, recalls, enforcement.
Rate limit: 40 req/min without key.

Docs: https://open.fda.gov/apis/
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query


_BASE_URL = "https://api.fda.gov"

# Map claim keywords to the best OpenFDA endpoint
_ENDPOINTS = {
    "adverse": "/drug/event.json",
    "side effect": "/drug/event.json",
    "recall": "/food/enforcement.json",
    "recalled": "/food/enforcement.json",
    "approved": "/drug/drugsfda.json",
    "approval": "/drug/drugsfda.json",
    "fda approved": "/drug/drugsfda.json",
    "label": "/drug/label.json",
    "warning": "/drug/label.json",
}


def _pick_endpoint(claim_text: str) -> str:
    """Choose the best OpenFDA endpoint based on claim content."""
    lower = claim_text.lower()
    for kw, endpoint in _ENDPOINTS.items():
        if kw in lower:
            return endpoint
    # Default: drug adverse events (largest dataset)
    return "/drug/event.json"


def _extract_drug_name(claim_text: str) -> str:
    """Try to extract a drug or substance name from claim text."""
    # Common drug names pattern: capitalized word that's likely a drug
    # This is a basic heuristic — pharma names are often capitalized proper nouns
    lower = claim_text.lower()
    drug_indicators = [
        "drug", "medication", "medicine", "pharmaceutical", "treatment",
        "therapy", "prescribed", "prescription", "dose", "dosage",
    ]
    has_drug_context = any(ind in lower for ind in drug_indicators)
    if not has_drug_context:
        return ""

    # Extract capitalized words that might be drug names
    candidates = re.findall(r'\b([A-Z][a-z]{3,}(?:in|ol|ide|ine|ate|one|an|ax|il|ar)?)\b', claim_text)
    if candidates:
        return candidates[0]
    return ""


def search_openfda(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search OpenFDA for drug/device data matching a health claim.

    Standard evidence source signature.
    """
    endpoint = _pick_endpoint(claim_text)
    query = build_search_query(claim_text, max_terms=4)
    if not query:
        return []

    # Build search parameter
    search_terms = query.replace('"', '').split()[:4]
    # OpenFDA uses Elasticsearch syntax
    search_param = "+AND+".join(search_terms)

    url = f"{_BASE_URL}{endpoint}"
    resp = rate_limited_get(
        url,
        source_name="openfda",
        params={
            "search": search_param,
            "limit": str(min(max_results, 5)),
        },
        timeout=15.0,
    )
    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    results_list = data.get("results", [])
    if not results_list:
        return []

    results = []
    for item in results_list[:max_results]:
        # Build snippet based on endpoint type
        snippet = ""
        title = "FDA Data"

        if "/drug/event" in endpoint:
            patient = item.get("patient", {})
            reactions = patient.get("reaction", [])
            drugs = patient.get("drug", [])
            reaction_names = [r.get("reactionmeddrapt", "") for r in reactions[:3]]
            drug_names = [d.get("medicinalproduct", "") for d in drugs[:3]]
            title = f"FDA Adverse Event Report"
            snippet = (
                f"Drugs: {', '.join(drug_names)}. "
                f"Reactions: {', '.join(reaction_names)}. "
                f"Serious: {item.get('serious', 'unknown')}. "
                f"Country: {item.get('occurcountry', 'unknown')}."
            )
        elif "/drug/label" in endpoint:
            openfda = item.get("openfda", {})
            brand = openfda.get("brand_name", [""])[0] if openfda.get("brand_name") else ""
            generic = openfda.get("generic_name", [""])[0] if openfda.get("generic_name") else ""
            title = f"FDA Drug Label: {brand or generic}"
            warnings = item.get("warnings", [""])
            snippet = f"Brand: {brand}. Generic: {generic}. "
            if warnings and isinstance(warnings, list):
                snippet += warnings[0][:500]
        elif "/food/enforcement" in endpoint:
            title = f"FDA Enforcement: {item.get('product_description', '')[:80]}"
            snippet = (
                f"Classification: {item.get('classification', '')}. "
                f"Reason: {item.get('reason_for_recall', '')[:300]}. "
                f"Status: {item.get('status', '')}."
            )
        elif "/drug/drugsfda" in endpoint:
            products = item.get("products", [])
            if products:
                p = products[0]
                title = f"FDA Approval: {p.get('brand_name', '')}"
                snippet = (
                    f"Brand: {p.get('brand_name', '')}. "
                    f"Active: {p.get('active_ingredients', '')}. "
                    f"Route: {p.get('route', '')}. "
                    f"Dosage: {p.get('dosage_form', '')}."
                )

        if snippet:
            results.append({
                "url": f"https://open.fda.gov/apis/{endpoint.split('/')[1]}/{endpoint.split('/')[2].replace('.json', '')}/",
                "title": title[:200],
                "source_name": "openfda",
                "evidence_type": "gov",
                "snippet": snippet[:2000],
            })

    return results[:max_results]
