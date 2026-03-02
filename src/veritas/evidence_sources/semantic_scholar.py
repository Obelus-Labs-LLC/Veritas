"""Semantic Scholar API — academic paper search across all disciplines.

Uses the Semantic Scholar Academic Graph API (free, no key for basic use):
  - Paper search with relevance ranking
  - 200M+ papers across all fields (far broader than arXiv alone)

Rate limit: 100 requests per 5 minutes without API key.
We use a 3-second interval to stay well within limits.

Ideal for: scientific claims, medical research, social science,
economics studies, education research, any claim citing studies.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

from .base import rate_limited_get, build_search_query

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# Broader academic relevance check than arXiv — includes health, social science
_ACADEMIC_TERMS = frozenset({
    "study", "studies", "research", "researchers", "published",
    "journal", "paper", "findings", "experiment", "experiments",
    "hypothesis", "methodology", "statistical", "sample",
    "correlation", "causation", "meta-analysis", "systematic review",
    "university", "professor", "phd", "peer-reviewed", "evidence",
    # Health/medical
    "patients", "clinical", "trial", "trials", "treatment",
    "therapy", "diagnosis", "disease", "drug", "vaccine",
    "mortality", "survival", "randomized", "placebo",
    "efficacy", "symptom", "symptoms", "infection", "chronic",
    "cancer", "tumor", "diabetes", "cardiovascular",
    # Science / biology
    "theory", "quantum", "physics", "biology", "chemistry",
    "evolution", "genome", "species", "cells", "molecule",
    "astronomy", "climate", "emissions", "energy",
    "gene", "genetic", "dna", "rna", "protein", "enzyme",
    "neuron", "neurons", "synapse", "brain", "neural",
    "crispr", "mutation", "chromosome", "organism",
    "atom", "particle", "photon", "electron", "nucleus",
    "ecosystem", "biodiversity", "habitat", "extinction",
    "temperature", "atmosphere", "ocean", "warming",
    # Social science / economics
    "economic", "inequality", "poverty", "demographic",
    "survey", "census", "behavioral", "cognitive",
    "psychology", "sociological", "longitudinal",
    "percent", "rate", "risk", "factor", "outcome",
    "population", "cohort", "analysis", "data",
})


def _has_academic_relevance(claim_text: str) -> bool:
    """Check if claim has academic/research relevance for Semantic Scholar."""
    lower = claim_text.lower()
    count = sum(1 for t in _ACADEMIC_TERMS if t in lower)
    if count >= 1:
        return True
    # Named entities with numbers often cite research
    has_entity = bool(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim_text))
    has_number = bool(re.search(r'\d{2,}', claim_text))
    if has_entity and has_number:
        return True
    return False


def search_semantic_scholar(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Semantic Scholar for academic papers relevant to a claim."""
    if not _has_academic_relevance(claim_text):
        return []

    query = build_search_query(claim_text, max_terms=10)
    if not query or len(query.strip()) < 5:
        return []

    # Use a longer delay for Semantic Scholar's stricter rate limits
    import time
    from .base import _LAST_REQUEST
    now = time.time()
    last = _LAST_REQUEST.get("semantic_scholar", 0)
    extra_wait = 3.0 - (now - last)  # 3-second interval instead of 1
    if extra_wait > 0:
        time.sleep(extra_wait)

    params = {
        "query": query,
        "limit": str(min(max_results, 10)),
        "fields": "title,abstract,year,venue,citationCount,url",
    }

    resp = rate_limited_get(_BASE_URL, source_name="semantic_scholar", params=params)

    # Handle 429 with a single retry after a longer wait
    if resp is None:
        # rate_limited_get returns None on errors including 429
        time.sleep(5.0)
        resp = rate_limited_get(_BASE_URL, source_name="semantic_scholar", params=params)

    if resp is None:
        return []

    try:
        data = resp.json()
    except (ValueError, Exception):
        return []

    papers = data.get("data", [])
    if not papers:
        return []

    results: List[Dict[str, Any]] = []
    for paper in papers[:max_results]:
        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()
        year = paper.get("year", "")
        venue = (paper.get("venue") or "").strip()
        citations = paper.get("citationCount", 0)
        paper_url = paper.get("url", "")
        paper_id = paper.get("paperId", "")

        if not title:
            continue

        # Build a useful URL
        url = paper_url or f"https://www.semanticscholar.org/paper/{paper_id}"

        # Build snippet
        parts = []
        if year:
            parts.append(f"({year})")
        if venue:
            parts.append(f"Published in: {venue}")
        if citations:
            parts.append(f"Citations: {citations}")
        if abstract:
            parts.append(abstract[:1500])
        snippet = " | ".join(parts) if parts else title

        results.append({
            "url": url,
            "title": title,
            "source_name": "semantic_scholar",
            "evidence_type": "paper",
            "snippet": snippet[:2000],
        })

    return results
