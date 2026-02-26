"""Assisted verification — auto-discover evidence for claims using free APIs.

Pipeline for each claim:
  1. Generate search queries from claim text
  2. Hit free structured APIs (Crossref, arXiv, PubMed, SEC EDGAR, yfinance, Wikipedia, FRED, Google Fact Check)
  3. Score each result against the claim
  4. Store top N as evidence_suggestions
  5. Optionally set status_auto if guardrails pass
"""

from __future__ import annotations
import re
import time
from typing import List, Dict, Any, Tuple

from .models import Claim, EvidenceSuggestion, new_id
from .scoring import score_evidence, compute_auto_status, classify_finance_claim
from .evidence_sources import ALL_SOURCES
from .evidence_sources.sec_edgar import infer_source_entity
from .evidence_sources.yfinance_source import _TICKER_MAP
from . import db


# ------------------------------------------------------------------
# Entity / company detection helpers
# ------------------------------------------------------------------

# Extended aliases beyond _TICKER_MAP (these map to display names, not tickers)
_ENTITY_ALIASES: Dict[str, str] = {
    "openai": "OpenAI",
    "open ai": "OpenAI",
    "deepmind": "DeepMind",
    "stripe": "Stripe",
    "spacex": "SpaceX",
    "bytedance": "ByteDance",
    "tiktok": "TikTok",
    "twitter": "Twitter",
    "x corp": "X Corp",
}


def _has_company_mention(text_lower: str) -> bool:
    """Check if text mentions a known company name or ticker.

    Returns True if any company from _TICKER_MAP or _ENTITY_ALIASES is found.
    """
    # Check _TICKER_MAP keys (already lowercase)
    for name in _TICKER_MAP:
        if name in text_lower:
            return True
    # Check _ENTITY_ALIASES keys (already lowercase)
    for name in _ENTITY_ALIASES:
        if name in text_lower:
            return True
    return False


# Academic / research indicators
_ACADEMIC_TERMS = frozenset({
    "study", "studies", "research", "researchers", "published",
    "journal", "peer-reviewed", "paper", "findings", "experiment",
    "hypothesis", "methodology", "statistical", "sample size",
    "correlation", "causation", "meta-analysis", "systematic review",
    "university", "professor", "phd",
})

# Health / medical indicators
_HEALTH_TERMS = frozenset({
    "patients", "clinical", "trial", "trials", "treatment",
    "therapy", "diagnosis", "symptoms", "disease", "drug",
    "fda", "vaccine", "mortality", "survival", "dosage",
    "placebo", "double-blind", "randomized",
})

# Financial metric indicators (beyond just "finance" category)
_FINANCIAL_METRIC_TERMS = frozenset({
    "revenue", "revenues", "earnings", "income", "profit",
    "margin", "eps", "p/e", "pe ratio", "market cap",
    "stock price", "share price", "dividend", "valuation",
    "billion", "million", "quarter", "quarterly", "annual",
    "growth rate", "operating", "capex", "cash flow",
    "balance sheet", "debt", "equity", "ipo",
})


_MACRO_TERMS = frozenset({
    "gdp", "inflation", "unemployment", "interest rate",
    "federal reserve", "monetary policy", "fiscal policy",
    "recession", "cpi", "consumer price", "trade deficit",
    "national debt", "federal debt", "money supply",
    "treasury", "mortgage rate", "housing starts",
})


def _smart_select_sources(
    claim_text: str,
    category: str,
    category_sources: List[Tuple[str, Any]],
) -> List[Tuple[str, Any]]:
    """Re-rank sources based on claim content for smarter API selection.

    Takes the category-ordered sources and re-ranks based on content signals:
      - Company mentions → boost yfinance to top
      - Academic language → boost arxiv
      - Health terms → boost pubmed
      - Financial metrics → boost yfinance + sec_edgar
      - Macroeconomic terms → boost fred
      - Named entities (people, orgs) → boost wikipedia

    Returns reordered list (same sources, different order).
    """
    text_lower = claim_text.lower()
    source_dict = {name: fn for name, fn in category_sources}

    # Build boost scores for each source
    boosts: Dict[str, int] = {name: 0 for name, _ in category_sources}

    # Signal 1: Company mention → strongly boost yfinance + wikipedia
    if _has_company_mention(text_lower):
        if "yfinance" in boosts:
            boosts["yfinance"] += 10
        if "sec_edgar" in boosts:
            boosts["sec_edgar"] += 5
        if "wikipedia" in boosts:
            boosts["wikipedia"] += 4

    # Signal 2: Financial metrics → boost yfinance + sec_edgar
    fin_count = sum(1 for t in _FINANCIAL_METRIC_TERMS if t in text_lower)
    if fin_count >= 2:
        if "yfinance" in boosts:
            boosts["yfinance"] += 8
        if "sec_edgar" in boosts:
            boosts["sec_edgar"] += 4

    # Signal 3: Academic language → boost arxiv
    acad_count = sum(1 for t in _ACADEMIC_TERMS if t in text_lower)
    if acad_count >= 2:
        if "arxiv" in boosts:
            boosts["arxiv"] += 8
        if "crossref" in boosts:
            boosts["crossref"] += 4

    # Signal 4: Health terms → boost pubmed
    health_count = sum(1 for t in _HEALTH_TERMS if t in text_lower)
    if health_count >= 2:
        if "pubmed" in boosts:
            boosts["pubmed"] += 8

    # Signal 5: Macroeconomic terms → boost fred
    macro_count = sum(1 for t in _MACRO_TERMS if t in text_lower)
    if macro_count >= 1:
        if "fred" in boosts:
            boosts["fred"] += 10
    # Also boost fred for claims with "percent" + economic context
    if "percent" in text_lower and macro_count >= 1:
        if "fred" in boosts:
            boosts["fred"] += 5

    # Signal 6: Named entities (people, places, orgs) → boost wikipedia
    # Wikipedia is the best source for entity-level fact verification
    import re
    named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', claim_text)
    if len(named_entities) >= 1:
        if "wikipedia" in boosts:
            boosts["wikipedia"] += 6

    # Signal 7: Political / public-figure claims → boost google_factcheck
    # Fact-checkers cover political statements, viral claims, health misinformation
    _FACTCHECK_TERMS = frozenset({
        "president", "congress", "senator", "representative",
        "government", "administration", "white house", "campaign",
        "claimed", "claim", "said", "says", "stated", "according",
        "false", "true", "misleading", "debunked",
        "unemployment", "crime", "border", "immigration",
        "vaccine", "covid", "pandemic",
    })
    fc_count = sum(1 for t in _FACTCHECK_TERMS if t in text_lower)
    if fc_count >= 2:
        if "google_factcheck" in boosts:
            boosts["google_factcheck"] += 10
    elif fc_count >= 1:
        if "google_factcheck" in boosts:
            boosts["google_factcheck"] += 5

    # Re-rank: sort by (boost descending, then preserve original order)
    indexed = [(name, fn, boosts.get(name, 0), i)
               for i, (name, fn) in enumerate(category_sources)]
    indexed.sort(key=lambda x: (-x[2], x[3]))

    return [(name, fn) for name, fn, _, _ in indexed]


# ------------------------------------------------------------------
# Category-based source selection
# ------------------------------------------------------------------

def _select_sources_for_category(category: str) -> List[Tuple[str, Any]]:
    """Choose which API sources to query based on claim category.

    All sources are tried for 'general'; category-specific sources are prioritised.
    """
    # Map categories to preferred sources
    priority = {
        "finance": ["yfinance", "sec_edgar", "fred", "google_factcheck", "crossref", "wikipedia"],
        "health": ["pubmed", "google_factcheck", "crossref", "wikipedia"],
        "science": ["arxiv", "crossref", "pubmed", "wikipedia"],
        "tech": ["arxiv", "crossref", "google_factcheck", "wikipedia"],
        "politics": ["google_factcheck", "crossref", "wikipedia"],
        "military": ["google_factcheck", "crossref", "wikipedia"],
        "general": ["google_factcheck", "wikipedia", "crossref", "arxiv"],
    }
    preferred = priority.get(category, ["crossref"])

    # Reorder ALL_SOURCES so preferred ones come first
    source_dict = {name: fn for name, fn in ALL_SOURCES}
    ordered = []
    for name in preferred:
        if name in source_dict:
            ordered.append((name, source_dict[name]))
    # Add remaining sources
    for name, fn in ALL_SOURCES:
        if name not in preferred:
            ordered.append((name, fn))

    return ordered


def assist_claim(
    claim: Claim,
    max_per_claim: int = 5,
    dry_run: bool = False,
    source_entity: str = "",
) -> Dict[str, Any]:
    """Run assisted verification for a single claim.

    Args:
        claim: The claim to verify.
        max_per_claim: Max evidence suggestions to keep.
        dry_run: If True, don't write to DB.
        source_entity: Company/entity name from source metadata (for EDGAR query injection).

    Returns a report dict with:
      - suggestions_found: int
      - suggestions_stored: int
      - status_auto: str
      - auto_confidence: float
      - best_score: int
      - finance_claim_type: str (for finance claims)
    """
    all_results: List[Dict[str, Any]] = []

    # Classify finance claims for guardrails
    finance_claim_type = ""
    if claim.category == "finance":
        finance_claim_type = classify_finance_claim(claim.text)

    # Query each source (smart routing re-ranks based on claim content)
    sources = _select_sources_for_category(claim.category)
    sources = _smart_select_sources(claim.text, claim.category, sources)
    for source_name, search_fn in sources:
        try:
            if source_name == "sec_edgar":
                # Pass entity injection + enrichment for EDGAR
                results = search_fn(
                    claim.text,
                    max_results=3,
                    source_entity=source_entity,
                    enrich=True,
                )
            else:
                results = search_fn(claim.text, max_results=3)
            for r in results:
                r["_source_name"] = source_name
            all_results.extend(results)
        except Exception:
            continue  # skip failed sources silently

    if not all_results:
        return {
            "suggestions_found": 0,
            "suggestions_stored": 0,
            "status_auto": "unknown",
            "auto_confidence": 0.0,
            "best_score": 0,
            "finance_claim_type": finance_claim_type,
        }

    # Score all results
    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for r in all_results:
        s, sigs = score_evidence(
            claim_text=claim.text,
            claim_category=claim.category,
            evidence_title=r.get("title", ""),
            evidence_snippet=r.get("snippet", ""),
            evidence_type=r.get("evidence_type", "other"),
            source_name=r.get("source_name", ""),
        )
        scored.append((s, sigs, r))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_per_claim]

    # Build EvidenceSuggestion objects
    suggestions = []
    for s, sigs, r in top:
        if s < 5:  # skip zero/near-zero scores
            continue
        suggestions.append(EvidenceSuggestion(
            id=new_id(),
            claim_id=claim.id,
            url=r.get("url", ""),
            title=r.get("title", ""),
            source_name=r.get("source_name", ""),
            evidence_type=r.get("evidence_type", "other"),
            score=s,
            signals=sigs,
            snippet=r.get("snippet", "")[:200],
        ))

    # Store suggestions
    stored = 0
    if suggestions and not dry_run:
        stored = db.insert_evidence_suggestions(suggestions)

    # Compute auto status from best evidence
    best_score = top[0][0] if top else 0
    best_signals = top[0][1] if top else ""
    best_type = top[0][2].get("evidence_type", "other") if top else "other"
    status_auto, auto_confidence = compute_auto_status(
        best_score, best_type, best_signals, claim.confidence_language,
        finance_claim_type=finance_claim_type,
    )

    # Update claim auto status
    if not dry_run and status_auto != "unknown":
        db.update_claim_auto_status(claim.id, status_auto, auto_confidence)

    return {
        "suggestions_found": len(all_results),
        "suggestions_stored": stored,
        "status_auto": status_auto,
        "auto_confidence": auto_confidence,
        "best_score": best_score,
        "finance_claim_type": finance_claim_type,
    }


def assist_source(
    source_id: str,
    max_per_claim: int = 5,
    budget_minutes: int = 10,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run assisted verification for all claims in a source.

    Fetches source metadata for entity injection (company name for EDGAR queries).
    Returns aggregate report.
    """
    claims = db.get_claims_for_source(source_id)
    if not claims:
        raise ValueError(f"No claims found for source '{source_id}'. Run `veritas claims` first.")

    # Infer entity from source metadata for EDGAR query injection
    source = db.get_source(source_id)
    source_entity = ""
    if source:
        source_entity = infer_source_entity(source.title, source.channel)

    # Clear previous suggestions for this source
    if not dry_run:
        db.delete_suggestions_for_source(source_id)

    start_time = time.time()
    deadline = start_time + (budget_minutes * 60)

    total_suggestions = 0
    total_stored = 0
    auto_supported = 0
    auto_partial = 0
    claim_reports: List[Dict[str, Any]] = []

    for claim in claims:
        # Check budget
        if time.time() > deadline:
            break

        report = assist_claim(
            claim,
            max_per_claim=max_per_claim,
            dry_run=dry_run,
            source_entity=source_entity,
        )
        claim_reports.append({
            "claim_id": claim.id,
            "text_excerpt": claim.text[:80],
            "category": claim.category,
            **report,
        })

        total_suggestions += report["suggestions_found"]
        total_stored += report["suggestions_stored"]
        if report["status_auto"] == "supported":
            auto_supported += 1
        elif report["status_auto"] == "partial":
            auto_partial += 1

    elapsed = time.time() - start_time

    return {
        "source_id": source_id,
        "source_entity": source_entity,
        "claims_processed": len(claim_reports),
        "claims_total": len(claims),
        "total_suggestions_found": total_suggestions,
        "total_suggestions_stored": total_stored,
        "auto_supported": auto_supported,
        "auto_partial": auto_partial,
        "auto_unknown": len(claim_reports) - auto_supported - auto_partial,
        "elapsed_seconds": round(elapsed, 1),
        "dry_run": dry_run,
        "claim_reports": claim_reports,
    }
