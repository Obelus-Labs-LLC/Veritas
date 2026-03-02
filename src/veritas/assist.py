"""Assisted verification — auto-discover evidence for claims using free APIs.

Pipeline for each claim:
  1. Generate search queries from claim text
  2. Hit free structured APIs (15 sources: Crossref, arXiv, PubMed, SEC EDGAR,
     yfinance, Wikipedia, FRED, Google Fact Check, OpenFDA, BLS, CBO,
     USASpending, Census, World Bank, PatentsView)
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

# Drug / FDA indicators
_DRUG_TERMS = frozenset({
    "drug", "fda", "adverse", "recall", "recalled", "approved",
    "approval", "pharmaceutical", "side effect", "medication",
    "dosage", "prescription", "label", "warning",
})

# Labor / employment indicators (for BLS)
_LABOR_TERMS = frozenset({
    "jobs", "employment", "unemployment", "labor", "labour",
    "payroll", "payrolls", "wages", "hourly earnings", "workforce",
    "hiring", "layoffs", "quit rate", "job openings",
    "labor force", "participation rate", "nonfarm",
})

# Government spending / budget indicators
_SPENDING_TERMS = frozenset({
    "spending", "budget", "deficit", "surplus", "national debt",
    "federal debt", "appropriation", "entitlement",
    "social security", "medicare", "medicaid", "cbo",
    "congressional budget", "sequestration", "debt ceiling",
    "stimulus", "bailout", "contract", "grant",
    "government spending", "federal spending", "pentagon",
})

# Demographics / census indicators
_DEMOGRAPHICS_TERMS = frozenset({
    "population", "census", "demographic", "demographics",
    "median income", "household income", "poverty", "poverty rate",
    "homeownership", "rent", "uninsured", "health insurance",
    "college", "bachelor", "education attainment",
})

# International / development indicators
_INTERNATIONAL_TERMS = frozenset({
    "gdp", "gni", "global", "world", "international",
    "developing", "developed", "trade", "exports", "imports",
    "foreign aid", "external debt", "gini", "inequality",
    "life expectancy", "infant mortality", "literacy",
    "renewable energy", "co2 emissions", "carbon emissions",
})

# Patent / innovation indicators
_PATENT_TERMS = frozenset({
    "patent", "patents", "patented", "invention", "innovation",
    "intellectual property", "ip", "patent filing",
    "utility patent", "design patent", "trademark",
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

    # Signal 8: Drug / FDA terms → boost openfda
    drug_count = sum(1 for t in _DRUG_TERMS if t in text_lower)
    if drug_count >= 2:
        if "openfda" in boosts:
            boosts["openfda"] += 10
    elif drug_count >= 1:
        if "openfda" in boosts:
            boosts["openfda"] += 5

    # Signal 9: Labor / employment terms → boost bls
    labor_count = sum(1 for t in _LABOR_TERMS if t in text_lower)
    if labor_count >= 2:
        if "bls" in boosts:
            boosts["bls"] += 10
    elif labor_count >= 1:
        if "bls" in boosts:
            boosts["bls"] += 5

    # Signal 10: Budget / spending terms → boost cbo + usaspending
    spending_count = sum(1 for t in _SPENDING_TERMS if t in text_lower)
    if spending_count >= 2:
        if "cbo" in boosts:
            boosts["cbo"] += 10
        if "usaspending" in boosts:
            boosts["usaspending"] += 8
    elif spending_count >= 1:
        if "cbo" in boosts:
            boosts["cbo"] += 5
        if "usaspending" in boosts:
            boosts["usaspending"] += 4

    # Signal 11: Demographics / census terms → boost census
    demo_count = sum(1 for t in _DEMOGRAPHICS_TERMS if t in text_lower)
    if demo_count >= 2:
        if "census" in boosts:
            boosts["census"] += 10
    elif demo_count >= 1:
        if "census" in boosts:
            boosts["census"] += 5

    # Signal 12: International / development terms → boost worldbank
    intl_count = sum(1 for t in _INTERNATIONAL_TERMS if t in text_lower)
    if intl_count >= 2:
        if "worldbank" in boosts:
            boosts["worldbank"] += 10
    elif intl_count >= 1:
        if "worldbank" in boosts:
            boosts["worldbank"] += 5

    # Signal 13: Patent / innovation terms → boost patentsview
    patent_count = sum(1 for t in _PATENT_TERMS if t in text_lower)
    if patent_count >= 1:
        if "patentsview" in boosts:
            boosts["patentsview"] += 8

    # Signal 14: SEC institutional terms → boost sec_gov
    _SEC_TERMS = frozenset({
        "sec", "commission", "enforcement", "examination",
        "registrant", "registration", "registered", "division",
        "transfer agent", "broker-dealer", "investment adviser",
        "municipal advisor", "rulemaking", "whistleblower",
        "filing fee", "investor protection",
    })
    sec_count = sum(1 for t in _SEC_TERMS if t in text_lower)
    if sec_count >= 2:
        if "sec_gov" in boosts:
            boosts["sec_gov"] += 10
        if "sec_edgar" in boosts:
            boosts["sec_edgar"] += 5
    elif sec_count >= 1:
        if "sec_gov" in boosts:
            boosts["sec_gov"] += 8

    # Signal 15: Named entities → boost wikidata (structured entity facts)
    if len(named_entities) >= 1:
        if "wikidata" in boosts:
            boosts["wikidata"] += 8
    # Historical dates (pre-2000) → wikidata is strong for historical facts
    if re.search(r'\b(1[5-9]\d{2})\b', claim_text):
        if "wikidata" in boosts:
            boosts["wikidata"] += 5

    # Signal 16: Academic/research terms → boost semantic_scholar
    if acad_count >= 1:
        if "semantic_scholar" in boosts:
            boosts["semantic_scholar"] += 8
    # Health + science terms also benefit from academic papers
    if health_count >= 1:
        if "semantic_scholar" in boosts:
            boosts["semantic_scholar"] += 4

    # Signal 17: DuckDuckGo — universal fallback for claims with no strong signal
    # If no other source got boosted above 5, give duckduckgo a moderate boost
    max_other_boost = max((v for k, v in boosts.items() if k != "duckduckgo"), default=0)
    if max_other_boost < 5:
        if "duckduckgo" in boosts:
            boosts["duckduckgo"] += 6

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
    # local_dataset is always first — zero latency, highest precision
    priority = {
        "finance": ["local_dataset", "yfinance", "sec_edgar", "sec_gov", "fred", "bls", "cbo", "usaspending", "google_factcheck", "crossref", "wikipedia", "wikidata", "duckduckgo"],
        "health": ["local_dataset", "pubmed", "openfda", "google_factcheck", "crossref", "semantic_scholar", "wikipedia", "wikidata", "duckduckgo"],
        "science": ["local_dataset", "arxiv", "semantic_scholar", "crossref", "pubmed", "worldbank", "wikipedia", "wikidata", "duckduckgo"],
        "tech": ["local_dataset", "arxiv", "crossref", "patentsview", "google_factcheck", "wikipedia", "wikidata", "duckduckgo"],
        "politics": ["local_dataset", "google_factcheck", "sec_gov", "cbo", "usaspending", "crossref", "wikipedia", "wikidata", "duckduckgo"],
        "military": ["local_dataset", "google_factcheck", "usaspending", "crossref", "wikipedia", "wikidata", "duckduckgo"],
        "education": ["local_dataset", "census", "worldbank", "crossref", "google_factcheck", "semantic_scholar", "wikipedia", "wikidata", "duckduckgo"],
        "energy_climate": ["local_dataset", "worldbank", "crossref", "arxiv", "google_factcheck", "wikipedia", "wikidata", "duckduckgo"],
        "labor": ["local_dataset", "bls", "fred", "census", "google_factcheck", "crossref", "wikipedia", "wikidata", "duckduckgo"],
        "general": [
            "local_dataset", "google_factcheck", "wikipedia", "crossref",
            "arxiv", "pubmed", "sec_gov", "sec_edgar", "yfinance", "fred",
            "openfda", "bls", "cbo", "usaspending", "census", "worldbank",
            "patentsview", "wikidata", "duckduckgo", "semantic_scholar",
        ],
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
    upload_date: str = "",
) -> Dict[str, Any]:
    """Run assisted verification for a single claim.

    Args:
        claim: The claim to verify.
        max_per_claim: Max evidence suggestions to keep.
        dry_run: If True, don't write to DB.
        source_entity: Company/entity name from source metadata (for EDGAR query injection).
        upload_date: Source upload date for temporal filtering.

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

    # Temporal context: use claim_date if available, fallback to upload_date
    claim_date = getattr(claim, "claim_date", "") or ""

    # Query each source (smart routing re-ranks based on claim content)
    sources = _select_sources_for_category(claim.category)
    sources = _smart_select_sources(claim.text, claim.category, sources)
    for source_name, search_fn in sources:
        try:
            if source_name == "sec_edgar":
                # Pass entity injection + enrichment + temporal context for EDGAR
                results = search_fn(
                    claim.text,
                    max_results=3,
                    source_entity=source_entity,
                    enrich=True,
                    claim_date=claim_date,
                    upload_date=upload_date,
                )
            elif source_name == "yfinance":
                # Pass temporal context for historical data
                results = search_fn(claim.text, max_results=3, claim_date=claim_date)
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

    # Score all results (with temporal context)
    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for r in all_results:
        s, sigs = score_evidence(
            claim_text=claim.text,
            claim_category=claim.category,
            evidence_title=r.get("title", ""),
            evidence_snippet=r.get("snippet", ""),
            evidence_type=r.get("evidence_type", "other"),
            source_name=r.get("source_name", ""),
            claim_date=claim_date,
            evidence_date=r.get("evidence_date", ""),
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


# ------------------------------------------------------------------
# Claim verifiability scoring — prioritize claims worth checking
# ------------------------------------------------------------------

def _verifiability_score(claim: "Claim") -> int:
    """Score how likely a claim is to be verifiable against evidence sources.

    Higher score = more verifiable = should be checked first.
    Claims with specific numbers + named entities score highest.
    Personal anecdotes, opinions, and vague narrative score lowest.

    Returns 0-100.
    """
    text = claim.text
    text_lower = text.lower()
    score = 0

    # Numbers are the strongest verifiability signal
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    significant_numbers = [n for n in numbers if len(n) >= 2 or float(n) >= 10]
    if significant_numbers:
        score += min(30, len(significant_numbers) * 10)

    # Named entities (multi-word proper nouns)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    if entities:
        score += min(20, len(entities) * 10)

    # Single proper nouns (capitalized words not at sentence start)
    words = text.split()
    proper_nouns = [w for i, w in enumerate(words)
                    if i > 0 and w[0].isupper() and w.isalpha() and len(w) > 2]
    if proper_nouns:
        score += min(10, len(proper_nouns) * 3)

    # Acronyms (2-5 uppercase letters like SEC, GDP, FDA, CPI, EDGAR)
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', text)
    _SKIP_ACRONYMS = {"I", "A", "THE", "AND", "BUT", "FOR", "NOT"}
    acronyms = [a for a in acronyms if a not in _SKIP_ACRONYMS]
    if acronyms:
        score += min(10, len(acronyms) * 5)

    # Financial/scientific/measurable terms
    _MEASURABLE = frozenset({
        "percent", "billion", "million", "thousand", "trillion",
        "revenue", "income", "earnings", "growth", "rate",
        "population", "gdp", "inflation", "unemployment",
        "patients", "deaths", "cases", "study", "data",
    })
    measurable_count = sum(1 for t in _MEASURABLE if t in text_lower)
    score += min(15, measurable_count * 5)

    # Date references boost verifiability
    if re.search(r'\b(19|20)\d{2}\b', text):
        score += 10

    # Currency symbols
    if re.search(r'[$€£¥]', text):
        score += 10

    # Penalty: personal/opinion language
    _PERSONAL = frozenset({
        "i think", "i believe", "in my opinion", "i feel",
        "my experience", "for me", "personally", "i guess",
        "probably", "maybe", "kind of", "sort of",
        "seems like", "looks like", "i would say",
    })
    for phrase in _PERSONAL:
        if phrase in text_lower:
            score -= 15
            break

    # Penalty: vague/narrative without specifics
    if not significant_numbers and not entities and not proper_nouns:
        score -= 20

    return max(0, min(100, score))


def assist_source(
    source_id: str,
    max_per_claim: int = 5,
    budget_minutes: int = 0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run assisted verification for all claims in a source.

    Fetches source metadata for entity injection (company name for EDGAR queries).
    Claims are sorted by verifiability score — most verifiable claims processed first.
    budget_minutes=0 means no budget limit (process all claims).

    Returns aggregate report.
    """
    claims = db.get_claims_for_source(source_id)
    if not claims:
        raise ValueError(f"No claims found for source '{source_id}'. Run `veritas claims` first.")

    # Infer entity and temporal context from source metadata
    source = db.get_source(source_id)
    source_entity = ""
    upload_date = ""
    source_title = ""
    source_channel = ""
    if source:
        source_entity = infer_source_entity(source.title, source.channel)
        upload_date = source.upload_date or ""
        source_title = source.title or ""
        source_channel = source.channel or ""

    # Re-categorise "general" claims using source context (fixes claims
    # extracted before context-aware categorisation was added).
    if source_title or source_channel:
        from .claim_extract import _classify_category
        recategorised = 0
        for claim in claims:
            if claim.category == "general":
                new_cat = _classify_category(claim.text, source_title, source_channel)
                if new_cat != "general":
                    claim.category = new_cat
                    recategorised += 1
                    if not dry_run:
                        db.update_claim_category(claim.id, new_cat)

    # Clear previous suggestions for this source
    if not dry_run:
        db.delete_suggestions_for_source(source_id)

    # Sort claims by verifiability score (most verifiable first)
    scored_claims = [(claim, _verifiability_score(claim)) for claim in claims]
    scored_claims.sort(key=lambda x: x[1], reverse=True)

    # Count claims below verifiability threshold
    low_verifiability = sum(1 for _, v in scored_claims if v < 10)

    start_time = time.time()
    deadline = start_time + (budget_minutes * 60) if budget_minutes > 0 else float("inf")

    total_suggestions = 0
    total_stored = 0
    auto_supported = 0
    auto_partial = 0
    skipped_low_verifiability = 0
    claim_reports: List[Dict[str, Any]] = []

    for claim, v_score in scored_claims:
        # Check budget
        if time.time() > deadline:
            break

        # Skip very low verifiability claims (score < 5) to save time
        # These are personal anecdotes, opinions, vague narrative
        if v_score < 5:
            skipped_low_verifiability += 1
            continue

        report = assist_claim(
            claim,
            max_per_claim=max_per_claim,
            dry_run=dry_run,
            source_entity=source_entity,
            upload_date=upload_date,
        )
        report["verifiability_score"] = v_score
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
        "claims_skipped_low_verifiability": skipped_low_verifiability,
        "total_suggestions_found": total_suggestions,
        "total_suggestions_stored": total_stored,
        "auto_supported": auto_supported,
        "auto_partial": auto_partial,
        "auto_unknown": len(claim_reports) - auto_supported - auto_partial,
        "elapsed_seconds": round(elapsed, 1),
        "budget_minutes": budget_minutes,
        "dry_run": dry_run,
        "claim_reports": claim_reports,
    }
