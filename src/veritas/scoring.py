"""Evidence scoring engine — computes how well an evidence result matches a claim.

All scoring is deterministic and explainable:
  - token overlap between claim and evidence title/snippet
  - boosts for named entities, numbers, dates matching
  - boosts for category-relevant terms
  - exact numeric match (big boost for EDGAR enriched snippets)
  - penalty for generic/vague titles
  - every signal that fires is logged for transparency
"""

from __future__ import annotations
import re
import string
from typing import Dict, Any, List, Tuple


# ---------------------------------------------------------------------------
# Normalisation (shared with claim_extract but independent)
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    t = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(t.split())


def _tokenize(text: str) -> set[str]:
    return set(_normalise(text).split())


# ---------------------------------------------------------------------------
# Number extraction helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r'\d+')

# Extract financial numbers with decimals (e.g., "113.8", "31.6", "2.82")
_FINANCIAL_NUM_RE = re.compile(r'\d+(?:\.\d+)?')


def _extract_claim_numbers(text: str) -> set[str]:
    """Extract significant numbers from claim text.

    Returns decimal strings like {"113.8", "403", "17", "31.6", "2.82"}.
    Filters out trivially common numbers (single digits, years handled separately).
    """
    nums = set(_FINANCIAL_NUM_RE.findall(text))
    # Remove very short/common numbers that match too loosely
    return {n for n in nums if len(n) >= 2 or float(n) >= 10}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_GENERIC_TITLES = frozenset([
    "introduction", "abstract", "summary", "chapter", "section", "appendix",
    "editorial", "letter", "comment", "reply", "correction", "erratum",
    "podcast", "episode", "transcript", "interview",
])


def score_evidence(
    claim_text: str,
    claim_category: str,
    evidence_title: str,
    evidence_snippet: str,
    evidence_type: str,
    source_name: str,
) -> Tuple[int, str]:
    """Score how well an evidence result matches a claim.

    Returns (score 0-100, pipe-delimited signals).
    """
    signals: List[str] = []
    score = 0

    claim_tokens = _tokenize(claim_text)
    title_tokens = _tokenize(evidence_title)
    snippet_tokens = _tokenize(evidence_snippet) if evidence_snippet else set()
    evidence_tokens = title_tokens | snippet_tokens

    # 1. Token overlap (0-30 points)
    if claim_tokens and evidence_tokens:
        overlap = claim_tokens & evidence_tokens
        overlap_ratio = len(overlap) / max(len(claim_tokens), 1)
        token_score = min(30, int(overlap_ratio * 60))  # 50% overlap = 30 pts
        if token_score > 0:
            score += token_score
            signals.append(f"token_overlap:{len(overlap)}")

    # 2. Named entity / proper noun match (0-15 points)
    claim_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim_text))
    if claim_entities:
        title_upper = evidence_title + " " + evidence_snippet
        matched_entities = [e for e in claim_entities if e.lower() in title_upper.lower()]
        if matched_entities:
            entity_score = min(15, len(matched_entities) * 5)
            score += entity_score
            signals.append(f"entity_match:{','.join(matched_entities[:3])}")

    # 3. Number match (0-10 points for basic, 0-20 for exact financial match)
    claim_nums = set(_NUM_RE.findall(claim_text))
    evidence_nums = set(_NUM_RE.findall(evidence_title + " " + evidence_snippet))
    if claim_nums and evidence_nums:
        matched_nums = claim_nums & evidence_nums
        if matched_nums:
            score += min(10, len(matched_nums) * 5)
            signals.append(f"number_match:{','.join(list(matched_nums)[:3])}")

    # 3b. Exact financial number match — big boost for enriched filing snippets
    #     Matches decimal numbers like "113.8", "31.6", "2.82"
    if evidence_snippet and len(evidence_snippet) > 200:
        claim_financial_nums = _extract_claim_numbers(claim_text)
        snippet_financial_nums = set(_FINANCIAL_NUM_RE.findall(evidence_snippet))
        exact_matches = claim_financial_nums & snippet_financial_nums
        if exact_matches:
            exact_boost = min(20, len(exact_matches) * 8)
            score += exact_boost
            signals.append(f"number_exact_match:{','.join(sorted(exact_matches)[:4])}")

    # 4. Category relevance boost (0-10 points)
    _CAT_TERMS = {
        "finance": {"rate", "inflation", "gdp", "economy", "market", "fiscal",
                     "monetary", "bank", "revenue", "revenues", "income",
                     "earnings", "margin", "operating", "cash", "flow",
                     "cap", "price", "eps", "dividend", "ratio", "stock",
                     "shares", "valuation", "profit", "quarterly"},
        "tech": {"ai", "model", "gpu", "software", "algorithm", "computing", "neural"},
        "health": {"health", "drug", "vaccine", "clinical", "patient", "disease", "treatment"},
        "science": {"research", "study", "climate", "energy", "species", "experiment"},
        "politics": {"vote", "election", "congress", "senate", "legislation", "policy"},
        "military": {"military", "defense", "weapon", "security", "intelligence"},
    }
    cat_terms = _CAT_TERMS.get(claim_category, set())
    if cat_terms and evidence_tokens:
        cat_overlap = cat_terms & evidence_tokens
        if cat_overlap:
            score += min(10, len(cat_overlap) * 3)
            signals.append(f"category_match:{claim_category}")

    # 5. Evidence type boost (0-15 points)
    if evidence_type in ("paper", "filing", "gov", "dataset"):
        score += 15
        signals.append(f"primary_source:{evidence_type}")
    elif evidence_type == "secondary":
        score += 5
        signals.append("secondary_source")

    # 6. Keyphrase match — multi-word sequences (0-10 points)
    claim_bigrams = _bigrams(claim_text)
    evidence_bigrams = _bigrams(evidence_title + " " + evidence_snippet)
    keyphrase_matches = claim_bigrams & evidence_bigrams
    if keyphrase_matches:
        score += min(10, len(keyphrase_matches) * 5)
        signals.append(f"keyphrase_hit:{len(keyphrase_matches)}")

    # 7. Generic title penalty (-10 points)
    title_lower_words = set(evidence_title.lower().split())
    if title_lower_words & _GENERIC_TITLES and len(title_lower_words) < 5:
        score = max(0, score - 10)
        signals.append("generic_title_penalty")

    # Clamp
    score = min(100, max(0, score))

    return score, "|".join(signals)


def _bigrams(text: str) -> set[str]:
    words = _normalise(text).split()
    return {f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)}


# ---------------------------------------------------------------------------
# Finance claim typing
# ---------------------------------------------------------------------------

_NUMERIC_KPI_TERMS = frozenset([
    "revenue", "revenues", "income", "earnings", "margin", "margins",
    "billion", "million", "percent", "eps", "capex", "depreciation",
    "cash flow", "free cash flow", "operating income", "net income",
    "growth", "dividend", "repurchase", "backlog",
])

_GUIDANCE_TERMS = frozenset([
    "expect", "expects", "expected", "outlook", "guidance",
    "forecast", "forecasts", "will", "anticipate", "anticipates",
    "anticipated", "forward-looking", "estimate", "estimates",
    "project", "projected", "plan", "plans", "intend", "intends",
])


def classify_finance_claim(claim_text: str) -> str:
    """Classify a finance claim as numeric_kpi, guidance, or other.

    Only numeric_kpi claims are eligible for AUTO PARTIAL/SUPPORTED.
    Guidance claims stay UNKNOWN by design.
    """
    lower = claim_text.lower()
    has_number = bool(re.search(r'\d', lower))

    # Check for numeric KPI: must have numbers + financial terms
    if has_number:
        for term in _NUMERIC_KPI_TERMS:
            if term in lower:
                # Also check for guidance terms — guidance with numbers is still guidance
                for g in _GUIDANCE_TERMS:
                    if g in lower:
                        return "guidance"
                return "numeric_kpi"

    # Check for guidance language (without numbers)
    for g in _GUIDANCE_TERMS:
        if g in lower:
            return "guidance"

    return "other"


# ---------------------------------------------------------------------------
# Auto-status guardrails
# ---------------------------------------------------------------------------

def compute_auto_status(
    best_score: int,
    best_evidence_type: str,
    best_signals: str,
    claim_confidence: str,
    finance_claim_type: str = "",
) -> Tuple[str, float]:
    """Determine auto verification status from best evidence score.

    Returns (status_auto, auto_confidence).

    Guardrails:
      - SUPPORTED only if primary source + score >= 85 + token_overlap + keyphrase_hit
      - PARTIAL if score 70-84
      - CONTRADICTED: never in v1
      - Guidance finance claims: always UNKNOWN
      - Everything else: UNKNOWN
    """
    # Hard guardrail: guidance claims never get auto-labeled
    if finance_claim_type == "guidance":
        return "unknown", best_score / 100.0

    if best_score < 70:
        return "unknown", best_score / 100.0

    signal_set = set(best_signals.split("|"))
    has_token_overlap = any(s.startswith("token_overlap") for s in signal_set)
    has_keyphrase = any(s.startswith("keyphrase_hit") for s in signal_set)
    has_exact_number = any(s.startswith("number_exact_match") for s in signal_set)
    is_primary = best_evidence_type in ("paper", "filing", "gov", "dataset")

    # SUPPORTED: strict guardrails
    # For filings with exact number matches, keyphrase is less critical
    if best_score >= 85 and is_primary and has_token_overlap:
        if has_keyphrase or has_exact_number:
            return "supported", best_score / 100.0

    # PARTIAL: moderate confidence
    if 70 <= best_score < 85:
        return "partial", best_score / 100.0

    # High score but missing some signals -> partial
    if best_score >= 85 and not (is_primary and (has_keyphrase or has_exact_number)):
        return "partial", best_score / 100.0

    return "unknown", best_score / 100.0
