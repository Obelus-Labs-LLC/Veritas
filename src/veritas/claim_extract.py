"""Deterministic claim extraction from transcript segments.

No LLM needed — uses rule-based heuristics:
  1. Build a stitched text window across adjacent segments for context.
  2. Split into sentences at punctuation boundaries.
  3. Keep sentences that contain numbers, dates, named entities (capitalised
     multi-word tokens), or assertion verbs — AND have a subject-like token.
  4. Reject dangling clauses that start with conjunctions.
  5. Classify confidence language (hedged / definitive / unknown).
  6. Deduplicate near-identical claims by normalised text similarity.
"""

from __future__ import annotations
import hashlib
import json
import re
import string
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Tuple

from .config import (
    ASSERTION_VERBS, HEDGE_WORDS, DEFINITIVE_WORDS, DEDUP_THRESHOLD,
)
from .models import Claim, Segment, new_id
from .paths import source_export_dir
from . import db


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MIN_CLAIM_WORDS = 7     # reject fragments shorter than this
MIN_CLAIM_CHARS = 40    # reject very short text
MAX_CLAIM_CHARS = 240   # cap stitched claim length
STITCH_FORWARD = 2      # max segments to look ahead for sentence end
STITCH_BACKWARD = 1     # max segments to look behind for sentence start

# Conjunctions that signal a dangling clause, not a self-contained claim
_DANGLING_STARTS = frozenset([
    "and", "but", "while", "because", "so", "which", "that",
    "or", "nor", "yet", "also", "then", "plus",
])

# Subject-like pronouns that can anchor a claim
_SUBJECT_PRONOUNS = frozenset([
    "it", "they", "we", "he", "she", "i", "you", "this", "that",
    "these", "those", "there", "one",
])


# ------------------------------------------------------------------
# Sentence splitting
# ------------------------------------------------------------------

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    """Rough sentence splitter that works on transcription text."""
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


# ------------------------------------------------------------------
# Segment stitching
# ------------------------------------------------------------------

def _stitch_window(segments: List[Segment], center: int) -> Tuple[str, float, float]:
    """Build a text window around segment *center* by merging adjacent segments.

    Returns (stitched_text, ts_start, ts_end).
    """
    lo = max(0, center - STITCH_BACKWARD)
    hi = min(len(segments) - 1, center + STITCH_FORWARD)

    parts = []
    for i in range(lo, hi + 1):
        parts.append(segments[i].text.strip())

    text = " ".join(parts)
    ts_start = segments[lo].start
    ts_end = segments[hi].end
    return text, ts_start, ts_end


def _extract_sentences_from_window(
    window_text: str,
    center_text: str,
    ts_start: float,
    ts_end: float,
) -> List[Tuple[str, float, float]]:
    """Split stitched window into sentences and keep those overlapping center segment.

    Returns list of (sentence, approx_start, approx_end).
    """
    sentences = _split_sentences(window_text)
    if not sentences:
        # If no sentence boundaries found, use center text directly
        if len(center_text.strip()) > 10:
            return [(center_text.strip(), ts_start, ts_end)]
        return []

    # Keep sentences that overlap with the center segment text
    # (i.e., contain at least some words from the center)
    center_words = set(center_text.lower().split())
    results: List[Tuple[str, float, float]] = []
    window_duration = ts_end - ts_start

    for i, sent in enumerate(sentences):
        sent_words = set(sent.lower().split())
        overlap = len(center_words & sent_words)
        # Require meaningful overlap with center segment
        if overlap < 3 and len(center_words) > 5:
            continue
        if overlap < 2:
            continue

        # Approximate timestamp for this sentence within the window
        if len(sentences) == 1:
            s_start, s_end = ts_start, ts_end
        else:
            slice_dur = window_duration / len(sentences)
            s_start = ts_start + slice_dur * i
            s_end = s_start + slice_dur

        # Truncate if too long
        if len(sent) > MAX_CLAIM_CHARS:
            sent = sent[:MAX_CLAIM_CHARS].rsplit(" ", 1)[0] + "..."

        results.append((sent, round(s_start, 3), round(s_end, 3)))

    return results


# ------------------------------------------------------------------
# Heuristic filters
# ------------------------------------------------------------------

_NUM_RE = re.compile(r'\d')
_DATE_RE = re.compile(
    r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|'
    r'(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2})\b',
    re.IGNORECASE,
)
_NAMED_ENTITY_RE = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')
_CAPITALIZED_RE = re.compile(r'\b[A-Z][a-z]{2,}\b')


def _has_number(s: str) -> bool:
    return bool(_NUM_RE.search(s))


def _has_date(s: str) -> bool:
    return bool(_DATE_RE.search(s))


def _has_named_entity(s: str) -> bool:
    return bool(_NAMED_ENTITY_RE.search(s))


def _has_assertion_verb(s: str) -> bool:
    words = set(s.lower().split())
    return bool(words & ASSERTION_VERBS)


def _has_subject(s: str) -> bool:
    """Check for a subject-like token: capitalized word, pronoun, or number."""
    lower_words = set(s.lower().split())
    # Pronoun subject
    if lower_words & _SUBJECT_PRONOUNS:
        return True
    # Capitalized token (proper noun / named entity)
    if _CAPITALIZED_RE.search(s):
        return True
    # Number as subject (e.g., "65,000 GitHub stars...")
    if _has_number(s):
        return True
    return False


def _starts_with_conjunction(s: str) -> bool:
    """Return True if the sentence starts with a dangling conjunction."""
    first_word = s.strip().split()[0].lower().rstrip(",") if s.strip() else ""
    return first_word in _DANGLING_STARTS


def _collect_signals(sentence: str) -> List[str]:
    """Return list of signal names that fired for this sentence."""
    signals: List[str] = []
    if _has_number(sentence):
        signals.append("number")
    if _has_date(sentence):
        signals.append("date")
    if _has_named_entity(sentence):
        signals.append("named_entity")
    if _has_assertion_verb(sentence):
        signals.append("assertion_verb")
    if _has_subject(sentence):
        signals.append("has_subject")
    return signals


def _is_candidate(sentence: str) -> bool:
    """Return True if sentence looks like a checkable, self-contained claim."""
    # Must have at least one claim-signal
    has_signal = (
        _has_number(sentence)
        or _has_date(sentence)
        or _has_named_entity(sentence)
        or _has_assertion_verb(sentence)
    )
    if not has_signal:
        return False

    # Must have a subject-like anchor
    if not _has_subject(sentence):
        return False

    return True


# ------------------------------------------------------------------
# Confidence classification
# ------------------------------------------------------------------

def _classify_confidence(sentence: str) -> str:
    words = set(sentence.lower().split())
    has_hedge = bool(words & HEDGE_WORDS)
    has_definitive = bool(words & DEFINITIVE_WORDS)
    if has_hedge and not has_definitive:
        return "hedged"
    if has_definitive and not has_hedge:
        return "definitive"
    return "unknown"


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(t.split())


def _deduplicate(claims: List[Claim], threshold: float = DEDUP_THRESHOLD) -> List[Claim]:
    """Remove near-duplicate claims based on normalised text similarity."""
    kept: List[Claim] = []
    seen_normalised: List[str] = []
    for c in claims:
        norm = _normalise(c.text)
        is_dup = False
        for prev in seen_normalised:
            if SequenceMatcher(None, norm, prev).ratio() >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
            seen_normalised.append(norm)
    return kept


# ------------------------------------------------------------------
# Claim hash (deterministic dedup, inspired by WeThePeople)
# ------------------------------------------------------------------

def _claim_hash(source_id: str, text: str) -> str:
    """SHA256 hash of source_id + normalised claim text for same-source dedup."""
    norm = _normalise(text)
    combined = f"{source_id}||{norm}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _claim_hash_global(text: str) -> str:
    """SHA256 hash of normalised text only — cross-source identity."""
    norm = _normalise(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------
# Category auto-classification (token-based, no LLM)
# ------------------------------------------------------------------

_CATEGORY_TERMS: dict[str, frozenset[str]] = {
    "finance": frozenset([
        "rate", "rates", "inflation", "gdp", "deficit", "debt", "bond", "bonds",
        "stock", "stocks", "market", "markets", "fed", "federal reserve",
        "treasury", "bank", "banking", "banks",
        "economy", "economic", "recession", "fiscal", "monetary", "yield", "dollar",
        "interest", "investment", "investments", "investor", "investors",
        "earnings", "revenue", "revenues", "profit", "profits",
        "billion", "trillion", "million",
        "currency", "tariff", "tariffs", "trade", "budget",
        # Earnings call / financial reporting vocabulary
        "margin", "margins", "capex", "depreciation", "amortization",
        "operating income", "net income", "cash flow", "free cash flow",
        "year-over-year", "quarter", "quarterly", "annual", "guidance",
        "shareholders", "dividend", "dividends", "repurchase", "buyback",
        "backlog", "expenses", "cost", "costs",
        "10-k", "10-q", "8-k", "sec", "filing", "filings",
        # Asset management / fund vocabulary
        "assets", "fund", "funds", "hedge fund", "mutual fund", "etf",
        "portfolio", "aum", "manages", "managing", "asset management",
        "firm", "financial", "wall street", "returns", "return",
        "valuation", "equity", "equities", "shares", "ipo",
        "price", "pricing", "cap", "wealth",
        # Company-specific financial terms
        "ceo", "cfo", "quarterly earnings", "annual report",
        "balance sheet", "income statement", "cash position",
        "growth", "percent", "percentage",
    ]),
    "tech": frozenset([
        "ai", "artificial intelligence", "machine learning", "gpu", "chip", "chips",
        "semiconductor", "software", "algorithm", "data", "model", "neural",
        "robot", "robotics", "autonomous", "cloud", "computing", "nvidia", "openai",
        "google", "microsoft", "apple", "meta", "startup", "github", "open source",
        "training", "inference", "llm", "transformer", "api",
        # Expanded tech
        "technology", "platform", "digital", "internet", "server", "servers",
        "database", "processor", "cpu", "hardware", "network",
        "encryption", "blockchain", "crypto", "bitcoin",
        "app", "application", "code", "programming", "developer", "developers",
        "machine", "automation", "quantum",
    ]),
    "politics": frozenset([
        "president", "congress", "senate", "house", "vote", "voted", "election",
        "democrat", "republican", "legislation", "law", "policy", "government",
        "administration", "cabinet", "supreme court", "constitutional", "bill",
        "bipartisan", "partisan", "campaign", "governor", "mayor",
        # Expanded
        "political", "politics", "democrat", "republicans", "democrats",
        "regulatory", "regulation", "regulations", "regulator",
        "federal", "state", "country", "countries", "nation", "nations",
    ]),
    "health": frozenset([
        "health", "healthcare", "hospital", "disease", "vaccine", "pandemic",
        "drug", "drugs", "fda", "clinical", "patient", "patients", "medical",
        "cancer", "treatment", "diagnosis", "mortality", "pharmaceutical",
        "cholesterol", "blood pressure", "trial", "trials", "study",
        "diet", "obesity", "heart", "stroke", "diabetes",
        # Expanded health
        "medicine", "doctor", "doctors", "physician", "nurse",
        "surgery", "symptom", "symptoms", "chronic", "acute",
        "infection", "antibiotic", "antibiotics", "therapy",
        "mental health", "depression", "anxiety",
        "nutrition", "calories", "protein", "carbohydrate", "carbohydrates",
        "fat", "saturated fat", "ldl", "hdl", "triglyceride", "triglycerides",
        "inflammation", "artery", "arteries", "coronary",
        "placebo", "randomized", "double-blind",
        "mediterranean diet", "framingham",
    ]),
    "science": frozenset([
        "research", "study", "experiment", "discovery", "nasa", "space",
        "climate", "temperature", "emissions", "carbon", "energy", "solar",
        "nuclear", "physics", "biology", "genome", "species",
        "cells", "immune", "bacteria", "virus", "protein", "dna", "rna",
        # Expanded science
        "scientist", "scientists", "researcher", "researchers",
        "published", "journal", "peer-reviewed", "findings",
        "hypothesis", "theory", "evidence", "data",
        "university", "professor", "laboratory", "lab",
        "evolution", "ecosystem", "biodiversity", "extinction",
        "astronomy", "telescope", "planet", "galaxy",
        "chemistry", "molecule", "atom", "element",
        "mathematics", "mathematical", "equation", "theorem",
        "correlation", "causation", "statistical", "statistically",
    ]),
    "military": frozenset([
        "military", "defense", "army", "navy", "war", "weapon", "weapons",
        "missile", "nuclear", "nato", "pentagon", "troops", "combat",
        "drone", "drones", "intelligence", "security", "sanctions",
    ]),
}


def _classify_category(text: str) -> str:
    """Classify a claim into a topic category by keyword scoring."""
    lower = text.lower()
    # Strip punctuation so "revenues," matches "revenues" and "quarter." matches "quarter"
    clean = lower.translate(str.maketrans("", "", string.punctuation))
    words = set(clean.split())
    best_cat = "general"
    best_score = 0

    for cat, terms in _CATEGORY_TERMS.items():
        score = 0
        for term in terms:
            # Single-word terms: check word set; multi-word: check substring
            if " " in term:
                if term in clean:
                    score += 2  # phrase match is stronger
            elif term in words:
                score += 1
        if score > best_score:
            best_score = score
            best_cat = cat

    # Require at least 2 points to assign a non-general category
    return best_cat if best_score >= 2 else "general"


# ------------------------------------------------------------------
# Boilerplate filter (reject YouTube transcript filler)
# ------------------------------------------------------------------

_BOILERPLATE_PATTERNS = frozenset([
    "subscribe", "like and subscribe", "hit the bell", "leave a comment",
    "check out", "link in the description", "sponsored by", "thanks for watching",
    "let me know", "in the comments", "smash that", "don't forget to",
    "follow me on", "join the", "patreon", "merch",
])


def _is_boilerplate(text: str) -> bool:
    """Return True if text looks like YouTube filler / self-promotion."""
    lower = text.lower()
    matches = sum(1 for pat in _BOILERPLATE_PATTERNS if pat in lower)
    return matches >= 2  # two or more signals → filler


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_claims_from_segments(segments: List[Segment], source_id: str) -> List[Claim]:
    """Extract candidate claims from a list of transcript segments.

    Uses segment stitching to build complete sentences, then filters for
    self-contained, checkable claims.
    """
    raw_claims: List[Claim] = []
    seen_texts: set = set()  # quick exact-dedup before expensive SequenceMatcher

    for seg_idx, seg in enumerate(segments):
        # Build stitched text window around this segment
        window_text, win_start, win_end = _stitch_window(segments, seg_idx)

        # Extract complete sentences from the window
        sentence_tuples = _extract_sentences_from_window(
            window_text, seg.text, win_start, win_end,
        )

        for sent, ts_start, ts_end in sentence_tuples:
            # Length filters
            if len(sent.split()) < MIN_CLAIM_WORDS or len(sent) < MIN_CLAIM_CHARS:
                continue

            # Reject dangling clauses
            if _starts_with_conjunction(sent):
                continue

            # Must be a candidate claim (signal + subject)
            if not _is_candidate(sent):
                continue

            # Reject boilerplate / YouTube filler
            if _is_boilerplate(sent):
                continue

            # Quick exact dedup via hash
            chash = _claim_hash(source_id, sent)
            if chash in seen_texts:
                continue
            seen_texts.add(chash)

            # Collect explainability signals
            signals = _collect_signals(sent)
            conf = _classify_confidence(sent)
            if conf != "unknown":
                signals.append(f"confidence:{conf}")
            cat = _classify_category(sent)
            if cat != "general":
                signals.append(f"category:{cat}")

            raw_claims.append(Claim(
                id=new_id(),
                source_id=source_id,
                text=sent,
                ts_start=ts_start,
                ts_end=ts_end,
                confidence_language=conf,
                category=cat,
                claim_hash=chash,
                claim_hash_global=_claim_hash_global(sent),
                signals="|".join(signals),
            ))

    return _deduplicate(raw_claims)


def extract_claims(source_id: str) -> List[Claim]:
    """Full pipeline: load transcript from disk, extract, store in DB, write claims.json."""
    tmeta = db.get_transcript(source_id)
    if tmeta is None:
        raise ValueError(f"No transcript found for source '{source_id}'. Run `veritas transcribe` first.")

    tpath = Path(tmeta.transcript_path)
    if not tpath.exists():
        raise FileNotFoundError(f"Transcript file missing: {tpath}")

    with open(tpath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    segments = [Segment(**s) for s in data.get("segments", [])]
    claims = extract_claims_from_segments(segments, source_id)

    # Clear previous claims for this source (allows re-running with tuned rules)
    db.delete_claims_for_source(source_id)

    if claims:
        db.insert_claims(claims)

    # Write claims.json export
    out_dir = source_export_dir(source_id)
    claims_path = out_dir / "claims.json"
    payload = [
        {
            "id": c.id,
            "text": c.text,
            "ts_start": c.ts_start,
            "ts_end": c.ts_end,
            "confidence_language": c.confidence_language,
            "category": c.category,
            "claim_hash": c.claim_hash[:16],
            "claim_hash_global": c.claim_hash_global[:16],
            "signals": c.signals,
            "status": c.status,
        }
        for c in claims
    ]
    with open(claims_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    return claims
