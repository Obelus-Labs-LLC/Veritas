"""Tests for deterministic claim extraction."""

import json
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.models import Segment, Claim, new_id
from veritas.claim_extract import (
    extract_claims_from_segments,
    _split_sentences,
    _is_candidate,
    _classify_confidence,
    _classify_category,
    _claim_hash,
    _claim_hash_global,
    _collect_signals,
    _is_boilerplate,
    _deduplicate,
    _normalise,
    _stitch_window,
    _starts_with_conjunction,
    _has_subject,
)


FIXTURE = Path(__file__).parent / "fixtures" / "sample_transcript.json"


def _load_segments():
    with open(FIXTURE, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [Segment(**s) for s in data["segments"]]


# -- Sentence splitting -----------------------------------------------

def test_split_sentences_basic():
    text = "Hello world. This is a test sentence. And another one here."
    parts = _split_sentences(text)
    assert len(parts) >= 2


def test_split_ignores_short_fragments():
    text = "Hi. OK. Sure thing, this is a proper sentence that should pass."
    parts = _split_sentences(text)
    for p in parts:
        assert len(p) > 10


# -- Candidate detection -----------------------------------------------

def test_candidate_with_number():
    assert _is_candidate("The deficit grew by 15 percent last year.")


def test_candidate_with_named_entity():
    assert _is_candidate("Goldman Sachs released a new report today.")


def test_candidate_with_assertion_verb():
    assert _is_candidate("This confirms the earlier prediction about rates.")


def test_candidate_with_date():
    assert _is_candidate("The policy changed on January 15 after the meeting.")


def test_non_candidate_generic():
    assert not _is_candidate("well you know it depends on what you think about things")


# -- Subject detection -------------------------------------------------

def test_has_subject_pronoun():
    assert _has_subject("It confirms the trend.")
    assert _has_subject("They announced the results.")


def test_has_subject_capitalized():
    assert _has_subject("Goldman released a report.")


def test_has_subject_number():
    assert _has_subject("65,000 stars in record time.")


# -- Conjunction filter ------------------------------------------------

def test_starts_with_conjunction():
    assert _starts_with_conjunction("and the data shows growth.")
    assert _starts_with_conjunction("But this was expected.")
    assert _starts_with_conjunction("while remembering everything.")
    assert _starts_with_conjunction("because rates are high.")


def test_not_conjunction():
    assert not _starts_with_conjunction("The Federal Reserve announced.")
    assert not _starts_with_conjunction("It has grown significantly.")


# -- Confidence classification ------------------------------------------

def test_hedged():
    assert _classify_confidence("This might cause a recession.") == "hedged"
    assert _classify_confidence("It could possibly work.") == "hedged"


def test_definitive():
    assert _classify_confidence("The data confirms the trend.") == "definitive"
    assert _classify_confidence("This proves the hypothesis.") == "definitive"


def test_unknown_confidence():
    assert _classify_confidence("The number went up by 5 percent.") == "unknown"


# -- Deduplication -----------------------------------------------------

def test_dedup_removes_near_identical():
    claims = [
        Claim(id="a", source_id="s", text="Inflation dropped to 2.3 percent in March."),
        Claim(id="b", source_id="s", text="Inflation dropped to 2.3 percent in march."),
        Claim(id="c", source_id="s", text="Unemployment is at 3.7 percent nationwide."),
    ]
    result = _deduplicate(claims, threshold=0.85)
    assert len(result) == 2
    ids = {c.id for c in result}
    assert "a" in ids
    assert "c" in ids


def test_dedup_keeps_distinct():
    claims = [
        Claim(id="a", source_id="s", text="The Fed cut rates by 25 basis points."),
        Claim(id="b", source_id="s", text="Unemployment hit a record low of 3.5 percent."),
    ]
    result = _deduplicate(claims, threshold=0.85)
    assert len(result) == 2


# -- Stitching ---------------------------------------------------------

def test_stitch_window_merges_adjacent():
    """Stitching should merge adjacent segment text into one window."""
    segs = [
        Segment(start=0.0, end=5.0, text="The European Central Bank"),
        Segment(start=5.0, end=10.0, text="announced it will maintain rates at 4.5 percent."),
    ]
    text, ts_start, ts_end = _stitch_window(segs, center=0)
    assert "European Central Bank" in text
    assert "4.5 percent" in text
    assert ts_start == 0.0
    assert ts_end == 10.0


def test_stitch_produces_complete_sentences():
    """Two fragment segments should yield a complete claim after stitching."""
    segs = [
        Segment(start=42.0, end=47.0, text="The European Central Bank"),
        Segment(start=47.0, end=53.0, text="announced it will maintain current interest rates at 4.5 percent through the first quarter."),
    ]
    claims = extract_claims_from_segments(segs, source_id="stitch_test")
    ecb_claims = [c for c in claims if "4.5" in c.text or "European" in c.text or "Central" in c.text]
    assert len(ecb_claims) >= 1, f"Expected ECB claim from stitching, got: {[c.text for c in claims]}"
    ecb = ecb_claims[0]
    assert len(ecb.text.split()) >= 7


# -- Full extraction from fixture --------------------------------------

def test_extract_from_fixture():
    segments = _load_segments()
    claims = extract_claims_from_segments(segments, source_id="test_src")
    assert len(claims) >= 3, f"Expected >=3 claims, got {len(claims)}"

    for c in claims:
        assert c.ts_start >= 0
        assert c.ts_end > c.ts_start

    confidences = {c.confidence_language for c in claims}
    assert "hedged" in confidences or "definitive" in confidences


def test_extract_preserves_source_id():
    segments = _load_segments()
    claims = extract_claims_from_segments(segments, source_id="MY_SRC")
    for c in claims:
        assert c.source_id == "MY_SRC"


def test_no_dangling_claims():
    """Claims should not start with conjunctions."""
    segments = _load_segments()
    claims = extract_claims_from_segments(segments, source_id="dangle_test")
    for c in claims:
        first_word = c.text.strip().split()[0].lower().rstrip(",")
        assert first_word not in ("and", "but", "while", "because", "so", "which", "that"), \
            f"Claim starts with conjunction: {c.text[:50]}"


def test_claims_have_subjects():
    """All extracted claims should have a subject-like token."""
    segments = _load_segments()
    claims = extract_claims_from_segments(segments, source_id="subj_test")
    for c in claims:
        assert _has_subject(c.text), f"Claim lacks subject: {c.text[:80]}"


# -- Claim hash (deterministic dedup) ------------------------------------

def test_claim_hash_deterministic():
    """Same source + text always produces the same hash."""
    h1 = _claim_hash("src1", "Inflation dropped to 2.3 percent.")
    h2 = _claim_hash("src1", "Inflation dropped to 2.3 percent.")
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex length


def test_claim_hash_case_insensitive():
    """Hash should be identical regardless of casing."""
    h1 = _claim_hash("src1", "The Fed Announced A Rate Cut.")
    h2 = _claim_hash("src1", "the fed announced a rate cut.")
    assert h1 == h2


def test_claim_hash_different_sources():
    """Different source IDs produce different hashes even for same text."""
    h1 = _claim_hash("src1", "Inflation is at 3 percent.")
    h2 = _claim_hash("src2", "Inflation is at 3 percent.")
    assert h1 != h2


# -- Category classification -----------------------------------------------

def test_category_finance():
    assert _classify_category("The Federal Reserve raised interest rates by 25 basis points.") == "finance"


def test_category_tech():
    assert _classify_category("NVIDIA announced a new GPU chip for AI training workloads.") == "tech"


def test_category_politics():
    assert _classify_category("The Senate voted to pass the bipartisan legislation.") == "politics"


def test_category_health():
    assert _classify_category("The FDA approved a new drug for cancer treatment.") == "health"


def test_category_general_fallback():
    assert _classify_category("Something happened somewhere today.") == "general"


# -- Boilerplate filter ---------------------------------------------------

def test_boilerplate_detected():
    assert _is_boilerplate("Subscribe and hit the bell for more content like this.")
    assert _is_boilerplate("Check out the link in the description for the sponsored deal.")


def test_non_boilerplate():
    assert not _is_boilerplate("The Federal Reserve announced a rate cut of 25 basis points.")
    assert not _is_boilerplate("Inflation dropped to 2.3 percent in March.")


# -- Extracted claims have hashes and categories --------------------------

def test_extracted_claims_have_hash_and_category():
    """All extracted claims should have a non-empty claim_hash and category."""
    segments = _load_segments()
    claims = extract_claims_from_segments(segments, source_id="hash_test")
    for c in claims:
        assert c.claim_hash, f"Claim missing hash: {c.text[:60]}"
        assert len(c.claim_hash) == 64, f"Hash wrong length: {c.claim_hash}"
        assert c.category in ("finance", "tech", "politics", "health", "science", "military", "education", "energy_climate", "labor", "general"), \
            f"Invalid category: {c.category}"


# -- Global hash (cross-source identity) ----------------------------------

def test_global_hash_deterministic():
    """Same text always produces the same global hash."""
    h1 = _claim_hash_global("Inflation dropped to 2.3 percent.")
    h2 = _claim_hash_global("Inflation dropped to 2.3 percent.")
    assert h1 == h2
    assert len(h1) == 64


def test_global_hash_ignores_source():
    """Global hash is source-independent — same text from different sources matches."""
    h_local_1 = _claim_hash("src_A", "Inflation dropped to 2.3 percent.")
    h_local_2 = _claim_hash("src_B", "Inflation dropped to 2.3 percent.")
    h_global = _claim_hash_global("Inflation dropped to 2.3 percent.")
    # Local hashes differ (different source_id)
    assert h_local_1 != h_local_2
    # Global hash is the same for both
    assert h_global == _claim_hash_global("Inflation dropped to 2.3 percent.")


def test_extracted_claims_have_global_hash():
    """All extracted claims should have a non-empty claim_hash_global."""
    segments = _load_segments()
    claims = extract_claims_from_segments(segments, source_id="ghash_test")
    for c in claims:
        assert c.claim_hash_global, f"Claim missing global hash: {c.text[:60]}"
        assert len(c.claim_hash_global) == 64


# -- Signal explainability -------------------------------------------------

def test_collect_signals_number():
    signals = _collect_signals("The deficit grew by 15 percent last year.")
    assert "number" in signals
    assert "has_subject" in signals


def test_collect_signals_named_entity():
    signals = _collect_signals("Goldman Sachs released a new report today.")
    assert "named_entity" in signals


def test_collect_signals_date():
    signals = _collect_signals("The policy changed on January 15 after the meeting.")
    assert "date" in signals


def test_collect_signals_assertion_verb():
    signals = _collect_signals("This confirms the earlier prediction about rates.")
    assert "assertion_verb" in signals


def test_extracted_claims_have_signals():
    """All extracted claims should have at least one signal logged."""
    segments = _load_segments()
    claims = extract_claims_from_segments(segments, source_id="sig_test")
    for c in claims:
        assert c.signals, f"Claim missing signals: {c.text[:60]}"
        signal_list = c.signals.split("|")
        assert len(signal_list) >= 2, f"Expected >=2 signals, got: {c.signals}"
        assert "has_subject" in signal_list, f"Missing 'has_subject' signal: {c.signals}"
