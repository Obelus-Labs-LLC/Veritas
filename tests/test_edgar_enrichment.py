"""Tests for EDGAR enrichment: entity injection, snippet extraction, scoring, and guardrails."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.evidence_sources.sec_edgar import (
    infer_source_entity,
    _html_to_text,
    extract_relevant_snippet,
)
from veritas.scoring import (
    score_evidence,
    compute_auto_status,
    classify_finance_claim,
    _extract_claim_numbers,
)


# ── Entity injection tests ────────────────────────────────────────


def test_infer_entity_alphabet():
    """Should recognize Alphabet from title."""
    assert infer_source_entity("Alphabet 2025 Q4 Earnings Call") == "Alphabet"


def test_infer_entity_google():
    """Should map Google to Alphabet alias."""
    assert infer_source_entity("Google Cloud Revenue Update") == "Alphabet"


def test_infer_entity_meta():
    """Should recognize Meta/Facebook."""
    assert infer_source_entity("Meta Platforms Q3 2025 Results") == "Meta"


def test_infer_entity_apple():
    """Should recognize Apple."""
    assert infer_source_entity("Apple Inc Annual Report") == "Apple"


def test_infer_entity_nvidia():
    """Should recognize Nvidia."""
    assert infer_source_entity("Nvidia AI Chip Sales Surge") == "Nvidia"


def test_infer_entity_fallback_proper_noun():
    """Should fallback to first proper noun for unknown entities."""
    entity = infer_source_entity("Samsung Electronics Q1 Results")
    assert entity == "Samsung" or entity == "Samsung Electronics"


def test_infer_entity_empty():
    """Empty input should return empty string."""
    assert infer_source_entity("") == ""


def test_infer_entity_channel():
    """Should use channel info for entity detection."""
    entity = infer_source_entity("Q4 Earnings Call", "Alphabet Investor Relations")
    assert entity == "Alphabet"


def test_infer_entity_skips_filler_words():
    """Should skip common filler words like 'The', 'How'."""
    entity = infer_source_entity("How The Economic Machine Works")
    # Should not return "How" or "The"
    assert entity not in ("How", "The")


# ── HTML text extraction tests ────────────────────────────────────


def test_html_to_text_basic():
    """Should extract visible text from HTML."""
    html = "<html><body><p>Hello World</p></body></html>"
    text = _html_to_text(html)
    assert "Hello World" in text


def test_html_to_text_strips_scripts():
    """Should strip script and style tags."""
    html = """
    <html><body>
        <script>var x = 1;</script>
        <style>.foo { color: red; }</style>
        <p>Visible text here.</p>
    </body></html>
    """
    text = _html_to_text(html)
    assert "Visible text here" in text
    assert "var x" not in text
    assert "color" not in text


def test_html_to_text_max_chars():
    """Should respect max_chars limit."""
    html = "<p>" + "A" * 1000 + "</p>"
    text = _html_to_text(html, max_chars=100)
    assert len(text) <= 100


def test_html_to_text_whitespace_normalization():
    """Should normalize whitespace."""
    html = "<p>Revenue    was     \n\n  $113.8     billion</p>"
    text = _html_to_text(html)
    assert "Revenue was $113.8 billion" in text


# ── Snippet extraction tests ─────────────────────────────────────


def test_snippet_extraction_exact_number_priority():
    """Snippet window should center on exact number matches."""
    # Build a fake filing text: 10K chars of filler, then the target section
    filler = "Lorem ipsum dolor sit amet. " * 200  # ~5600 chars
    target = "Total revenues were $113.8 billion for the quarter ended December 31, 2025. Operating income was $31.6 billion."
    filing_text = filler + target + filler

    claim = "Alphabet's total revenues were $113.8 billion with operating income of $31.6 billion."
    snippet = extract_relevant_snippet(filing_text, claim, window=4000)

    assert "113.8" in snippet
    assert "31.6" in snippet


def test_snippet_extraction_key_terms():
    """Snippet should find section with key financial terms when no exact numbers."""
    filler = "The quick brown fox jumped over the lazy dog. " * 200
    target = "Revenue growth in the cloud segment exceeded expectations with strong advertising performance."
    filing_text = filler + target + filler

    claim = "Cloud revenue and advertising were strong."
    snippet = extract_relevant_snippet(filing_text, claim, window=4000)

    assert "cloud" in snippet.lower() or "advertising" in snippet.lower() or "revenue" in snippet.lower()


def test_snippet_extraction_empty_filing():
    """Empty filing text should return empty string."""
    assert extract_relevant_snippet("", "Some claim text") == ""


def test_snippet_extraction_short_filing():
    """Short filing should return what's available."""
    filing = "Revenue was $50 million."
    snippet = extract_relevant_snippet(filing, "Revenue was $50 million.", window=4000)
    assert "Revenue" in snippet


# ── Financial number extraction tests ─────────────────────────────


def test_extract_claim_numbers_decimals():
    """Should extract decimal numbers like 113.8, 31.6, 2.82."""
    nums = _extract_claim_numbers("Revenue was $113.8 billion, EPS was $2.82")
    assert "113.8" in nums
    assert "2.82" in nums


def test_extract_claim_numbers_integers():
    """Should extract multi-digit integers."""
    nums = _extract_claim_numbers("Operating income increased 16 percent to $403 million")
    assert "403" in nums
    assert "16" in nums


def test_extract_claim_numbers_filters_single_digits():
    """Single digit numbers < 10 should be filtered out."""
    nums = _extract_claim_numbers("Growth of 3 percent in Q4 2025")
    assert "3" not in nums  # too short, < 10
    assert "2025" in nums


# ── Finance claim type classification tests ───────────────────────


def test_classify_numeric_kpi():
    """Claim with numbers + financial terms = numeric_kpi."""
    assert classify_finance_claim("Revenue was $113.8 billion in Q4 2025") == "numeric_kpi"


def test_classify_numeric_kpi_margin():
    """Margin percentage = numeric_kpi."""
    assert classify_finance_claim("Operating margin expanded to 32 percent") == "numeric_kpi"


def test_classify_guidance():
    """Forward-looking language = guidance."""
    assert classify_finance_claim("We expect revenue growth to accelerate in 2026") == "guidance"


def test_classify_guidance_with_numbers():
    """Guidance with numbers should still be guidance."""
    assert classify_finance_claim("We expect revenue of $120 billion in fiscal 2026") == "guidance"


def test_classify_other():
    """Generic statement without finance terms = other."""
    assert classify_finance_claim("The company is headquartered in Mountain View California") == "other"


# ── Scoring with enriched snippets tests ──────────────────────────


def test_score_exact_number_match_boost():
    """Enriched EDGAR snippet with exact number match should score significantly higher."""
    # Simulate an enriched filing snippet (>200 chars) with matching numbers
    snippet = (
        "For the quarter ended December 31, 2025, Alphabet Inc. reported total revenues of $113.8 billion, "
        "an increase of 12% from the prior year. Operating income was $31.6 billion, representing an operating "
        "margin of approximately 28%. Earnings per share were $2.82. Capital expenditures totaled $14.3 billion "
        "during the quarter."
    )
    score, sigs = score_evidence(
        claim_text="Alphabet reported total revenues of $113.8 billion with operating income of $31.6 billion and EPS of $2.82.",
        claim_category="finance",
        evidence_title="Alphabet Inc. - 10-K",
        evidence_snippet=snippet,
        evidence_type="filing",
        source_name="sec_edgar",
    )
    assert "number_exact_match" in sigs
    assert score >= 65  # should be substantially higher with exact number + filing boost


def test_score_no_exact_match_on_short_snippet():
    """Short snippets (meta-only) should NOT get exact number match boost."""
    score, sigs = score_evidence(
        claim_text="Revenue was $113.8 billion.",
        claim_category="finance",
        evidence_title="Alphabet Inc. - 10-K",
        evidence_snippet="Filed: 2025-02-04 | Period: 2024-12-31",
        evidence_type="filing",
        source_name="sec_edgar",
    )
    assert "number_exact_match" not in sigs


# ── Auto-status guardrails with finance claim type ────────────────


def test_guardrail_guidance_always_unknown():
    """Guidance claims should always stay UNKNOWN regardless of score."""
    status, conf = compute_auto_status(
        best_score=95,
        best_evidence_type="filing",
        best_signals="token_overlap:8|keyphrase_hit:3|number_exact_match:113.8,31.6|primary_source:filing",
        claim_confidence="definitive",
        finance_claim_type="guidance",
    )
    assert status == "unknown"


def test_guardrail_numeric_kpi_can_be_supported():
    """numeric_kpi claims CAN reach SUPPORTED with all conditions met."""
    status, conf = compute_auto_status(
        best_score=90,
        best_evidence_type="filing",
        best_signals="token_overlap:8|number_exact_match:113.8,31.6|primary_source:filing",
        claim_confidence="definitive",
        finance_claim_type="numeric_kpi",
    )
    assert status == "supported"


def test_guardrail_numeric_kpi_partial():
    """numeric_kpi with score 70-84 should be PARTIAL."""
    status, conf = compute_auto_status(
        best_score=78,
        best_evidence_type="filing",
        best_signals="token_overlap:5|number_match:113|primary_source:filing",
        claim_confidence="definitive",
        finance_claim_type="numeric_kpi",
    )
    assert status == "partial"


def test_guardrail_exact_number_substitutes_keyphrase():
    """Exact number match should substitute for keyphrase_hit in SUPPORTED check."""
    # Without keyphrase, but WITH exact number match -> should still be SUPPORTED
    status, conf = compute_auto_status(
        best_score=88,
        best_evidence_type="filing",
        best_signals="token_overlap:6|number_exact_match:113.8,31.6|primary_source:filing",
        claim_confidence="definitive",
        finance_claim_type="numeric_kpi",
    )
    assert status == "supported"

    # Without BOTH keyphrase and exact number match -> should NOT be supported
    status2, _ = compute_auto_status(
        best_score=88,
        best_evidence_type="filing",
        best_signals="token_overlap:6|primary_source:filing",
        claim_confidence="definitive",
        finance_claim_type="numeric_kpi",
    )
    assert status2 != "supported"


def test_guardrail_other_finance_type_normal():
    """'other' finance type should follow normal guardrails."""
    status, conf = compute_auto_status(
        best_score=90,
        best_evidence_type="filing",
        best_signals="token_overlap:8|keyphrase_hit:3|primary_source:filing",
        claim_confidence="definitive",
        finance_claim_type="other",
    )
    assert status == "supported"
