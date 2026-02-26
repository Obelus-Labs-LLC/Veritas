"""Tests for assisted verification: scoring, guardrails, evidence sources, DB migration."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.scoring import score_evidence, compute_auto_status, _normalise, _tokenize
from veritas.evidence_sources.base import build_search_query
from veritas.models import Source, Claim, EvidenceSuggestion, new_id


# ── Scoring tests ────────────────────────────────────────────────


def test_score_high_overlap():
    """High token overlap + primary source should score well."""
    score, sigs = score_evidence(
        claim_text="The Federal Reserve raised interest rates by 25 basis points in January.",
        claim_category="finance",
        evidence_title="Federal Reserve Interest Rate Decision January 2026",
        evidence_snippet="The Federal Open Market Committee raised rates by 25 basis points.",
        evidence_type="paper",
        source_name="crossref",
    )
    assert score >= 50
    assert "token_overlap" in sigs
    assert "primary_source" in sigs


def test_score_low_overlap():
    """Unrelated evidence should score low."""
    score, sigs = score_evidence(
        claim_text="The Federal Reserve raised interest rates by 25 basis points.",
        claim_category="finance",
        evidence_title="Guide to Baking Sourdough Bread",
        evidence_snippet="Mix flour and water to create a starter.",
        evidence_type="other",
        source_name="crossref",
    )
    assert score < 20


def test_score_entity_match_boost():
    """Named entities matching should boost score."""
    score, sigs = score_evidence(
        claim_text="Goldman Sachs reported 15% revenue growth last quarter.",
        claim_category="finance",
        evidence_title="Goldman Sachs Q4 2025 Earnings Report",
        evidence_snippet="Revenue grew 15% year-over-year.",
        evidence_type="filing",
        source_name="sec_edgar",
    )
    assert "entity_match" in sigs
    assert "number_match" in sigs


def test_score_category_boost():
    """Category-relevant terms should add points."""
    score, sigs = score_evidence(
        claim_text="The new AI model achieves state of the art results on benchmarks.",
        claim_category="tech",
        evidence_title="Neural Network Model Achieves New Benchmark in Computing",
        evidence_snippet="The AI model sets new records on standard benchmarks.",
        evidence_type="paper",
        source_name="arxiv",
    )
    assert "category_match" in sigs


def test_score_generic_title_penalty():
    """Generic titles should be penalised."""
    score, sigs = score_evidence(
        claim_text="Inflation dropped to 2.3 percent in March.",
        claim_category="finance",
        evidence_title="Introduction",
        evidence_snippet="",
        evidence_type="other",
        source_name="crossref",
    )
    assert "generic_title_penalty" in sigs


def test_score_keyphrase_hit():
    """Multi-word phrase matches should add points."""
    score, sigs = score_evidence(
        claim_text="Open source artificial intelligence is growing rapidly.",
        claim_category="tech",
        evidence_title="The Growth of Open Source Artificial Intelligence",
        evidence_snippet="AI growth in open source communities.",
        evidence_type="paper",
        source_name="crossref",
    )
    assert "keyphrase_hit" in sigs


# ── Guardrails tests ────────────────────────────────────────────


def test_guardrail_supported_requires_high_score():
    """AUTO SUPPORTED requires score >= 85."""
    status, conf = compute_auto_status(
        best_score=60,
        best_evidence_type="paper",
        best_signals="token_overlap:5|keyphrase_hit:2|primary_source:paper",
        claim_confidence="definitive",
    )
    assert status != "supported"


def test_guardrail_supported_requires_primary():
    """AUTO SUPPORTED requires primary source type."""
    status, conf = compute_auto_status(
        best_score=90,
        best_evidence_type="other",
        best_signals="token_overlap:5|keyphrase_hit:2",
        claim_confidence="definitive",
    )
    assert status != "supported"


def test_guardrail_supported_requires_keyphrase():
    """AUTO SUPPORTED requires keyphrase_hit signal."""
    status, conf = compute_auto_status(
        best_score=90,
        best_evidence_type="paper",
        best_signals="token_overlap:5|primary_source:paper",
        claim_confidence="definitive",
    )
    assert status != "supported"


def test_guardrail_supported_all_conditions():
    """AUTO SUPPORTED when all conditions are met."""
    status, conf = compute_auto_status(
        best_score=90,
        best_evidence_type="paper",
        best_signals="token_overlap:5|keyphrase_hit:2|primary_source:paper",
        claim_confidence="definitive",
    )
    assert status == "supported"
    assert conf >= 0.85


def test_guardrail_partial_mid_range():
    """Scores 70-84 should yield PARTIAL."""
    status, conf = compute_auto_status(
        best_score=75,
        best_evidence_type="paper",
        best_signals="token_overlap:3|primary_source:paper",
        claim_confidence="hedged",
    )
    assert status == "partial"


def test_guardrail_unknown_low_score():
    """Low scores should remain UNKNOWN."""
    status, conf = compute_auto_status(
        best_score=40,
        best_evidence_type="paper",
        best_signals="token_overlap:2",
        claim_confidence="definitive",
    )
    assert status == "unknown"


def test_guardrail_never_contradicted():
    """V1 should never auto-set CONTRADICTED."""
    # Even with high scores, status should not be "contradicted"
    for score in [50, 75, 90, 100]:
        status, _ = compute_auto_status(
            best_score=score,
            best_evidence_type="paper",
            best_signals="token_overlap:5|keyphrase_hit:2|primary_source:paper",
            claim_confidence="definitive",
        )
        assert status != "contradicted"


# ── Search query builder tests ───────────────────────────────────


def test_build_search_query_strips_filler():
    """Search query should strip common filler words."""
    q = build_search_query("The Federal Reserve is going to raise interest rates by 25 basis points.")
    assert "the" not in q.lower().split()
    assert "is" not in q.lower().split()
    assert "Federal" in q


def test_build_search_query_keeps_numbers():
    """Numbers should be kept in search queries."""
    q = build_search_query("Revenue grew by 15 percent in 2025.")
    assert "15" in q
    assert "2025" in q


def test_build_search_query_limits_terms():
    """Query should be limited to max_terms."""
    q = build_search_query("A B C D E F G H I J K L M N O P", max_terms=4)
    assert len(q.split()) <= 4


# ── Model tests ──────────────────────────────────────────────────


def test_claim_final_status_human_override():
    """Human status should override auto status."""
    c = Claim(status="unknown", status_auto="supported", auto_confidence=0.9, status_human="contradicted")
    assert c.final_status == "contradicted"


def test_claim_final_status_auto_fallback():
    """Auto status should be used when no human override."""
    c = Claim(status="unknown", status_auto="partial", auto_confidence=0.75, status_human=None)
    assert c.final_status == "partial"


def test_claim_final_status_legacy():
    """Legacy status used when both auto and human are absent."""
    c = Claim(status="supported", status_auto="unknown", status_human=None)
    assert c.final_status == "supported"


def test_evidence_suggestion_model():
    """EvidenceSuggestion should initialize with all fields."""
    s = EvidenceSuggestion(
        claim_id="test123",
        url="https://doi.org/example",
        title="Test Paper",
        source_name="crossref",
        evidence_type="paper",
        score=75,
        signals="token_overlap:4|primary_source:paper",
        snippet="A test snippet.",
    )
    assert s.claim_id == "test123"
    assert s.score == 75
    assert len(s.id) == 12


# ── DB migration test ────────────────────────────────────────────


def test_db_migration_adds_new_columns(tmp_path, monkeypatch):
    """DB migration should add new columns to existing tables."""
    import veritas.config as _config
    monkeypatch.setattr(_config, "DB_PATH", tmp_path / "test_migrate.sqlite")

    from veritas import db as _db
    from veritas.models import Source, Claim

    # Initialize DB (creates all tables with current schema)
    _db.init_db()

    # Insert a source and claim to confirm roundtrip
    src = Source(id="mig_test", url="http://example.com", title="Migration Test")
    _db.insert_source(src)

    claim = Claim(
        id="claim_mig",
        source_id="mig_test",
        text="Test claim for migration.",
        status_auto="partial",
        auto_confidence=0.72,
    )
    _db.insert_claims([claim])

    # Read it back and verify new fields
    got = _db.get_claim("claim_mig")
    assert got is not None
    assert got.status_auto == "partial"
    assert got.auto_confidence == 0.72

    # Verify evidence_suggestions table exists
    suggestions = _db.get_suggestions_for_claim("claim_mig")
    assert suggestions == []  # empty but no error


def test_db_evidence_suggestions_crud(tmp_path, monkeypatch):
    """Evidence suggestions should be insertable and retrievable."""
    import veritas.config as _config
    monkeypatch.setattr(_config, "DB_PATH", tmp_path / "test_evsug.sqlite")

    from veritas import db as _db

    _db.init_db()

    # Create source + claim first
    src = Source(id="sug_src", url="http://example.com", title="Sug Test")
    _db.insert_source(src)
    claim = Claim(id="sug_claim", source_id="sug_src", text="Test claim.")
    _db.insert_claims([claim])

    # Insert suggestions
    s1 = EvidenceSuggestion(
        claim_id="sug_claim", url="https://doi.org/1", title="Paper 1",
        source_name="crossref", evidence_type="paper", score=85,
        signals="token_overlap:5|primary_source:paper",
    )
    s2 = EvidenceSuggestion(
        claim_id="sug_claim", url="https://arxiv.org/1", title="Paper 2",
        source_name="arxiv", evidence_type="paper", score=60,
        signals="token_overlap:3",
    )
    count = _db.insert_evidence_suggestions([s1, s2])
    assert count == 2

    # Retrieve sorted by score DESC
    results = _db.get_suggestions_for_claim("sug_claim")
    assert len(results) == 2
    assert results[0].score >= results[1].score
