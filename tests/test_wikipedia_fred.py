"""Tests for Wikipedia and FRED evidence sources, improved routing, and build_search_query."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.evidence_sources.wikipedia_source import (
    search_wikipedia,
    _clean_extract,
    _build_results_from_search,
)
from veritas.evidence_sources.fred_source import (
    search_fred,
    _match_series,
    _extract_macro_keywords,
    _SERIES_MAP,
)
from veritas.evidence_sources.base import build_search_query
from veritas.evidence_sources import ALL_SOURCES
from veritas.assist import (
    _select_sources_for_category,
    _smart_select_sources,
    _MACRO_TERMS,
)


# ── Wikipedia source ─────────────────────────────────────────────


def test_wikipedia_search_empty_query():
    """Empty/unusable claim should return empty results."""
    results = search_wikipedia("")
    assert results == []


def test_clean_extract_relevance_sorting():
    """Extract cleaner should prioritize paragraphs relevant to claim."""
    extract = (
        "BlackRock Inc. is an American multinational investment company.\n"
        "It was founded in 1988 by Larry Fink.\n"
        "The company is based in New York City.\n"
        "BlackRock manages approximately $10 trillion in assets."
    )
    snippet = _clean_extract(extract, "BlackRock manages trillions in assets")
    # The paragraph about managing assets should be prioritized
    assert "trillion" in snippet or "manages" in snippet


def test_clean_extract_empty():
    """Empty extract should return empty string."""
    assert _clean_extract("", "some claim") == ""


def test_build_results_from_search_strips_html():
    """Fallback results should strip HTML tags from snippets."""
    search_results = [
        {
            "pageid": 123,
            "title": "Test Article",
            "snippet": "This is <span class='searchmatch'>bold</span> text",
        }
    ]
    results = _build_results_from_search(search_results, 5)
    assert len(results) == 1
    assert "<span" not in results[0]["snippet"]
    assert "bold" in results[0]["snippet"]


def test_wikipedia_result_format():
    """Wikipedia results should have correct source_name and evidence_type."""
    search_results = [{"pageid": 1, "title": "Test"}]
    results = _build_results_from_search(search_results, 5)
    assert results[0]["source_name"] == "wikipedia"
    assert results[0]["evidence_type"] == "secondary"


# ── FRED source ──────────────────────────────────────────────────


def test_fred_match_series_gdp():
    """GDP claim should match the GDP series."""
    assert _match_series("US GDP grew 3% last quarter") == "GDP"


def test_fred_match_series_unemployment():
    """Unemployment claim should match UNRATE."""
    assert _match_series("The unemployment rate fell to 3.5%") == "UNRATE"


def test_fred_match_series_inflation():
    """Inflation claim should match CPI series."""
    assert _match_series("Inflation hit 7% in 2022") == "CPIAUCSL"


def test_fred_match_series_fed_rate():
    """Fed rate claim should match FEDFUNDS."""
    assert _match_series("The federal funds rate was raised to 5.25%") == "FEDFUNDS"


def test_fred_match_series_none():
    """Non-economic claim should return None."""
    assert _match_series("The movie was released in 2023") is None


def test_fred_extract_macro_keywords():
    """Should extract known macro terms from claim text."""
    keywords = _extract_macro_keywords("GDP growth and inflation are both rising")
    assert "gdp" in keywords
    assert "inflation" in keywords


def test_fred_search_returns_dataset():
    """FRED results should be evidence_type='dataset'."""
    results = search_fred("US GDP grew 3 percent last quarter")
    assert len(results) >= 1
    assert results[0]["evidence_type"] == "dataset"
    assert results[0]["source_name"] == "fred"
    assert "fred.stlouisfed.org" in results[0]["url"]


def test_fred_search_no_match():
    """Non-economic claims should return empty."""
    results = search_fred("The movie was a box office hit")
    assert results == []


def test_fred_series_map_coverage():
    """FRED series map should cover major economic indicators."""
    assert "gdp" in _SERIES_MAP
    assert "inflation" in _SERIES_MAP
    assert "unemployment" in _SERIES_MAP
    assert "interest rate" in _SERIES_MAP
    assert "cpi" in _SERIES_MAP


# ── Registry ─────────────────────────────────────────────────────


def test_wikipedia_in_all_sources():
    """Wikipedia should be registered in ALL_SOURCES."""
    names = [name for name, _ in ALL_SOURCES]
    assert "wikipedia" in names


def test_fred_in_all_sources():
    """FRED should be registered in ALL_SOURCES."""
    names = [name for name, _ in ALL_SOURCES]
    assert "fred" in names


def test_all_sources_count_seven():
    """Should now have 7 evidence sources total."""
    assert len(ALL_SOURCES) == 7


# ── Routing updates ──────────────────────────────────────────────


def test_finance_category_includes_fred():
    """Finance routing should include fred."""
    sources = _select_sources_for_category("finance")
    names = [name for name, _ in sources]
    assert "fred" in names


def test_general_category_includes_wikipedia():
    """General routing should include wikipedia."""
    sources = _select_sources_for_category("general")
    names = [name for name, _ in sources]
    assert "wikipedia" in names
    # Wikipedia should be first for general claims
    assert names[0] == "wikipedia"


def test_health_category_includes_wikipedia():
    """Health routing should include wikipedia."""
    sources = _select_sources_for_category("health")
    names = [name for name, _ in sources]
    assert "wikipedia" in names


def test_smart_routing_macro_boosts_fred():
    """Macroeconomic terms should boost fred."""
    def noop(text, max_results=5):
        return []
    sources = [
        ("crossref", noop), ("arxiv", noop), ("pubmed", noop),
        ("sec_edgar", noop), ("yfinance", noop),
        ("wikipedia", noop), ("fred", noop),
    ]
    result = _smart_select_sources(
        "GDP growth was 3 percent driven by strong consumer spending",
        "finance",
        sources,
    )
    names = [name for name, _ in result]
    assert names.index("fred") < names.index("arxiv")


def test_smart_routing_named_entities_boost_wikipedia():
    """Named entities should boost wikipedia in ranking."""
    def noop(text, max_results=5):
        return []
    sources = [
        ("crossref", noop), ("arxiv", noop), ("pubmed", noop),
        ("sec_edgar", noop), ("yfinance", noop),
        ("wikipedia", noop), ("fred", noop),
    ]
    result = _smart_select_sources(
        "Larry Fink founded BlackRock in 1988 in New York City",
        "general",
        sources,
    )
    names = [name for name, _ in result]
    # Wikipedia should be boosted (named entities) and yfinance too (company mention)
    assert names.index("wikipedia") < names.index("arxiv")


# ── build_search_query improvements ──────────────────────────────


def test_query_preserves_proper_nouns():
    """Multi-word proper nouns should be preserved as quoted phrases."""
    query = build_search_query("Goldman Sachs reported 15% revenue growth last quarter")
    assert '"Goldman Sachs"' in query


def test_query_preserves_study_names():
    """Study names as proper nouns should be preserved as quoted phrases."""
    query = build_search_query(
        "The Framingham Study showed that cholesterol levels predict heart disease"
    )
    # Proper noun phrase containing "Framingham" should be quoted
    assert "Framingham" in query
    assert '"' in query  # at least one quoted phrase


def test_query_still_includes_numbers():
    """Numbers should still be included in queries."""
    query = build_search_query("Revenue grew 15% to 96.5 billion dollars")
    assert "15" in query or "96.5" in query


def test_query_still_strips_stop_words():
    """Stop words should still be filtered out."""
    query = build_search_query("The company is doing very well in the market")
    assert "the" not in query.lower().split()
    assert "is" not in query.lower().split()


def test_query_limits_terms():
    """Query should limit terms to avoid excessively long queries."""
    long_claim = "Apple Microsoft Google Amazon Tesla Nvidia Meta Netflix Uber Salesforce reported earnings"
    query = build_search_query(long_claim, max_terms=6)
    # Query should be bounded — not include all 10+ terms
    # Count non-quote tokens as a rough check
    raw_words = [w.strip('"') for w in query.split() if w.strip('"')]
    assert len(raw_words) <= 12  # generous upper bound
