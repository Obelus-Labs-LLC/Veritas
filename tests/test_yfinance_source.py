"""Tests for yfinance evidence source, smart routing, and scoring integration."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.evidence_sources.yfinance_source import (
    _extract_ticker,
    _format_market_data_snippet,
    _format_number,
    search_yfinance,
    _TICKER_MAP,
    _TICKER_BLACKLIST,
)
from veritas.assist import (
    _has_company_mention,
    _smart_select_sources,
    _select_sources_for_category,
    _ENTITY_ALIASES,
    _ACADEMIC_TERMS,
    _HEALTH_TERMS,
    _FINANCIAL_METRIC_TERMS,
)
from veritas.scoring import score_evidence


# ── Ticker extraction ─────────────────────────────────────────────


def test_extract_ticker_company_name():
    """Known company names should resolve to tickers."""
    assert _extract_ticker("Alphabet reported record revenue") == "GOOG"
    assert _extract_ticker("Microsoft earnings beat expectations") == "MSFT"
    assert _extract_ticker("Tesla stock surged 15%") == "TSLA"


def test_extract_ticker_case_insensitive():
    """Company name matching should be case-insensitive."""
    assert _extract_ticker("NVIDIA dominates GPU market") == "NVDA"
    assert _extract_ticker("Goldman Sachs reported profits") == "GS"


def test_extract_ticker_multi_word():
    """Multi-word company names should match."""
    assert _extract_ticker("JP Morgan raised forecasts") == "JPM"
    assert _extract_ticker("Bank of America increased dividends") == "BAC"
    assert _extract_ticker("Berkshire Hathaway acquired a stake") == "BRK-B"


def test_extract_ticker_explicit_symbol():
    """Explicit ticker symbols in text should be detected."""
    assert _extract_ticker("GOOG traded at 180 today") == "GOOG"
    assert _extract_ticker("Shares of MSFT rose 3%") == "MSFT"


def test_extract_ticker_no_match():
    """Claims without company references should return None."""
    assert _extract_ticker("The economy grew 3% last quarter") is None
    assert _extract_ticker("Inflation is rising globally") is None


def test_extract_ticker_blacklist():
    """Common English words that look like tickers should be filtered."""
    # "THE" and "FOR" are in the blacklist
    assert _extract_ticker("THE market grew FOR years") is None


def test_extract_ticker_longer_name_preferred():
    """Longer company names should match before shorter ones."""
    # "jp morgan" should match before "jp" alone
    ticker = _extract_ticker("JP Morgan Chase reported earnings")
    assert ticker == "JPM"


# ── Number formatting ────────────────────────────────────────────


def test_format_number_trillions():
    assert _format_number(2.5e12) == "2.50T"


def test_format_number_billions():
    assert _format_number(350.018e9) == "350.02B"


def test_format_number_millions():
    assert _format_number(42.5e6) == "42.50M"


def test_format_number_small():
    assert _format_number(2.82) == "2.82"


def test_format_number_none():
    assert _format_number(None) == "N/A"


# ── Snippet formatting ──────────────────────────────────────────


def test_format_snippet_includes_company_name():
    """Snippet should include company name and ticker."""
    info = {"symbol": "GOOG", "shortName": "Alphabet Inc.", "marketCap": 2e12}
    snippet = _format_market_data_snippet(info, "Alphabet revenue")
    assert "Alphabet Inc." in snippet
    assert "GOOG" in snippet


def test_format_snippet_includes_metrics():
    """Snippet should include financial metrics when available."""
    info = {
        "symbol": "AAPL",
        "shortName": "Apple Inc.",
        "marketCap": 3e12,
        "totalRevenue": 400e9,
        "trailingEps": 6.42,
        "trailingPE": 28.5,
    }
    snippet = _format_market_data_snippet(info, "Apple earnings")
    assert "Market Cap" in snippet
    assert "Revenue" in snippet
    assert "EPS" in snippet
    assert "P/E Ratio" in snippet


def test_format_snippet_raw_values():
    """Snippet should include raw numeric values for scoring engine matching."""
    info = {
        "symbol": "MSFT",
        "shortName": "Microsoft",
        "totalRevenue": 236e9,
        "trailingEps": 11.53,
    }
    snippet = _format_market_data_snippet(info, "Microsoft revenue")
    assert "Raw values" in snippet


def test_format_snippet_percentages():
    """Percentage fields should be multiplied by 100."""
    info = {
        "symbol": "TSLA",
        "shortName": "Tesla",
        "profitMargins": 0.125,
        "revenueGrowth": 0.28,
    }
    snippet = _format_market_data_snippet(info, "Tesla growth")
    assert "12.5%" in snippet
    assert "28.0%" in snippet


# ── search_yfinance function ────────────────────────────────────


def test_search_yfinance_no_ticker():
    """No ticker found → empty results."""
    results = search_yfinance("The economy is growing")
    assert results == []


@patch("yfinance.Ticker")
def test_search_yfinance_returns_dataset(mock_ticker_cls):
    """yfinance result should have evidence_type='dataset'."""
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "symbol": "GOOG",
        "shortName": "Alphabet Inc.",
        "marketCap": 2e12,
        "totalRevenue": 350e9,
    }
    mock_ticker.news = []
    mock_ticker_cls.return_value = mock_ticker

    results = search_yfinance("Alphabet reported 350 billion in revenue")
    assert len(results) >= 1
    assert results[0]["evidence_type"] == "dataset"
    assert results[0]["source_name"] == "yfinance"
    assert "GOOG" in results[0]["url"]


@patch("yfinance.Ticker")
def test_search_yfinance_includes_news(mock_ticker_cls):
    """yfinance should return news items as secondary evidence."""
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "symbol": "TSLA",
        "shortName": "Tesla Inc.",
        "marketCap": 800e9,
    }
    mock_ticker.news = [
        {"title": "Tesla Q4 Earnings Beat", "publisher": "Reuters", "link": "https://example.com/1"},
        {"title": "Tesla Expands Factory", "publisher": "Bloomberg", "link": "https://example.com/2"},
    ]
    mock_ticker_cls.return_value = mock_ticker

    results = search_yfinance("Tesla stock performance")
    assert len(results) >= 2
    # First result is dataset, rest are secondary
    assert results[0]["evidence_type"] == "dataset"
    assert results[1]["evidence_type"] == "secondary"


@patch("yfinance.Ticker")
def test_search_yfinance_empty_info(mock_ticker_cls):
    """Empty info dict should return empty results."""
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker_cls.return_value = mock_ticker

    results = search_yfinance("Alphabet earnings")
    assert results == []


# ── _has_company_mention ─────────────────────────────────────────


def test_has_company_mention_ticker_map():
    """Should detect companies from _TICKER_MAP."""
    assert _has_company_mention("alphabet reported revenue") is True
    assert _has_company_mention("nvidia gpu sales soared") is True
    assert _has_company_mention("blackrock manages trillions") is True


def test_has_company_mention_entity_aliases():
    """Should detect companies from _ENTITY_ALIASES."""
    assert _has_company_mention("openai released gpt-5") is True
    assert _has_company_mention("spacex launched a rocket") is True


def test_has_company_mention_no_match():
    """No company mention should return False."""
    assert _has_company_mention("the economy grew 3 percent") is False
    assert _has_company_mention("inflation rates are rising") is False


# ── _smart_select_sources ────────────────────────────────────────


def _make_sources():
    """Create a fake source list for testing ordering."""
    def noop(text, max_results=5):
        return []
    return [
        ("crossref", noop),
        ("arxiv", noop),
        ("pubmed", noop),
        ("sec_edgar", noop),
        ("yfinance", noop),
    ]


def test_smart_routing_company_mention_boosts_yfinance():
    """Company mention should push yfinance to front."""
    sources = _make_sources()
    result = _smart_select_sources(
        "Alphabet reported 350 billion in revenue last quarter",
        "finance",
        sources,
    )
    names = [name for name, _ in result]
    assert names[0] == "yfinance", f"Expected yfinance first, got {names}"


def test_smart_routing_academic_boosts_arxiv():
    """Academic language should push arxiv to front."""
    sources = _make_sources()
    result = _smart_select_sources(
        "A new study published by researchers found correlation between variables",
        "science",
        sources,
    )
    names = [name for name, _ in result]
    assert names.index("arxiv") < names.index("yfinance")


def test_smart_routing_health_boosts_pubmed():
    """Health terms should push pubmed to front."""
    sources = _make_sources()
    result = _smart_select_sources(
        "Clinical trials showed the drug treatment improved patient outcomes",
        "health",
        sources,
    )
    names = [name for name, _ in result]
    assert names.index("pubmed") < names.index("yfinance")


def test_smart_routing_financial_metrics():
    """Financial metric language should boost yfinance + sec_edgar."""
    sources = _make_sources()
    result = _smart_select_sources(
        "The company revenue grew with improved profit margins",
        "finance",
        sources,
    )
    names = [name for name, _ in result]
    # yfinance and sec_edgar should be near top
    assert names.index("yfinance") < names.index("arxiv")
    assert names.index("sec_edgar") < names.index("arxiv")


def test_smart_routing_preserves_all_sources():
    """Smart routing should not drop any sources, only reorder."""
    sources = _make_sources()
    result = _smart_select_sources(
        "Alphabet earnings beat expectations",
        "finance",
        sources,
    )
    original_names = {name for name, _ in sources}
    result_names = {name for name, _ in result}
    assert original_names == result_names


def test_smart_routing_no_signals_preserves_order():
    """No content signals → preserve original category ordering."""
    sources = _make_sources()
    result = _smart_select_sources(
        "Something happened recently",
        "general",
        sources,
    )
    # With no signals, original order should be preserved
    assert [name for name, _ in result] == [name for name, _ in sources]


# ── Registry tests ───────────────────────────────────────────────


def test_yfinance_in_all_sources():
    """yfinance should be registered in ALL_SOURCES."""
    from veritas.evidence_sources import ALL_SOURCES
    names = [name for name, _ in ALL_SOURCES]
    assert "yfinance" in names


def test_all_sources_count():
    """Should have 7 evidence sources total."""
    from veritas.evidence_sources import ALL_SOURCES
    assert len(ALL_SOURCES) == 7


def test_yfinance_in_finance_category():
    """Finance category should include yfinance as first source."""
    sources = _select_sources_for_category("finance")
    names = [name for name, _ in sources]
    assert "yfinance" in names
    assert names[0] == "yfinance"


# ── Scoring integration ─────────────────────────────────────────


def test_scoring_dataset_gets_primary_boost():
    """evidence_type='dataset' (yfinance) should get 15-point primary boost."""
    score, sigs = score_evidence(
        claim_text="Alphabet market cap is 2 trillion dollars",
        claim_category="finance",
        evidence_title="Alphabet Inc. (GOOG) - Market Data",
        evidence_snippet="Alphabet Inc. (GOOG) | Market Cap: $2.00T | Revenue (TTM): $350.02B",
        evidence_type="dataset",
        source_name="yfinance",
    )
    assert "primary_source:dataset" in sigs
    assert score >= 30  # should be decent with overlap + primary boost


def test_scoring_finance_category_new_terms():
    """New finance terms (cap, price, eps, etc.) should trigger category boost."""
    score, sigs = score_evidence(
        claim_text="The stock price reached new highs",
        claim_category="finance",
        evidence_title="Stock Market Report",
        evidence_snippet="Stock price cap ratio valuation quarterly eps dividend analysis",
        evidence_type="other",
        source_name="crossref",
    )
    assert "category_match:finance" in sigs


def test_scoring_yfinance_snippet_number_match():
    """yfinance snippet with matching numbers should fire number_exact_match."""
    score, sigs = score_evidence(
        claim_text="Alphabet revenue was 350 billion dollars with EPS of 2.82",
        claim_category="finance",
        evidence_title="Alphabet Inc. (GOOG) - Market Data",
        # Snippet > 200 chars to trigger exact financial number matching
        evidence_snippet=(
            "Alphabet Inc. (GOOG) | Market Cap: $2.00T | Revenue (TTM): $350.02B | "
            "Net Income: $100.68B | EPS: $2.82 | P/E Ratio: $23.15 | "
            "Current Price: $180.50 | 52wk High: $195.00 | 52wk Low: $130.00 | "
            "Dividend Yield: N/A | Profit Margin: 28.8% | Revenue Growth: 14.0% | "
            "Raw values: 2000.0 350.0 100.7 2.82 23.15 180.50"
        ),
        evidence_type="dataset",
        source_name="yfinance",
    )
    assert "number_match" in sigs or "number_exact_match" in sigs
    assert score >= 40


# ── TICKER_MAP / TICKER_BLACKLIST sanity ─────────────────────────


def test_ticker_map_has_major_companies():
    """Ticker map should include major companies."""
    assert "alphabet" in _TICKER_MAP
    assert "google" in _TICKER_MAP
    assert "microsoft" in _TICKER_MAP
    assert "apple" in _TICKER_MAP
    assert "nvidia" in _TICKER_MAP
    assert "blackrock" in _TICKER_MAP


def test_ticker_blacklist_filters_common_words():
    """Blacklist should contain common English words that look like tickers."""
    assert "THE" in _TICKER_BLACKLIST
    assert "AND" in _TICKER_BLACKLIST
    assert "CEO" in _TICKER_BLACKLIST
    assert "GDP" in _TICKER_BLACKLIST
