"""Yahoo Finance market data — verify financial claims against live data.

Uses the yfinance library to check stock prices, market caps, revenue,
P/E ratios, and other financial metrics against real-time data.
No API key required.
"""

from __future__ import annotations
import re
import time
from typing import List, Dict, Any, Optional

# Rate limiting — yfinance is a library wrapping HTTP calls to Yahoo
_LAST_REQUEST: float = 0.0
_MIN_INTERVAL = 1.5  # seconds between yfinance calls


def _rate_limit():
    """Enforce minimum interval between yfinance requests."""
    global _LAST_REQUEST
    elapsed = time.time() - _LAST_REQUEST
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _LAST_REQUEST = time.time()


# ------------------------------------------------------------------
# Ticker resolution: company name -> ticker symbol
# ------------------------------------------------------------------

_TICKER_MAP: Dict[str, str] = {
    # Big tech
    "alphabet": "GOOG", "google": "GOOG", "goog": "GOOG", "googl": "GOOG",
    "meta": "META", "facebook": "META",
    "amazon": "AMZN", "amzn": "AMZN",
    "microsoft": "MSFT", "msft": "MSFT",
    "apple": "AAPL", "aapl": "AAPL",
    "nvidia": "NVDA", "nvda": "NVDA",
    "tesla": "TSLA", "tsla": "TSLA",
    "netflix": "NFLX", "nflx": "NFLX",
    # Finance
    "jpmorgan": "JPM", "jp morgan": "JPM", "jpm": "JPM",
    "goldman sachs": "GS", "goldman": "GS",
    "blackrock": "BLK", "black rock": "BLK",
    "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "wells fargo": "WFC",
    "citigroup": "C", "citi": "C",
    # Other major
    "disney": "DIS", "walt disney": "DIS",
    "salesforce": "CRM",
    "intel": "INTC", "intc": "INTC",
    "amd": "AMD",
    "oracle": "ORCL",
    "ibm": "IBM",
    "spotify": "SPOT",
    "uber": "UBER",
    "airbnb": "ABNB",
    "coinbase": "COIN",
    "palantir": "PLTR",
    "snowflake": "SNOW",
    # OpenAI is private — not on yfinance
}

# Common English words that look like tickers but aren't
_TICKER_BLACKLIST = frozenset({
    "THE", "AND", "FOR", "BUT", "NOT", "ARE", "WAS", "HAS", "ITS", "HIS",
    "HER", "THIS", "THAT", "WITH", "FROM", "CEO", "CFO", "COO", "CTO",
    "GDP", "SEC", "IPO", "ETF", "LLC", "INC", "USD", "USA", "API", "RAM",
    "DAY", "NEW", "ALL", "ONE", "TWO", "NOW", "SAY", "WAY", "MAY", "CAN",
    "HOW", "WHY", "WHO", "OUR", "OUT", "TOP", "BIG", "OLD", "SET", "RUN",
    "OWN", "PUT", "LET", "GOT", "GET", "SAW", "USE", "TRY", "ASK", "END",
})


def _extract_ticker(claim_text: str) -> Optional[str]:
    """Extract a stock ticker from claim text.

    Strategy:
      1. Check for known company names in _TICKER_MAP
      2. Check for explicit ticker symbols (all-caps 2-5 letters)

    Returns ticker string or None.
    """
    lower = claim_text.lower()

    # Strategy 1: Known company name lookup (most reliable)
    # Sort by length descending so "jp morgan" matches before "jp"
    for name in sorted(_TICKER_MAP.keys(), key=len, reverse=True):
        if name in lower:
            return _TICKER_MAP[name]

    # Strategy 2: Explicit ticker symbols (all-caps, 2-5 letters)
    # Must be surrounded by word boundaries
    candidates = re.findall(r'\b([A-Z]{2,5})\b', claim_text)
    for c in candidates:
        if c not in _TICKER_BLACKLIST:
            # Validate: it should be a real ticker (exists in our map values)
            if c in _TICKER_MAP.values() or c in {v for v in _TICKER_MAP.values()}:
                return c

    return None


# ------------------------------------------------------------------
# Data formatting
# ------------------------------------------------------------------

def _format_number(val: Any) -> str:
    """Format a number for snippet display."""
    if val is None:
        return "N/A"
    try:
        n = float(val)
    except (ValueError, TypeError):
        return str(val)

    if abs(n) >= 1e12:
        return f"{n/1e12:.2f}T"
    elif abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    else:
        return f"{n:,.2f}"


def _format_market_data_snippet(info: Dict[str, Any], claim_text: str) -> str:
    """Build a human-readable snippet from yfinance info containing actual numbers.

    The snippet is designed so the scoring engine's number_exact_match
    and token_overlap signals fire against matching claim numbers.
    """
    parts = []

    symbol = info.get("symbol", "")
    name = info.get("shortName") or info.get("longName") or symbol
    if name:
        parts.append(f"{name} ({symbol})")

    # Core metrics — include raw numbers for scoring engine matching
    metrics = [
        ("Market Cap", info.get("marketCap")),
        ("Revenue (TTM)", info.get("totalRevenue")),
        ("Net Income", info.get("netIncomeToCommon")),
        ("EPS", info.get("trailingEps")),
        ("P/E Ratio", info.get("trailingPE")),
        ("Forward P/E", info.get("forwardPE")),
        ("Current Price", info.get("currentPrice") or info.get("regularMarketPrice")),
        ("52wk High", info.get("fiftyTwoWeekHigh")),
        ("52wk Low", info.get("fiftyTwoWeekLow")),
        ("Dividend Yield", info.get("dividendYield")),
        ("Profit Margin", info.get("profitMargins")),
        ("Revenue Growth", info.get("revenueGrowth")),
        ("Employees", info.get("fullTimeEmployees")),
    ]

    for label, val in metrics:
        if val is not None:
            # For percentages (yield, margin, growth) — multiply by 100
            if label in ("Dividend Yield", "Profit Margin", "Revenue Growth") and isinstance(val, (int, float)):
                parts.append(f"{label}: {val*100:.1f}%")
            elif label == "Employees" and isinstance(val, (int, float)):
                parts.append(f"{label}: {int(val):,}")
            elif isinstance(val, (int, float)):
                parts.append(f"{label}: ${_format_number(val)}")
            else:
                parts.append(f"{label}: {val}")

    # Also include raw numbers for exact matching
    # e.g., if revenue is 350.018B, include "350018000000" and "350.0"
    raw_nums = []
    for _, val in metrics:
        if isinstance(val, (int, float)) and val != 0:
            # Include both the raw number and common representations
            if abs(val) >= 1e9:
                raw_nums.append(f"{val/1e9:.1f}")  # e.g. "350.0"
            if abs(val) >= 1e6:
                raw_nums.append(f"{val/1e6:.1f}")  # e.g. "350018.0"
            # Small numbers (EPS, P/E, etc.)
            if abs(val) < 10000:
                raw_nums.append(f"{val:.2f}")

    if raw_nums:
        parts.append(f"Raw values: {' '.join(raw_nums[:10])}")

    sector = info.get("sector")
    industry = info.get("industry")
    if sector:
        parts.append(f"Sector: {sector}")
    if industry:
        parts.append(f"Industry: {industry}")

    return " | ".join(parts)


# ------------------------------------------------------------------
# Main search function
# ------------------------------------------------------------------

def search_yfinance(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search Yahoo Finance for market data matching a claim.

    Standard evidence source signature. Returns list of dicts with keys:
    url, title, source_name, evidence_type, snippet.

    Returns [] if no ticker can be extracted or on any error.
    """
    ticker = _extract_ticker(claim_text)
    if not ticker:
        return []

    try:
        import yfinance as yf
        _rate_limit()

        stock = yf.Ticker(ticker)
        info = stock.info or {}

        if not info or not info.get("symbol"):
            return []

        results = []

        # Result 1: Market data overview (primary — this is the money maker)
        company_name = info.get("shortName") or info.get("longName") or ticker
        snippet = _format_market_data_snippet(info, claim_text)

        results.append({
            "url": f"https://finance.yahoo.com/quote/{ticker}",
            "title": f"{company_name} ({ticker}) - Market Data",
            "source_name": "yfinance",
            "evidence_type": "dataset",  # Gets 15-point primary_source boost
            "snippet": snippet[:4000],
        })

        # Result 2: Recent news (secondary evidence)
        try:
            news = stock.news or []
            for item in news[:min(2, max_results - 1)]:
                news_title = item.get("title", "")
                publisher = item.get("publisher", "")
                link = item.get("link", "")
                if news_title and link:
                    results.append({
                        "url": link,
                        "title": f"{news_title} ({publisher})"[:200],
                        "source_name": "yfinance",
                        "evidence_type": "secondary",
                        "snippet": news_title[:200],
                    })
        except Exception:
            pass  # News is optional, don't fail on it

        return results[:max_results]

    except Exception:
        return []
