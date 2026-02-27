"""SEC EDGAR full-text search + filing enrichment.

Pipeline:
  1. Search EDGAR EFTS for filings matching claim + entity name
  2. Pick top candidates (10-K, 10-Q, 8-K)
  3. Fetch filing HTML, extract text snippet for scoring
  4. Cache fetched filings to avoid re-downloading

Docs: https://efts.sec.gov/LATEST/search-index?q=...
No API key required.  Rate limit: 10 req/sec.
"""

from __future__ import annotations
import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base import rate_limited_get, build_search_query

_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# SEC requires User-Agent format: "Company AdminContact@domain"
# See https://www.sec.gov/os/accessing-edgar-data
_SEC_UA = "Veritas Research Tool research@veritas-app.local"

# Cache dir for fetched filing text
_CACHE_DIR: Optional[Path] = None


def _get_cache_dir() -> Path:
    """Lazy-init cache dir under data/cache/edgar/."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        from .. import config
        _CACHE_DIR = config.DATA_DIR / "cache" / "edgar"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


# ------------------------------------------------------------------
# Entity injection: infer company name from source metadata
# ------------------------------------------------------------------

_ENTITY_ALIASES: Dict[str, List[str]] = {
    "alphabet": ["Alphabet", "Google", "GOOG"],
    "meta": ["Meta", "Facebook", "META"],
    "amazon": ["Amazon", "AMZN"],
    "microsoft": ["Microsoft", "MSFT"],
    "apple": ["Apple", "AAPL"],
    "nvidia": ["Nvidia", "NVDA"],
    "tesla": ["Tesla", "TSLA"],
}


def infer_source_entity(title: str, channel: str = "") -> str:
    """Extract a company/entity name from source metadata.

    Returns a short entity string to inject into EDGAR queries,
    or empty string if none found.
    """
    combined = f"{title} {channel}".strip()
    if not combined:
        return ""

    combined_lower = combined.lower()
    for key, aliases in _ENTITY_ALIASES.items():
        # Check the canonical key (e.g. "alphabet")
        if key in combined_lower:
            return aliases[0]
        # Also check all alias values (e.g. "Google", "GOOG")
        for alias in aliases:
            if alias.lower() in combined_lower:
                return aliases[0]

    # Fallback: first capitalized entity from title
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', title)
    skip = {"The", "How", "Why", "What", "This", "That", "New", "Free", "Open"}
    entities = [e for e in entities if e.split()[0] not in skip]
    if entities:
        return entities[0]

    return ""


# ------------------------------------------------------------------
# HTML text extraction (lightweight, no dependencies)
# ------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style tags."""

    def __init__(self):
        super().__init__()
        self._text_parts: List[str] = []
        self._skip = False
        self._skip_tags = {"script", "style", "head", "meta", "link"}

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self._skip_tags:
            self._skip = True

    def handle_endtag(self, tag):
        if tag.lower() in self._skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._text_parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._text_parts)


def _html_to_text(html: str, max_chars: int = 60000) -> str:
    """Convert HTML to plain text, return at most max_chars."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = parser.get_text()
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_chars]


# ------------------------------------------------------------------
# Filing text fetch + cache
# ------------------------------------------------------------------

def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _fetch_filing_text(filing_url: str) -> str:
    """Fetch a filing's primary document HTML and extract text.

    Uses index.json to find the primary .htm document, then fetches and parses it.
    Caches to disk. Returns full extracted text (up to 60K chars).
    """
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / f"{_cache_key(filing_url)}.txt"

    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="replace")

    base_url = filing_url.rstrip("/")
    text = ""

    # Strategy 1: Use index.json to find the primary .htm document
    index_json_url = base_url + "/index.json"
    resp = rate_limited_get(
        index_json_url,
        source_name="sec_edgar_fetch",
        timeout=15.0,
        headers={
            "User-Agent": _SEC_UA,
            "Accept": "application/json",
        },
    )
    if resp is not None:
        try:
            data = resp.json()
            items = data.get("directory", {}).get("item", [])
            # Find .htm files, excluding index/R-pages/xbrl viewer files
            htm_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "")
                if not name.endswith(".htm"):
                    continue
                name_lower = name.lower()
                # Skip index, XBRL viewer (R1.htm, R2.htm), and report files
                if "index" in name_lower or name_lower.startswith("r") and name_lower[1:].split(".")[0].isdigit():
                    continue
                size = 0
                try:
                    size = int(item.get("size", 0))
                except (ValueError, TypeError):
                    pass
                htm_items.append((name, size))

            # Sort: exhibit files first (press releases), then by size descending
            def _sort_key(ns):
                name, size = ns
                nl = name.lower()
                is_exhibit = "exhibit" in nl or "ex99" in nl or "ex-99" in nl
                return (0 if is_exhibit else 1, -size)

            htm_items.sort(key=_sort_key)

            # Try top candidates — need real financial text (not just XBRL metadata)
            for fname, _ in htm_items[:4]:
                doc_url = f"{base_url}/{fname}"
                doc_resp = rate_limited_get(
                    doc_url,
                    source_name="sec_edgar_fetch",
                    timeout=25.0,
                    headers={
                        "User-Agent": _SEC_UA,
                        "Accept": "text/html",
                    },
                )
                if doc_resp is not None and len(doc_resp.text) > 500:
                    candidate_text = _html_to_text(doc_resp.text)
                    # Require meaningful financial content (not just XBRL codes)
                    has_financial_words = any(
                        w in candidate_text.lower()
                        for w in ("revenue", "income", "earnings", "operating", "quarter", "fiscal")
                    )
                    if len(candidate_text) > 1000 and has_financial_words:
                        text = candidate_text
                        break
                    # Fallback: accept any text > 5000 chars (likely a real filing)
                    if len(candidate_text) > 5000:
                        text = candidate_text
                        break
        except Exception:
            pass

    # Strategy 2: Fallback — parse the filing index HTML page for /Archives/ links
    if not text:
        resp2 = rate_limited_get(
            filing_url,
            source_name="sec_edgar_fetch",
            timeout=20.0,
            headers={
                "User-Agent": _SEC_UA,
                "Accept": "text/html",
            },
        )
        if resp2 is not None:
            html = resp2.text
            # Only match links in the Archives path (actual filing docs, not nav)
            archive_links = re.findall(
                r'href="(/Archives/[^"]+\.htm[l]?)"', html, re.I
            )
            for link in archive_links[:3]:
                doc_url = f"https://www.sec.gov{link}"
                doc_resp = rate_limited_get(
                    doc_url,
                    source_name="sec_edgar_fetch",
                    timeout=25.0,
                    headers={
                        "User-Agent": _SEC_UA,
                        "Accept": "text/html",
                    },
                )
                if doc_resp is not None and len(doc_resp.text) > 500:
                    text = _html_to_text(doc_resp.text)
                    if len(text) > 200:
                        break

    if text:
        try:
            cache_file.write_text(text, encoding="utf-8")
        except Exception:
            pass

    return text


# ------------------------------------------------------------------
# Snippet extraction: find the best matching window in filing text
# ------------------------------------------------------------------

def extract_relevant_snippet(
    filing_text: str,
    claim_text: str,
    window: int = 4000,
) -> str:
    """Find the section of filing_text most relevant to the claim.

    Priority: exact number matches > key financial terms > fallback to start.
    """
    if not filing_text:
        return ""

    # Extract numbers from claim (e.g., "113.8", "403", "17", "31.6")
    claim_nums = set(re.findall(r'\d+(?:\.\d+)?', claim_text))
    claim_lower = claim_text.lower()
    key_terms = set()
    for w in claim_lower.split():
        w = w.strip(".,!?;:\"'()[]$%")
        if w in {
            "revenue", "revenues", "income", "earnings", "margin", "margins",
            "billion", "million", "percent", "growth", "operating", "net",
            "cash", "flow", "capex", "depreciation", "cloud", "advertising",
            "search", "youtube", "subscriptions", "expenses", "costs",
            "quarter", "quarterly", "annual", "dividend", "repurchase",
            "backlog", "share", "shares", "eps",
        }:
            key_terms.add(w)

    best_pos = 0
    best_score = 0
    text_lower = filing_text.lower()
    step = 200
    half_window = window // 2

    for pos in range(0, max(1, len(filing_text) - window), step):
        chunk = text_lower[pos:pos + window]
        score = 0
        for num in claim_nums:
            if num in chunk:
                score += 15  # exact number match is highest priority
        for term in key_terms:
            if term in chunk:
                score += 3
        if score > best_score:
            best_score = score
            best_pos = pos

    start = max(0, best_pos)
    end = min(len(filing_text), start + window)
    return filing_text[start:end].strip()


# ------------------------------------------------------------------
# Main search function
# ------------------------------------------------------------------

def _compute_date_range(claim_date: str, upload_date: str) -> tuple[str, str]:
    """Compute EDGAR date range from claim/source temporal context.

    Priority: claim_date (year in claim text) > upload_date (source upload).
    Falls back to 2018-2026 if no temporal context.
    Returns (startdt, enddt) as YYYY-MM-DD strings.
    """
    anchor_year = None
    if claim_date and claim_date.isdigit() and len(claim_date) == 4:
        anchor_year = int(claim_date)
    elif upload_date:
        # yt-dlp format: YYYYMMDD or ISO date
        try:
            anchor_year = int(upload_date[:4])
        except (ValueError, IndexError):
            pass

    if anchor_year and 1990 <= anchor_year <= 2030:
        return f"{anchor_year - 1}-01-01", f"{anchor_year + 1}-12-31"

    return "2018-01-01", "2026-12-31"


def search_sec_edgar(
    claim_text: str,
    max_results: int = 5,
    source_entity: str = "",
    enrich: bool = True,
    claim_date: str = "",
    upload_date: str = "",
) -> List[Dict[str, Any]]:
    """Search SEC EDGAR for filings matching a claim.

    Args:
        claim_text: The claim to search for.
        max_results: Max filing hits to return.
        source_entity: Company name from source metadata (injected into query).
        enrich: If True, fetch filing text and extract snippet for scoring.
        claim_date: Year extracted from claim text (e.g. "2022").
        upload_date: Source upload date from yt-dlp (e.g. "20250204").

    Returns list of dicts with keys: url, title, source_name, evidence_type, snippet.
    """
    query = build_search_query(claim_text, max_terms=6)
    if not query:
        return []

    # Entity injection: prepend company name if not already present
    if source_entity and source_entity.lower() not in query.lower():
        query = f"{source_entity} {query}"

    startdt, enddt = _compute_date_range(claim_date, upload_date)

    resp = rate_limited_get(
        _SEARCH_URL,
        source_name="sec_edgar",
        params={
            "q": query,
            "dateRange": "custom",
            "startdt": startdt,
            "enddt": enddt,
        },
        headers={
            "User-Agent": _SEC_UA,
            "Accept": "application/json",
        },
    )
    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    hits = data.get("hits", {}).get("hits", [])
    results = []
    for hit in hits[:max_results]:
        src = hit.get("_source", {})
        file_date = src.get("file_date", "")
        form = src.get("form", "")
        if not form and src.get("root_forms"):
            form = src["root_forms"][0]
        period = src.get("period_ending", "")

        display_names = src.get("display_names", [])
        entity_name = display_names[0].split("(CIK")[0].strip() if display_names else ""

        adsh = src.get("adsh", "")
        ciks = src.get("ciks", [])
        cik = ciks[0] if ciks else ""
        if adsh and cik:
            adsh_clean = adsh.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_clean}/"
        elif adsh:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&accession={adsh}"
        else:
            url = ""

        title = f"{entity_name} - {form}" if entity_name else form
        meta_snippet = ""
        if file_date:
            meta_snippet += f"Filed: {file_date}"
        if period:
            meta_snippet += f" | Period: {period}"

        if url and title:
            results.append({
                "url": url,
                "title": title[:200],
                "source_name": "sec_edgar",
                "evidence_type": "filing",
                "snippet": meta_snippet[:200],
                "evidence_date": file_date[:4] if file_date else "",
            })

    # Enrichment: fetch filing text and extract relevant snippet
    if enrich and results:
        # Enrich top 2 filings only (rate limit + time budget)
        seen: set = set()
        for r in results[:2]:
            filing_url = r["url"]
            if filing_url in seen:
                continue
            seen.add(filing_url)

            filing_text = _fetch_filing_text(filing_url)
            if filing_text and len(filing_text) > 200:
                snippet = extract_relevant_snippet(filing_text, claim_text, window=4000)
                if snippet and len(snippet) > 50:
                    r["snippet"] = snippet[:4000]
                    r["_enriched"] = True
                    r["_snippet_len"] = len(snippet)

    return results
