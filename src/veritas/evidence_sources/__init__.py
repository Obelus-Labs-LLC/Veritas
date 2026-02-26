"""Evidence source modules — free API clients for assisted verification."""

from .crossref import search_crossref
from .arxiv import search_arxiv
from .pubmed import search_pubmed
from .sec_edgar import search_sec_edgar
from .yfinance_source import search_yfinance
from .wikipedia_source import search_wikipedia
from .fred_source import search_fred
from .google_factcheck import search_google_factcheck

ALL_SOURCES = [
    ("crossref", search_crossref),
    ("arxiv", search_arxiv),
    ("pubmed", search_pubmed),
    ("sec_edgar", search_sec_edgar),
    ("yfinance", search_yfinance),
    ("wikipedia", search_wikipedia),
    ("fred", search_fred),
    ("google_factcheck", search_google_factcheck),
]
