"""Evidence source modules — free API clients for assisted verification."""

from .crossref import search_crossref
from .arxiv import search_arxiv
from .pubmed import search_pubmed
from .sec_edgar import search_sec_edgar
from .yfinance_source import search_yfinance
from .wikipedia_source import search_wikipedia
from .fred_source import search_fred
from .google_factcheck import search_google_factcheck
from .openfda import search_openfda
from .bls import search_bls
from .cbo import search_cbo
from .usaspending import search_usaspending
from .census import search_census
from .worldbank import search_worldbank
from .patentsview import search_patentsview
from .local_datasets import search_local_datasets
from .sec_gov import search_sec_gov

ALL_SOURCES = [
    ("local_dataset", search_local_datasets),
    ("crossref", search_crossref),
    ("arxiv", search_arxiv),
    ("pubmed", search_pubmed),
    ("sec_edgar", search_sec_edgar),
    ("sec_gov", search_sec_gov),
    ("yfinance", search_yfinance),
    ("wikipedia", search_wikipedia),
    ("fred", search_fred),
    ("google_factcheck", search_google_factcheck),
    ("openfda", search_openfda),
    ("bls", search_bls),
    ("cbo", search_cbo),
    ("usaspending", search_usaspending),
    ("census", search_census),
    ("worldbank", search_worldbank),
    ("patentsview", search_patentsview),
]
