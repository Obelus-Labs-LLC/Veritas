"""Tests for 7 new evidence sources (Step 2 expansion).

OpenFDA, BLS, CBO, USASpending, Census, World Bank, PatentsView.
All tests mock external HTTP calls.
"""

import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")


# ===========================================================================
# Registry tests — all 7 sources registered in __init__.py
# ===========================================================================

class TestRegistryExpansion:
    def test_all_sources_count_is_20(self):
        from veritas.evidence_sources import ALL_SOURCES
        assert len(ALL_SOURCES) == 20

    def test_new_sources_all_registered(self):
        from veritas.evidence_sources import ALL_SOURCES
        names = [name for name, _ in ALL_SOURCES]
        for expected in ("openfda", "bls", "cbo", "usaspending", "census", "worldbank",
                         "patentsview", "wikidata", "duckduckgo", "semantic_scholar"):
            assert expected in names, f"{expected} not found in ALL_SOURCES"

    def test_original_sources_still_present(self):
        from veritas.evidence_sources import ALL_SOURCES
        names = [name for name, _ in ALL_SOURCES]
        for expected in ("crossref", "arxiv", "pubmed", "sec_edgar", "yfinance", "wikipedia", "fred", "google_factcheck"):
            assert expected in names, f"{expected} missing from ALL_SOURCES"


# ===========================================================================
# Routing tests — new sources in category priorities
# ===========================================================================

class TestRoutingExpansion:
    def setup_method(self):
        from veritas.assist import _select_sources_for_category
        self.select = _select_sources_for_category

    def test_finance_includes_bls_and_cbo(self):
        sources = self.select("finance")
        names = [n for n, _ in sources]
        assert "bls" in names
        assert "cbo" in names

    def test_health_includes_openfda(self):
        sources = self.select("health")
        names = [n for n, _ in sources]
        assert "openfda" in names
        # openfda should be near the top for health
        assert names.index("openfda") < names.index("wikipedia")

    def test_tech_includes_patentsview(self):
        sources = self.select("tech")
        names = [n for n, _ in sources]
        assert "patentsview" in names

    def test_politics_includes_cbo_and_usaspending(self):
        sources = self.select("politics")
        names = [n for n, _ in sources]
        assert "cbo" in names
        assert "usaspending" in names

    def test_labor_category_routing(self):
        sources = self.select("labor")
        names = [n for n, _ in sources]
        assert "bls" in names
        # local_dataset is first for all categories, BLS is second for labor
        assert names[0] == "local_dataset"
        assert names[1] == "bls"

    def test_education_category_routing(self):
        sources = self.select("education")
        names = [n for n, _ in sources]
        assert "census" in names
        assert "worldbank" in names

    def test_energy_climate_category_routing(self):
        sources = self.select("energy_climate")
        names = [n for n, _ in sources]
        assert "worldbank" in names

    def test_general_includes_new_sources(self):
        sources = self.select("general")
        names = [n for n, _ in sources]
        assert "bls" in names
        assert "census" in names


# ===========================================================================
# Smart routing signal tests
# ===========================================================================

class TestSmartRoutingSignals:
    def setup_method(self):
        from veritas.assist import _smart_select_sources, _select_sources_for_category
        self.smart = _smart_select_sources
        self.select = _select_sources_for_category

    def test_drug_terms_boost_openfda(self):
        sources = self.select("health")
        reranked = self.smart("the drug caused adverse side effects in patients", "health", sources)
        names = [n for n, _ in reranked]
        assert "openfda" in names
        # openfda should be boosted near top
        assert names.index("openfda") <= 2

    def test_labor_terms_boost_bls(self):
        sources = self.select("general")
        reranked = self.smart("the unemployment rate and employment numbers dropped", "general", sources)
        names = [n for n, _ in reranked]
        assert "bls" in names

    def test_spending_terms_boost_cbo_and_usaspending(self):
        sources = self.select("politics")
        reranked = self.smart("federal spending on the deficit and national debt is rising", "politics", sources)
        names = [n for n, _ in reranked]
        assert "cbo" in names
        assert "usaspending" in names

    def test_demographics_boost_census(self):
        sources = self.select("general")
        reranked = self.smart("the population and median income in California", "general", sources)
        names = [n for n, _ in reranked]
        assert "census" in names

    def test_international_terms_boost_worldbank(self):
        sources = self.select("science")
        reranked = self.smart("global GDP growth and trade in developing countries", "science", sources)
        names = [n for n, _ in reranked]
        assert "worldbank" in names

    def test_patent_terms_boost_patentsview(self):
        sources = self.select("tech")
        reranked = self.smart("the company filed a patent for their innovation", "tech", sources)
        names = [n for n, _ in reranked]
        assert "patentsview" in names


# ===========================================================================
# OpenFDA tests
# ===========================================================================

class TestOpenFDA:
    def setup_method(self):
        from veritas.evidence_sources.openfda import search_openfda, _pick_endpoint
        self.search = search_openfda
        self.pick_endpoint = _pick_endpoint

    def test_picks_adverse_event_endpoint(self):
        assert self.pick_endpoint("adverse reactions to the drug") == "/drug/event.json"

    def test_picks_recall_endpoint(self):
        assert self.pick_endpoint("the product was recalled by FDA") == "/food/enforcement.json"

    def test_picks_approval_endpoint(self):
        assert self.pick_endpoint("FDA approved the new treatment") == "/drug/drugsfda.json"

    def test_default_endpoint_is_drug_event(self):
        assert self.pick_endpoint("something about health") == "/drug/event.json"

    @patch("veritas.evidence_sources.openfda.rate_limited_get")
    def test_returns_results_from_api(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{
                "patient": {
                    "reaction": [{"reactionmeddrapt": "Nausea"}],
                    "drug": [{"medicinalproduct": "Aspirin"}],
                },
                "serious": 1,
                "occurcountry": "US",
            }]
        }
        mock_get.return_value = mock_resp

        results = self.search("adverse effects of the drug aspirin")
        assert len(results) == 1
        assert results[0]["source_name"] == "openfda"
        assert results[0]["evidence_type"] == "gov"
        assert "Aspirin" in results[0]["snippet"]

    @patch("veritas.evidence_sources.openfda.rate_limited_get")
    def test_returns_empty_on_failure(self, mock_get):
        mock_get.return_value = None
        results = self.search("drug side effects from medication")
        assert results == []

    def test_returns_empty_on_empty_input(self):
        results = self.search("")
        assert results == []


# ===========================================================================
# BLS tests
# ===========================================================================

class TestBLS:
    def setup_method(self):
        from veritas.evidence_sources.bls import search_bls, _match_series
        self.search = search_bls
        self.match = _match_series

    def test_matches_unemployment(self):
        match = self.match("the unemployment rate is 3.5%")
        assert match is not None
        assert match[0] == "LNS14000000"

    def test_matches_cpi(self):
        match = self.match("CPI rose to 301")
        assert match is not None
        assert match[0] == "CUUR0000SA0"

    def test_matches_wages(self):
        match = self.match("average hourly earnings wages went up")
        assert match is not None

    def test_no_match_for_unrelated(self):
        match = self.match("Apple stock price went up today")
        assert match is None

    @patch("veritas.evidence_sources.bls.rate_limited_get")
    def test_returns_results_from_api(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [{
                    "data": [
                        {"year": "2024", "periodName": "December", "value": "4.1"},
                        {"year": "2024", "periodName": "November", "value": "4.2"},
                    ]
                }]
            }
        }
        mock_get.return_value = mock_resp

        results = self.search("unemployment rate in the US")
        assert len(results) == 1
        assert results[0]["source_name"] == "bls"
        assert results[0]["evidence_type"] == "gov"
        assert "4.1" in results[0]["snippet"]

    @patch("veritas.evidence_sources.bls.rate_limited_get")
    def test_returns_basic_snippet_on_api_failure(self, mock_get):
        mock_get.return_value = None
        results = self.search("unemployment rate trends")
        assert len(results) == 1
        assert "Bureau of Labor Statistics" in results[0]["snippet"]

    def test_returns_empty_for_unrelated_claim(self):
        results = self.search("Apple reported record revenue")
        assert results == []


# ===========================================================================
# CBO tests
# ===========================================================================

class TestCBO:
    def setup_method(self):
        from veritas.evidence_sources.cbo import search_cbo, _has_cbo_relevance
        self.search = search_cbo
        self.has_relevance = _has_cbo_relevance

    def test_has_relevance_for_budget(self):
        assert self.has_relevance("the federal budget deficit grew")

    def test_has_relevance_for_social_security(self):
        assert self.has_relevance("Social Security spending is unsustainable")

    def test_no_relevance_for_unrelated(self):
        assert not self.has_relevance("Apple stock price went up")

    @patch("veritas.evidence_sources.cbo.rate_limited_get")
    def test_returns_results_from_govinfo(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [{
                "title": "The Budget and Economic Outlook: 2024 to 2034",
                "packageLink": "https://www.govinfo.gov/content/pkg/BUDGET-2024",
                "dateIssued": "2024-02-07",
                "docClass": "BUDGET",
            }]
        }
        mock_get.return_value = mock_resp

        results = self.search("federal budget deficit projection")
        assert len(results) == 1
        assert results[0]["source_name"] == "cbo"
        assert results[0]["evidence_type"] == "gov"
        assert "Budget" in results[0]["title"]

    @patch("veritas.evidence_sources.cbo.rate_limited_get")
    def test_fallback_to_cbo_search_link(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_get.return_value = mock_resp

        results = self.search("CBO cost estimate for Medicare")
        assert len(results) == 1
        assert "cbo.gov" in results[0]["url"]

    def test_returns_empty_for_unrelated(self):
        results = self.search("Tesla stock price today")
        assert results == []


# ===========================================================================
# USASpending tests
# ===========================================================================

class TestUSASpending:
    def setup_method(self):
        from veritas.evidence_sources.usaspending import search_usaspending, _has_spending_relevance
        self.search = search_usaspending
        self.has_relevance = _has_spending_relevance

    def test_has_relevance_for_contracts(self):
        assert self.has_relevance("government contracts worth billions")

    def test_has_relevance_for_spending(self):
        assert self.has_relevance("federal spending on defense increased")

    def test_no_relevance_for_unrelated(self):
        assert not self.has_relevance("Apple launched a new iPhone")

    @patch("veritas.evidence_sources.usaspending.requests.post")
    def test_returns_results_from_api(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [{
                "Recipient Name": "Lockheed Martin",
                "Award Amount": 5000000000,
                "Awarding Agency": "Department of Defense",
                "Description": "Fighter jet contract",
                "Award ID": "ABC123",
            }]
        }
        mock_post.return_value = mock_resp

        results = self.search("defense spending on military contracts")
        assert len(results) >= 1
        assert results[0]["source_name"] == "usaspending"
        assert results[0]["evidence_type"] == "gov"

    @patch("veritas.evidence_sources.usaspending.requests.post")
    def test_returns_empty_on_api_failure(self, mock_post):
        mock_post.side_effect = Exception("Network error")
        results = self.search("government spending on infrastructure")
        assert results == []

    def test_returns_empty_for_unrelated(self):
        results = self.search("Nvidia released new GPU")
        assert results == []


# ===========================================================================
# Census tests
# ===========================================================================

class TestCensus:
    def setup_method(self):
        from veritas.evidence_sources.census import search_census, _match_query, _extract_state
        self.search = search_census
        self.match = _match_query
        self.extract_state = _extract_state

    def test_matches_population(self):
        match = self.match("the population of the United States")
        assert match is not None
        assert "B01003_001E" in match["variables"]

    def test_matches_median_income(self):
        match = self.match("median income has risen")
        assert match is not None
        assert "B19013_001E" in match["variables"]

    def test_matches_poverty(self):
        match = self.match("poverty rate in America")
        assert match is not None

    def test_no_match_for_unrelated(self):
        match = self.match("Apple stock went up")
        assert match is None

    def test_extracts_california(self):
        fips = self.extract_state("poverty in California is high")
        assert fips == "06"

    def test_extracts_texas(self):
        fips = self.extract_state("population of Texas")
        assert fips == "48"

    def test_no_state_extraction(self):
        fips = self.extract_state("national population growth")
        assert fips is None

    @patch("veritas.evidence_sources.census.rate_limited_get")
    def test_returns_results_from_api(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            ["NAME", "B01003_001E", "us"],
            ["United States", "331449281", "1"],
        ]
        mock_get.return_value = mock_resp

        results = self.search("US population total")
        assert len(results) == 1
        assert results[0]["source_name"] == "census"
        assert results[0]["evidence_type"] == "gov"
        assert "331,449,281" in results[0]["snippet"]

    @patch("veritas.evidence_sources.census.rate_limited_get")
    def test_returns_basic_snippet_on_failure(self, mock_get):
        mock_get.return_value = None
        results = self.search("population of America")
        assert len(results) == 1
        assert "Census Bureau" in results[0]["snippet"]

    def test_returns_empty_for_unrelated(self):
        results = self.search("Nvidia GPU performance benchmarks")
        assert results == []


# ===========================================================================
# World Bank tests
# ===========================================================================

class TestWorldBank:
    def setup_method(self):
        from veritas.evidence_sources.worldbank import search_worldbank, _match_indicator, _extract_country
        self.search = search_worldbank
        self.match = _match_indicator
        self.extract_country = _extract_country

    def test_matches_gdp(self):
        match = self.match("GDP of China")
        assert match is not None
        assert match[0] == "NY.GDP.MKTP.CD"

    def test_matches_life_expectancy(self):
        match = self.match("life expectancy in Japan")
        assert match is not None
        assert match[0] == "SP.DYN.LE00.IN"

    def test_matches_co2(self):
        match = self.match("co2 emissions from India")
        assert match is not None

    def test_no_match_for_unrelated(self):
        match = self.match("Apple reported quarterly earnings")
        assert match is None

    def test_extracts_china(self):
        assert self.extract_country("GDP of China is growing") == "CN"

    def test_extracts_us(self):
        assert self.extract_country("GDP of the United States") == "US"

    def test_defaults_to_world(self):
        assert self.extract_country("global GDP growth") == "WLD"

    @patch("veritas.evidence_sources.worldbank.rate_limited_get")
    def test_returns_results_from_api(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"page": 1, "pages": 1, "total": 3},
            [
                {"date": "2023", "value": 17963171000000, "country": {"value": "China"}},
                {"date": "2022", "value": 17886000000000, "country": {"value": "China"}},
            ]
        ]
        mock_get.return_value = mock_resp

        results = self.search("GDP of China")
        assert len(results) == 1
        assert results[0]["source_name"] == "worldbank"
        assert results[0]["evidence_type"] == "dataset"
        assert "China" in results[0]["snippet"]

    @patch("veritas.evidence_sources.worldbank.rate_limited_get")
    def test_returns_basic_snippet_on_failure(self, mock_get):
        mock_get.return_value = None
        results = self.search("GDP of China")
        assert len(results) == 1
        assert "World Bank" in results[0]["snippet"]

    def test_returns_empty_for_unrelated(self):
        results = self.search("Tesla stock price went up")
        assert results == []


# ===========================================================================
# PatentsView tests
# ===========================================================================

class TestPatentsView:
    def setup_method(self):
        from veritas.evidence_sources.patentsview import search_patentsview, _has_patent_relevance
        self.search = search_patentsview
        self.has_relevance = _has_patent_relevance

    def test_has_relevance_for_patent(self):
        assert self.has_relevance("the company filed several patents")

    def test_has_relevance_for_innovation(self):
        assert self.has_relevance("innovation in artificial intelligence")

    def test_no_relevance_for_unrelated(self):
        assert not self.has_relevance("the stock price went up today")

    def test_returns_fallback_link_without_api_key(self):
        """Without PATENTSVIEW_API_KEY, should return a reference link."""
        # Clear the API key for this test
        import veritas.evidence_sources.patentsview as pv
        original_key = pv._API_KEY
        pv._API_KEY = ""
        try:
            results = self.search("Apple filed a patent for their innovation")
            assert len(results) == 1
            assert results[0]["source_name"] == "patentsview"
            assert results[0]["evidence_type"] == "gov"
            assert "patentsview.org" in results[0]["url"]
        finally:
            pv._API_KEY = original_key

    def test_returns_empty_for_unrelated(self):
        results = self.search("the weather is nice today")
        assert results == []

    @patch("veritas.evidence_sources.patentsview.rate_limited_get")
    def test_returns_results_with_api_key(self, mock_get):
        import veritas.evidence_sources.patentsview as pv
        original_key = pv._API_KEY
        pv._API_KEY = "test-key"
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "patents": [{
                    "patent_id": "12345678",
                    "patent_title": "Machine Learning System",
                    "patent_date": "2024-03-15",
                    "patent_abstract": "A system for automated learning",
                }]
            }
            mock_get.return_value = mock_resp

            results = self.search("patent for machine learning innovation")
            assert len(results) == 1
            assert results[0]["source_name"] == "patentsview"
            assert "12345678" in results[0]["url"]
        finally:
            pv._API_KEY = original_key


# ===========================================================================
# Scoring integration — new source evidence types
# ===========================================================================

class TestNewSourceScoring:
    def setup_method(self):
        from veritas.scoring import score_evidence
        self.score = score_evidence

    def test_gov_type_gets_primary_boost(self):
        """evidence_type='gov' should get the 15-point primary_source boost."""
        score, signals = self.score(
            claim_text="unemployment rate dropped to 3.5 percent",
            claim_category="labor",
            evidence_title="BLS: Unemployment Rate, Seasonally Adjusted",
            evidence_snippet="Unemployment Rate. Recent values: December 2024: 4.1; November 2024: 4.2",
            evidence_type="gov",
            source_name="bls",
        )
        assert "primary_source:gov" in signals
        assert score >= 30

    def test_dataset_type_gets_primary_boost(self):
        """evidence_type='dataset' should get the 15-point primary_source boost."""
        score, signals = self.score(
            claim_text="GDP of China reached 17 trillion dollars",
            claim_category="finance",
            evidence_title="World Bank: GDP (current US$) (CN)",
            evidence_snippet="GDP (current US$) - China. Data: 2023: $17963.2B; 2022: $17886.0B",
            evidence_type="dataset",
            source_name="worldbank",
        )
        assert "primary_source:dataset" in signals
        assert score >= 30


# ===========================================================================
# DuckDuckGo Instant Answers tests
# ===========================================================================

class TestDuckDuckGo:
    def setup_method(self):
        from veritas.evidence_sources.duckduckgo import (
            search_duckduckgo, _extract_ddg_queries, _parse_ddg_response
        )
        self.search = search_duckduckgo
        self.extract_queries = _extract_ddg_queries
        self.parse_response = _parse_ddg_response

    def test_extract_multi_word_entity(self):
        queries = self.extract_queries("Steve Jobs co-founded Apple")
        assert "Steve Jobs" in queries

    def test_extract_acronym(self):
        queries = self.extract_queries("NVIDIA designs graphics cards")
        assert "NVIDIA" in queries

    def test_extract_sentence_start_entity(self):
        queries = self.extract_queries("Apple was founded in 1976")
        assert "Apple" in queries

    def test_extract_fallback_keywords(self):
        queries = self.extract_queries("the speed of light in a vacuum")
        assert len(queries) >= 1
        assert "speed" in queries[0]

    def test_extract_empty_returns_empty(self):
        queries = self.extract_queries("")
        assert queries == []

    def test_parse_abstract_response(self):
        data = {
            "Abstract": "Apple Inc. is an American multinational technology company.",
            "AbstractURL": "https://en.wikipedia.org/wiki/Apple_Inc.",
            "AbstractSource": "Wikipedia",
            "Heading": "Apple Inc.",
        }
        results = self.parse_response(data, 5)
        assert len(results) == 1
        assert results[0]["source_name"] == "duckduckgo"
        assert results[0]["evidence_type"] == "secondary"
        assert "Apple" in results[0]["snippet"]

    def test_parse_answer_response(self):
        data = {
            "Abstract": "",
            "Answer": "299,792,458 m/s",
            "AbstractURL": "https://duckduckgo.com",
            "Heading": "Speed of Light",
        }
        results = self.parse_response(data, 5)
        assert len(results) == 1
        assert "299,792,458" in results[0]["snippet"]

    def test_parse_related_topics(self):
        data = {
            "Abstract": "",
            "RelatedTopics": [
                {"Text": "Topic about something", "FirstURL": "https://example.com"},
                {"Topics": [{"Text": "Sub-group"}]},  # should be skipped
            ],
        }
        results = self.parse_response(data, 5)
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"

    def test_parse_empty_response(self):
        results = self.parse_response({}, 5)
        assert results == []

    @patch("veritas.evidence_sources.duckduckgo.rate_limited_get")
    def test_search_returns_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "Abstract": "Tesla, Inc. is an American automotive company.",
            "AbstractURL": "https://en.wikipedia.org/wiki/Tesla,_Inc.",
            "AbstractSource": "Wikipedia",
            "Heading": "Tesla, Inc.",
            "RelatedTopics": [],
        }
        mock_get.return_value = mock_resp

        results = self.search("Tesla reported record revenue in Q4")
        assert len(results) >= 1
        assert results[0]["source_name"] == "duckduckgo"
        assert "Tesla" in results[0]["snippet"]

    @patch("veritas.evidence_sources.duckduckgo.rate_limited_get")
    def test_search_returns_empty_on_api_failure(self, mock_get):
        mock_get.return_value = None
        results = self.search("Tesla revenue Q4")
        assert results == []


# ===========================================================================
# Wikidata tests
# ===========================================================================

class TestWikidata:
    def setup_method(self):
        from veritas.evidence_sources.wikidata import (
            search_wikidata, _has_entity_relevance, _extract_entity_query, _format_value
        )
        self.search = search_wikidata
        self.has_relevance = _has_entity_relevance
        self.extract_entity = _extract_entity_query
        self.format_value = _format_value

    def test_has_relevance_multi_word_proper_noun(self):
        assert self.has_relevance("Goldman Sachs is a bank")

    def test_has_relevance_acronym(self):
        assert self.has_relevance("NVIDIA makes GPUs")

    def test_has_relevance_mid_sentence_name(self):
        assert self.has_relevance("the company Apple was founded")

    def test_no_relevance_for_generic(self):
        assert not self.has_relevance("the economy is doing well")

    def test_extract_multi_word_entity(self):
        assert self.extract_entity("Steve Jobs founded Apple") == "Steve Jobs"

    def test_extract_acronym(self):
        assert self.extract_entity("NVIDIA makes chips") == "NVIDIA"

    def test_extract_mid_sentence(self):
        assert self.extract_entity("the company Apple was big") == "Apple"

    def test_extract_empty_for_generic(self):
        assert self.extract_entity("things are going well") == ""

    def test_format_string_value(self):
        snak = {"datavalue": {"type": "string", "value": "hello"}}
        assert self.format_value(snak) == "hello"

    def test_format_time_value(self):
        snak = {"datavalue": {"type": "time", "value": {"time": "+1976-04-01T00:00:00Z"}}}
        assert self.format_value(snak) == "1976-04-01"

    def test_format_quantity_value(self):
        snak = {"datavalue": {"type": "quantity", "value": {"amount": "+5000000", "unit": "1"}}}
        assert self.format_value(snak) == "+5000000"

    @patch("veritas.evidence_sources.wikidata.rate_limited_get")
    def test_search_returns_results(self, mock_get):
        # First call: wbsearchentities
        search_resp = MagicMock()
        search_resp.json.return_value = {
            "search": [{
                "id": "Q312",
                "label": "Apple Inc.",
                "description": "American multinational technology company",
            }]
        }
        # Second call: wbgetentities
        entity_resp = MagicMock()
        entity_resp.json.return_value = {
            "entities": {
                "Q312": {
                    "claims": {
                        "P571": [{
                            "mainsnak": {"datavalue": {"type": "time", "value": {"time": "+1976-04-01T00:00:00Z"}}}
                        }]
                    }
                }
            }
        }
        mock_get.side_effect = [search_resp, entity_resp]

        results = self.search("Apple Inc was founded in 1976")
        assert len(results) >= 1
        assert results[0]["source_name"] == "wikidata"
        assert results[0]["evidence_type"] == "dataset"
        assert "1976-04-01" in results[0]["snippet"]

    @patch("veritas.evidence_sources.wikidata.rate_limited_get")
    def test_search_returns_empty_on_no_entity(self, mock_get):
        mock_get.return_value = None
        results = self.search("things are going well today")
        assert results == []


# ===========================================================================
# Semantic Scholar tests
# ===========================================================================

class TestSemanticScholar:
    def setup_method(self):
        from veritas.evidence_sources.semantic_scholar import (
            search_semantic_scholar, _has_academic_relevance
        )
        self.search = search_semantic_scholar
        self.has_relevance = _has_academic_relevance

    def test_has_relevance_for_gene(self):
        assert self.has_relevance("CRISPR gene editing modifies DNA")

    def test_has_relevance_for_clinical(self):
        assert self.has_relevance("the clinical trial showed efficacy")

    def test_has_relevance_for_research(self):
        assert self.has_relevance("research suggests a correlation")

    def test_has_relevance_for_brain(self):
        assert self.has_relevance("neurons in the brain fire signals")

    def test_no_relevance_for_finance(self):
        assert not self.has_relevance("Apple reported revenue of 25 billion")

    def test_no_relevance_for_generic(self):
        assert not self.has_relevance("we are well positioned for growth")

    def test_relevance_for_entity_with_numbers(self):
        """Named entity + numbers should pass (often cites research)."""
        assert self.has_relevance("The Framingham Study tracked 5209 participants")

    @patch("veritas.evidence_sources.semantic_scholar.rate_limited_get")
    def test_search_returns_papers(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [{
                "paperId": "abc123",
                "title": "CRISPR-Cas9 Gene Editing: A Revolution",
                "abstract": "This paper reviews CRISPR technology.",
                "year": 2023,
                "venue": "Nature Reviews",
                "citationCount": 150,
                "url": "https://www.semanticscholar.org/paper/abc123",
            }]
        }
        mock_get.return_value = mock_resp

        results = self.search("CRISPR gene editing technology")
        assert len(results) == 1
        assert results[0]["source_name"] == "semantic_scholar"
        assert results[0]["evidence_type"] == "paper"
        assert "CRISPR" in results[0]["title"]
        assert "Citations: 150" in results[0]["snippet"]

    @patch("veritas.evidence_sources.semantic_scholar.rate_limited_get")
    def test_search_returns_empty_on_failure(self, mock_get):
        mock_get.return_value = None
        results = self.search("CRISPR gene editing research")
        assert results == []

    def test_search_skips_irrelevant_claims(self):
        """Finance claims should be filtered out at the relevance check."""
        results = self.search("Apple stock price went up today")
        assert results == []


# ===========================================================================
# New source routing + boost signal tests
# ===========================================================================

class TestNewSourceRouting:
    def setup_method(self):
        from veritas.assist import _select_sources_for_category, _smart_select_sources
        self.select = _select_sources_for_category
        self.smart = _smart_select_sources

    def test_general_includes_all_20_sources(self):
        sources = self.select("general")
        names = [n for n, _ in sources]
        assert len(names) == 20

    def test_duckduckgo_in_every_category(self):
        """DuckDuckGo should be in every category as a universal fallback."""
        categories = ["finance", "health", "science", "politics", "tech",
                       "labor", "education", "energy_climate", "general"]
        for cat in categories:
            sources = self.select(cat)
            names = [n for n, _ in sources]
            assert "duckduckgo" in names, f"duckduckgo missing from {cat}"

    def test_wikidata_in_categories(self):
        for cat in ["science", "politics", "tech", "general"]:
            sources = self.select(cat)
            names = [n for n, _ in sources]
            assert "wikidata" in names, f"wikidata missing from {cat}"

    def test_semantic_scholar_in_categories(self):
        for cat in ["science", "health", "general"]:
            sources = self.select(cat)
            names = [n for n, _ in sources]
            assert "semantic_scholar" in names, f"semantic_scholar missing from {cat}"

    def test_named_entity_boosts_wikidata(self):
        sources = self.select("general")
        reranked = self.smart(
            "Albert Einstein developed the theory of relativity", "general", sources
        )
        names = [n for n, _ in reranked]
        assert "wikidata" in names
        # Should be boosted near top
        assert names.index("wikidata") <= 5

    def test_academic_terms_boost_semantic_scholar(self):
        sources = self.select("general")
        reranked = self.smart(
            "the clinical trial demonstrated vaccine efficacy", "general", sources
        )
        names = [n for n, _ in reranked]
        assert "semantic_scholar" in names

    def test_ddg_fallback_when_no_strong_signal(self):
        sources = self.select("general")
        reranked = self.smart(
            "things are going well this quarter", "general", sources
        )
        names = [n for n, _ in reranked]
        assert "duckduckgo" in names
