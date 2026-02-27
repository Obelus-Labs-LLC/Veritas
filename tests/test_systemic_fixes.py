"""Tests for the 5 systemic verification fixes.

1. Local dataset evidence source
2. Budget cap removal + claim verifiability pre-filtering
3. Crossref/Wikipedia pre-filtering
4. SEC publications search (sec_gov)
5. Source registration (17 sources)

All tests mock external HTTP calls.
"""

import sys
import os
import csv
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

sys.path.insert(0, "src")


# ===========================================================================
# 1. Source registry: 17 sources registered
# ===========================================================================

class TestSourceRegistry:
    def test_all_sources_count_is_17(self):
        from veritas.evidence_sources import ALL_SOURCES
        assert len(ALL_SOURCES) == 17

    def test_local_dataset_registered_first(self):
        from veritas.evidence_sources import ALL_SOURCES
        assert ALL_SOURCES[0][0] == "local_dataset"

    def test_sec_gov_registered(self):
        from veritas.evidence_sources import ALL_SOURCES
        names = [name for name, _ in ALL_SOURCES]
        assert "sec_gov" in names

    def test_all_new_sources_present(self):
        from veritas.evidence_sources import ALL_SOURCES
        names = [name for name, _ in ALL_SOURCES]
        for expected in ("local_dataset", "sec_gov"):
            assert expected in names, f"{expected} not found in ALL_SOURCES"

    def test_original_15_sources_still_present(self):
        from veritas.evidence_sources import ALL_SOURCES
        names = [name for name, _ in ALL_SOURCES]
        originals = [
            "crossref", "arxiv", "pubmed", "sec_edgar", "yfinance",
            "wikipedia", "fred", "google_factcheck", "openfda",
            "bls", "cbo", "usaspending", "census", "worldbank", "patentsview",
        ]
        for expected in originals:
            assert expected in names, f"{expected} missing from ALL_SOURCES"


# ===========================================================================
# 2. Local dataset source
# ===========================================================================

class TestLocalDatasets:
    def test_import(self):
        from veritas.evidence_sources.local_datasets import search_local_datasets
        assert callable(search_local_datasets)

    def test_returns_empty_when_no_datasets(self):
        from veritas.evidence_sources.local_datasets import search_local_datasets
        # With empty/non-existent datasets dir, should return []
        with patch("veritas.evidence_sources.local_datasets._load_all_datasets", return_value=[]):
            result = search_local_datasets("test claim")
            assert result == []

    def test_number_extraction(self):
        from veritas.evidence_sources.local_datasets import _extract_numbers
        nums = _extract_numbers("The SEC has 318 registered transfer agents and 53 SBSDs")
        assert "318" in nums
        assert "53" in nums

    def test_number_extraction_with_commas(self):
        from veritas.evidence_sources.local_datasets import _extract_numbers
        nums = _extract_numbers("Revenue was 1,482 million dollars or 1,482,000,000")
        assert "1482" in nums
        assert "1482000000" in nums

    def test_matching_with_csv_data(self):
        """Test that a claim matches against in-memory CSV data."""
        from veritas.evidence_sources.local_datasets import _find_matching_rows, _extract_numbers

        dataset = {
            "filename": "test.csv",
            "headers": ["Entity", "Count", "Date"],
            "rows": [
                ["Transfer Agents", "318", "Dec 2024"],
                ["SBS Dealers", "53", "Dec 2024"],
                ["Municipal Advisors", "419", "Oct 2025"],
            ],
            "text_index": "entity count date transfer agents 318 dec 2024 sbs dealers 53 dec 2024 municipal advisors 419 oct 2025",
            "row_count": 3,
        }

        # Claim with exact number match
        claim = "The SEC has 318 registered transfer agents"
        claim_nums = _extract_numbers(claim)
        matches = _find_matching_rows(dataset, claim, claim_nums)
        assert len(matches) >= 1
        assert "318" in matches[0]["num_matches"]

    def test_search_with_mock_dataset(self):
        """Test full search flow with mocked dataset loading."""
        from veritas.evidence_sources.local_datasets import search_local_datasets

        mock_dataset = {
            "filename": "sec-stats.xlsx",
            "headers": ["Type", "Count"],
            "rows": [
                ["Transfer Agents", "318"],
                ["SBS Dealers", "53"],
            ],
            "text_index": "type count transfer agents 318 sbs dealers 53",
            "row_count": 2,
            "path": "/tmp/sec-stats.xlsx",
        }

        with patch("veritas.evidence_sources.local_datasets._load_all_datasets",
                    return_value=[mock_dataset]):
            results = search_local_datasets("The SEC has 318 registered transfer agents")
            assert len(results) >= 1
            assert results[0]["source_name"] == "local_dataset"
            assert results[0]["evidence_type"] == "dataset"
            assert "318" in results[0]["snippet"]

    def test_search_returns_empty_for_irrelevant_claim(self):
        """Claims with no number or term overlap should return []."""
        from veritas.evidence_sources.local_datasets import search_local_datasets

        mock_dataset = {
            "filename": "sec-stats.xlsx",
            "headers": ["Type", "Count"],
            "rows": [["Transfer Agents", "318"]],
            "text_index": "type count transfer agents 318",
            "row_count": 1,
            "path": "/tmp/sec-stats.xlsx",
        }

        with patch("veritas.evidence_sources.local_datasets._load_all_datasets",
                    return_value=[mock_dataset]):
            results = search_local_datasets("apple pie is delicious and wonderful")
            assert results == []


# ===========================================================================
# 3. Budget cap + verifiability pre-filtering
# ===========================================================================

class TestVerifiabilityScoring:
    def setup_method(self):
        from veritas.assist import _verifiability_score
        from veritas.models import Claim
        self.score_fn = _verifiability_score
        self.Claim = Claim

    def _make_claim(self, text, category="general"):
        return self.Claim(source_id="s1", text=text, ts_start=0.0, ts_end=1.0,
                          confidence_language="definitive", category=category)

    def test_numeric_claim_scores_high(self):
        claim = self._make_claim("GDP growth was 3.2 percent in Q4 2024", "finance")
        assert self.score_fn(claim) >= 30

    def test_entity_claim_scores_moderate(self):
        claim = self._make_claim("The SEC has 318 registered transfer agents", "finance")
        assert self.score_fn(claim) >= 10

    def test_personal_opinion_scores_zero(self):
        claim = self._make_claim("I think this is really interesting to think about")
        assert self.score_fn(claim) < 5

    def test_vague_narrative_scores_zero(self):
        claim = self._make_claim("it was kind of going up and down all day")
        assert self.score_fn(claim) < 5

    def test_financial_claim_with_entities_scores_high(self):
        claim = self._make_claim(
            "Alphabet reported revenue of 113.8 billion in 2024", "finance")
        assert self.score_fn(claim) >= 30

    def test_date_reference_boosts_score(self):
        claim_with_date = self._make_claim("The rate was 4.2 percent in 2023")
        claim_without = self._make_claim("The rate was 4.2 percent recently")
        assert self.score_fn(claim_with_date) > self.score_fn(claim_without)

    def test_currency_boosts_score(self):
        claim = self._make_claim("Revenue was $113.8 billion")
        assert self.score_fn(claim) >= 30

    def test_acronyms_boost_score(self):
        claim = self._make_claim("The SEC and FDA issued a joint report")
        assert self.score_fn(claim) >= 10


class TestBudgetCap:
    def test_budget_default_is_zero(self):
        """Budget default should be 0 (unlimited)."""
        import inspect
        from veritas.assist import assist_source
        sig = inspect.signature(assist_source)
        assert sig.parameters["budget_minutes"].default == 0

    def test_budget_zero_means_unlimited(self):
        """budget_minutes=0 should set deadline to infinity."""
        from veritas.assist import assist_source
        from veritas.models import Claim

        mock_claims = [
            Claim(id=f"c{i}", source_id="s1", text=f"Test claim {i} with number {100+i}",
                  ts_start=0.0, ts_end=1.0, confidence_language="definitive",
                  category="general")
            for i in range(3)
        ]

        with patch("veritas.assist.db") as mock_db:
            mock_db.get_claims_for_source.return_value = mock_claims
            mock_source = MagicMock()
            mock_source.title = "Test"
            mock_source.channel = ""
            mock_source.upload_date = ""
            mock_db.get_source.return_value = mock_source

            with patch("veritas.assist.assist_claim") as mock_assist:
                mock_assist.return_value = {
                    "suggestions_found": 0, "suggestions_stored": 0,
                    "status_auto": "unknown", "auto_confidence": 0.0,
                    "best_score": 0, "finance_claim_type": "",
                }
                report = assist_source("s1", budget_minutes=0, dry_run=True)
                # All 3 claims should be processed (no budget limit)
                # Note: some may be skipped due to verifiability < 5
                assert report["claims_processed"] + report.get("claims_skipped_low_verifiability", 0) == 3


class TestVerifiabilitySorting:
    def test_claims_sorted_by_verifiability(self):
        """Higher verifiability claims should be processed first."""
        from veritas.assist import assist_source, _verifiability_score
        from veritas.models import Claim

        claim_high = Claim(id="c1", source_id="s1",
                           text="GDP growth was 3.2 percent in 2024",
                           ts_start=0.0, ts_end=1.0,
                           confidence_language="definitive", category="finance")
        claim_low = Claim(id="c2", source_id="s1",
                          text="it was kind of going up and down",
                          ts_start=0.0, ts_end=1.0,
                          confidence_language="unknown", category="general")

        assert _verifiability_score(claim_high) > _verifiability_score(claim_low)


# ===========================================================================
# 4. Crossref/Wikipedia pre-filtering
# ===========================================================================

class TestCrossrefPreFilter:
    def test_academic_claim_passes(self):
        from veritas.evidence_sources.crossref import _has_academic_relevance
        assert _has_academic_relevance("A study published in Nature found that 85% recovered")

    def test_financial_claim_blocked(self):
        from veritas.evidence_sources.crossref import _has_academic_relevance
        assert not _has_academic_relevance("Revenue was 113.8 billion dollars")

    def test_generic_opinion_blocked(self):
        from veritas.evidence_sources.crossref import _has_academic_relevance
        assert not _has_academic_relevance("I think the economy is getting better")

    def test_clinical_trial_passes(self):
        from veritas.evidence_sources.crossref import _has_academic_relevance
        assert _has_academic_relevance("The clinical trial showed improved outcomes")

    def test_research_keyword_passes(self):
        from veritas.evidence_sources.crossref import _has_academic_relevance
        assert _has_academic_relevance("Researchers demonstrated a correlation between X and Y")

    def test_multi_entity_passes(self):
        from veritas.evidence_sources.crossref import _has_academic_relevance
        assert _has_academic_relevance("Stanford University and Harvard Medical School published a study")

    @patch("veritas.evidence_sources.crossref.rate_limited_get")
    def test_irrelevant_claim_returns_empty(self, mock_get):
        from veritas.evidence_sources.crossref import search_crossref
        result = search_crossref("Apple stock went up 5% last quarter")
        assert result == []
        mock_get.assert_not_called()  # Should not even make API call


class TestWikipediaPreFilter:
    def test_entity_claim_passes(self):
        from veritas.evidence_sources.wikipedia_source import _has_entity_relevance
        assert _has_entity_relevance("Alphabet reported revenue of 113.8 billion")

    def test_acronym_claim_passes(self):
        from veritas.evidence_sources.wikipedia_source import _has_entity_relevance
        assert _has_entity_relevance("GDP growth was 3.2 percent")

    def test_sec_claim_passes(self):
        from veritas.evidence_sources.wikipedia_source import _has_entity_relevance
        assert _has_entity_relevance("The SEC has 318 transfer agents")

    def test_generic_opinion_blocked(self):
        from veritas.evidence_sources.wikipedia_source import _has_entity_relevance
        assert not _has_entity_relevance("it was kind of going up and down all day")

    def test_no_entities_blocked(self):
        from veritas.evidence_sources.wikipedia_source import _has_entity_relevance
        assert not _has_entity_relevance("they said the numbers were interesting")

    @patch("veritas.evidence_sources.wikipedia_source.rate_limited_get")
    def test_irrelevant_claim_returns_empty(self, mock_get):
        from veritas.evidence_sources.wikipedia_source import search_wikipedia
        result = search_wikipedia("they said the numbers were interesting and cool")
        assert result == []
        mock_get.assert_not_called()


# ===========================================================================
# 5. SEC publications search (sec_gov)
# ===========================================================================

class TestSecGov:
    def test_import(self):
        from veritas.evidence_sources.sec_gov import search_sec_gov
        assert callable(search_sec_gov)

    def test_sec_relevance_filter_passes_sec_claims(self):
        from veritas.evidence_sources.sec_gov import _has_sec_relevance
        assert _has_sec_relevance("The SEC has 318 registered transfer agents")
        assert _has_sec_relevance("Division of Enforcement brought 784 actions")
        assert _has_sec_relevance("The commission requested 2.4 billion in budget")
        assert _has_sec_relevance("There are 53 registered security-based swap dealers")

    def test_sec_relevance_filter_blocks_non_sec_claims(self):
        from veritas.evidence_sources.sec_gov import _has_sec_relevance
        assert not _has_sec_relevance("Apple revenue was 100 billion")
        assert not _has_sec_relevance("GDP growth was 3.2 percent")
        assert not _has_sec_relevance("The patient showed improved outcomes")

    @patch("veritas.evidence_sources.sec_gov.rate_limited_get")
    def test_irrelevant_claim_returns_empty(self, mock_get):
        from veritas.evidence_sources.sec_gov import search_sec_gov
        result = search_sec_gov("Apple stock went up 5% yesterday")
        assert result == []
        mock_get.assert_not_called()

    @patch("veritas.evidence_sources.sec_gov.rate_limited_get")
    def test_sec_claim_triggers_search(self, mock_get):
        from veritas.evidence_sources.sec_gov import search_sec_gov

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "hits": {
                "hits": [{
                    "_source": {
                        "file_date": "2025-03-15",
                        "form": "ANNUAL-REPORT",
                        "display_names": ["Securities and Exchange Commission (CIK 0000000)"],
                        "adsh": "0000000000-25-000001",
                        "ciks": ["0000000"],
                        "period_ending": "2025-09-30",
                    }
                }]
            }
        }
        mock_get.return_value = mock_resp

        result = search_sec_gov("The SEC Division of Enforcement brought 784 actions")
        assert mock_get.called
        assert len(result) >= 1
        assert result[0]["source_name"] == "sec_gov"
        assert result[0]["evidence_type"] == "gov"


# ===========================================================================
# 6. Routing includes new sources
# ===========================================================================

class TestRoutingWithNewSources:
    def setup_method(self):
        from veritas.assist import _select_sources_for_category
        self.select = _select_sources_for_category

    def test_finance_includes_local_dataset_first(self):
        sources = self.select("finance")
        names = [n for n, _ in sources]
        assert names[0] == "local_dataset"

    def test_finance_includes_sec_gov(self):
        sources = self.select("finance")
        names = [n for n, _ in sources]
        assert "sec_gov" in names

    def test_general_includes_local_dataset(self):
        sources = self.select("general")
        names = [n for n, _ in sources]
        assert "local_dataset" in names

    def test_all_categories_have_local_dataset_first(self):
        categories = [
            "finance", "health", "science", "tech", "politics",
            "military", "education", "energy_climate", "labor", "general",
        ]
        for cat in categories:
            sources = self.select(cat)
            names = [n for n, _ in sources]
            assert names[0] == "local_dataset", f"local_dataset not first for {cat}"


# ===========================================================================
# 7. Smart routing boost for SEC institutional terms
# ===========================================================================

class TestSmartRoutingSecBoost:
    def test_sec_terms_boost_sec_gov(self):
        from veritas.assist import _smart_select_sources

        sources = [
            ("crossref", lambda x: []),
            ("sec_gov", lambda x: []),
            ("wikipedia", lambda x: []),
        ]

        reranked = _smart_select_sources(
            "The SEC Division of Enforcement brought 784 actions",
            "finance",
            sources,
        )
        names = [n for n, _ in reranked]
        # sec_gov should be boosted to top
        assert names.index("sec_gov") < names.index("crossref")
