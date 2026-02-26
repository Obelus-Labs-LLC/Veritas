"""Tests for Google Fact Check Explorer integration."""

import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers: mock API response data
# ---------------------------------------------------------------------------

def _make_mock_response(claims_data):
    """Build a mock Fact Check Explorer response."""
    wrapper = [["claims_response", claims_data]]
    return ")]}'\n" + json.dumps(wrapper)


def _make_entry(claim_text, claimant, reviews):
    """Build a single claim entry in the Explorer format.

    reviews: list of (pub_name, pub_site, url, rating, title_snippet)
    """
    review_arrays = []
    for pub_name, pub_site, url, rating, title_snippet in reviews:
        # Match the real API structure: [[pub_name, pub_site], url, null, rating, null, [null, id], lang, null, title_snippet, ...]
        rev = [
            [pub_name, pub_site],  # 0: publisher
            url,                    # 1: url
            None,                   # 2: timestamp
            rating,                 # 3: rating
            None,                   # 4
            [None, "123"],          # 5: id
            "en",                   # 6: language
            None,                   # 7
            title_snippet,          # 8: title snippet
        ]
        review_arrays.append(rev)

    claim_array = [
        claim_text,              # 0: claim text
        [claimant, "/m/abc"],    # 1: claimant info
        1700000000,              # 2: timestamp
        review_arrays,           # 3: reviews
    ]
    # Entry format: [claim_array, thumbnail_url, relevance_score]
    return [claim_array, "https://example.com/thumb.jpg", 5.0]


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestParseExplorerResponse:
    def setup_method(self):
        import sys
        sys.path.insert(0, "src")
        from veritas.evidence_sources.google_factcheck import _parse_explorer_response
        self.parse = _parse_explorer_response

    def test_parses_single_claim(self):
        entry = _make_entry(
            "Unemployment is at a record low",
            "Joe Biden",
            [("PolitiFact", "politifact.com", "https://pf.com/check1", "Half True", "Biden's claim about jobs")]
        )
        raw = ")]}'\n" + json.dumps([["claims_response", [entry]]])
        raw = raw[raw.index("\n") + 1:]
        result = self.parse(raw)
        assert len(result) == 1
        assert result[0]["claim_text"] == "Unemployment is at a record low"
        assert result[0]["claimant"] == "Joe Biden"
        assert len(result[0]["reviews"]) == 1
        assert result[0]["reviews"][0]["publisher_name"] == "PolitiFact"
        assert result[0]["reviews"][0]["rating"] == "Half True"

    def test_parses_multiple_claims(self):
        entries = [
            _make_entry("Claim A", "Person A", [("Snopes", "snopes.com", "https://snopes.com/1", "False", "Title A")]),
            _make_entry("Claim B", "Person B", [("AFP", "afp.com", "https://afp.com/1", "True", "Title B")]),
        ]
        raw = json.dumps([["claims_response", entries]])
        result = self.parse(raw)
        assert len(result) == 2
        assert result[0]["claim_text"] == "Claim A"
        assert result[1]["claim_text"] == "Claim B"

    def test_parses_multiple_reviews(self):
        entry = _make_entry(
            "GDP growth is 5%",
            "Politician",
            [
                ("Reuters", "reuters.com", "https://reuters.com/1", "Misleading", "GDP check"),
                ("AFP", "afp.com", "https://afp.com/2", "False", "GDP analysis"),
            ]
        )
        raw = json.dumps([["claims_response", [entry]]])
        result = self.parse(raw)
        assert len(result[0]["reviews"]) == 2

    def test_handles_empty_response(self):
        raw = json.dumps([["claims_response", []]])
        result = self.parse(raw)
        assert result == []

    def test_handles_malformed_json(self):
        result = self.parse("not json")
        assert result == []

    def test_handles_missing_reviews(self):
        # Claim array with only 3 elements (no reviews)
        entry = [[
            "Some claim",
            ["Person", "/m/abc"],
            1700000000,
        ], "thumb.jpg", 3.0]
        raw = json.dumps([["claims_response", [entry]]])
        result = self.parse(raw)
        assert result == []

    def test_skips_entries_without_url(self):
        # Review with empty URL
        entry = _make_entry(
            "A claim",
            "Someone",
            [("Publisher", "pub.com", "", "False", "Title")]
        )
        raw = json.dumps([["claims_response", [entry]]])
        result = self.parse(raw)
        assert len(result) == 1  # parse succeeds, _format_result will filter


# ---------------------------------------------------------------------------
# Format result tests
# ---------------------------------------------------------------------------

class TestFormatResult:
    def setup_method(self):
        import sys
        sys.path.insert(0, "src")
        from veritas.evidence_sources.google_factcheck import _format_result
        self.format = _format_result

    def test_formats_basic_result(self):
        item = {
            "claim_text": "Crime rate is the highest ever",
            "claimant": "Politician X",
            "reviews": [{
                "publisher_name": "PolitiFact",
                "publisher_site": "politifact.com",
                "url": "https://politifact.com/check/123",
                "rating": "Mostly False",
                "title_snippet": "Crime stats debunked",
            }],
        }
        result = self.format(item)
        assert result is not None
        assert result["source_name"] == "google_factcheck"
        assert result["evidence_type"] == "factcheck"
        assert "PolitiFact" in result["title"]
        assert "Mostly False" in result["title"]
        assert "Crime rate is the highest ever" in result["snippet"]
        assert "Politician X" in result["snippet"]
        assert result["url"] == "https://politifact.com/check/123"

    def test_returns_none_for_missing_url(self):
        item = {
            "claim_text": "A claim",
            "reviews": [{"publisher_name": "P", "url": "", "rating": "False", "title_snippet": "", "publisher_site": ""}],
        }
        result = self.format(item)
        assert result is None

    def test_returns_none_for_no_reviews(self):
        item = {"claim_text": "A claim", "reviews": []}
        result = self.format(item)
        assert result is None

    def test_includes_multiple_reviewers_in_snippet(self):
        item = {
            "claim_text": "Claim X",
            "claimant": "",
            "reviews": [
                {"publisher_name": "Snopes", "url": "https://snopes.com/1", "rating": "False", "title_snippet": "", "publisher_site": "snopes.com"},
                {"publisher_name": "AFP", "url": "https://afp.com/1", "rating": "Misleading", "title_snippet": "", "publisher_site": "afp.com"},
            ],
        }
        result = self.format(item)
        assert "Also checked:" in result["snippet"]
        assert "AFP" in result["snippet"]

    def test_snippet_capped_at_2000_chars(self):
        item = {
            "claim_text": "X" * 1500,
            "claimant": "Y" * 500,
            "reviews": [{
                "publisher_name": "P", "url": "https://example.com", "rating": "R",
                "title_snippet": "T" * 500, "publisher_site": "p.com",
            }],
        }
        result = self.format(item)
        assert len(result["snippet"]) <= 2000


# ---------------------------------------------------------------------------
# Integration tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestSearchGoogleFactcheck:
    def setup_method(self):
        import sys
        sys.path.insert(0, "src")
        from veritas.evidence_sources.google_factcheck import search_google_factcheck
        self.search = search_google_factcheck

    @patch("veritas.evidence_sources.google_factcheck.rate_limited_get")
    def test_returns_results_from_api(self, mock_get):
        entry = _make_entry(
            "Inflation is at 2%",
            "Fed Chair",
            [("Reuters", "reuters.com", "https://reuters.com/check", "Mostly True", "Inflation analysis")]
        )
        mock_resp = MagicMock()
        mock_resp.text = _make_mock_response([entry])
        mock_get.return_value = mock_resp

        results = self.search("inflation rate is 2 percent")
        assert len(results) == 1
        assert results[0]["source_name"] == "google_factcheck"
        assert results[0]["evidence_type"] == "factcheck"
        assert "Reuters" in results[0]["title"]

    @patch("veritas.evidence_sources.google_factcheck.rate_limited_get")
    def test_returns_empty_on_api_failure(self, mock_get):
        mock_get.return_value = None
        results = self.search("some claim")
        assert results == []

    @patch("veritas.evidence_sources.google_factcheck.rate_limited_get")
    def test_returns_empty_on_empty_query(self, mock_get):
        results = self.search("")
        assert results == []
        mock_get.assert_not_called()

    @patch("veritas.evidence_sources.google_factcheck.rate_limited_get")
    def test_respects_max_results(self, mock_get):
        entries = [
            _make_entry(f"Claim {i}", "Person", [("Pub", "pub.com", f"https://pub.com/{i}", "False", f"Title {i}")])
            for i in range(10)
        ]
        mock_resp = MagicMock()
        mock_resp.text = _make_mock_response(entries)
        mock_get.return_value = mock_resp

        results = self.search("test claim", max_results=3)
        assert len(results) == 3

    @patch("veritas.evidence_sources.google_factcheck.rate_limited_get")
    def test_handles_malformed_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = ")]}'\nnot valid json"
        mock_get.return_value = mock_resp
        results = self.search("test claim")
        assert results == []


# ---------------------------------------------------------------------------
# Registry and routing tests
# ---------------------------------------------------------------------------

class TestRegistryAndRouting:
    def setup_method(self):
        import sys
        sys.path.insert(0, "src")

    def test_google_factcheck_in_all_sources(self):
        from veritas.evidence_sources import ALL_SOURCES
        names = [name for name, _ in ALL_SOURCES]
        assert "google_factcheck" in names

    def test_source_count_is_8(self):
        from veritas.evidence_sources import ALL_SOURCES
        assert len(ALL_SOURCES) == 8

    def test_politics_routing_includes_factcheck(self):
        from veritas.assist import _select_sources_for_category
        sources = _select_sources_for_category("politics")
        names = [name for name, _ in sources]
        assert "google_factcheck" in names
        # Should be first for politics
        assert names.index("google_factcheck") < names.index("crossref")

    def test_general_routing_includes_factcheck(self):
        from veritas.assist import _select_sources_for_category
        sources = _select_sources_for_category("general")
        names = [name for name, _ in sources]
        assert "google_factcheck" in names

    def test_health_routing_includes_factcheck(self):
        from veritas.assist import _select_sources_for_category
        sources = _select_sources_for_category("health")
        names = [name for name, _ in sources]
        assert "google_factcheck" in names


# ---------------------------------------------------------------------------
# Scoring integration tests
# ---------------------------------------------------------------------------

class TestScoringIntegration:
    def setup_method(self):
        import sys
        sys.path.insert(0, "src")
        from veritas.scoring import score_evidence, compute_auto_status
        self.score_evidence = score_evidence
        self.compute_auto_status = compute_auto_status

    def test_factcheck_gets_primary_boost(self):
        score, signals = self.score_evidence(
            claim_text="Unemployment rate is at a record low",
            claim_category="politics",
            evidence_title="Fact Check: Unemployment rate claim",
            evidence_snippet="Claim: Unemployment rate is at a record low | Rating: Half True | unemployment rate record low statistics",
            evidence_type="factcheck",
            source_name="google_factcheck",
        )
        assert "factcheck_source" in signals
        assert score >= 30  # should get token overlap + factcheck boost

    def test_factcheck_qualifies_as_primary_for_supported(self):
        # factcheck evidence_type should count as primary for auto-status
        status, confidence = self.compute_auto_status(
            best_score=90,
            best_evidence_type="factcheck",
            best_signals="token_overlap:8|keyphrase_hit:2|factcheck_source",
            claim_confidence="definitive",
        )
        assert status == "supported"

    def test_factcheck_partial_at_moderate_score(self):
        status, _ = self.compute_auto_status(
            best_score=75,
            best_evidence_type="factcheck",
            best_signals="token_overlap:5|factcheck_source",
            claim_confidence="unknown",
        )
        assert status == "partial"
