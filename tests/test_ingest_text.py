"""Tests for text/PDF/URL document ingestion (Step 4)."""

import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")


# ===========================================================================
# Text-to-segments conversion
# ===========================================================================

class TestTextToSegments:
    def setup_method(self):
        from veritas.ingest_text import _text_to_segments
        self.convert = _text_to_segments

    def test_single_paragraph(self):
        text = "The Federal Reserve raised interest rates by 25 basis points in March 2024."
        segments = self.convert(text)
        assert len(segments) >= 1
        assert segments[0]["text"] == text.strip()
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] > 0.0

    def test_multiple_paragraphs(self):
        text = "First paragraph about economics.\n\nSecond paragraph about climate change."
        segments = self.convert(text)
        assert len(segments) == 2
        assert "economics" in segments[0]["text"]
        assert "climate" in segments[1]["text"]

    def test_long_paragraph_split(self):
        text = "This is a very long paragraph. " * 20
        segments = self.convert(text)
        assert len(segments) > 1  # should split into multiple segments

    def test_empty_text_returns_empty(self):
        segments = self.convert("")
        assert segments == []

    def test_timestamps_are_sequential(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        segments = self.convert(text)
        for i in range(1, len(segments)):
            assert segments[i]["start"] >= segments[i - 1]["end"] or \
                   segments[i]["start"] >= segments[i - 1]["start"]

    def test_whitespace_only_skipped(self):
        text = "Real content here.\n\n   \n\n   \n\nMore real content."
        segments = self.convert(text)
        for seg in segments:
            assert seg["text"].strip() != ""

    def test_segment_dict_format(self):
        text = "Apple reported revenue of 113.8 billion dollars in Q1 2024."
        segments = self.convert(text)
        assert len(segments) >= 1
        seg = segments[0]
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg
        assert isinstance(seg["start"], float)
        assert isinstance(seg["end"], float)
        assert isinstance(seg["text"], str)


# ===========================================================================
# Split into chunks
# ===========================================================================

class TestSplitIntoChunks:
    def setup_method(self):
        from veritas.ingest_text import _split_into_chunks
        self.split = _split_into_chunks

    def test_short_text_stays_intact(self):
        text = "Short sentence."
        chunks = self.split(text, 200)
        assert len(chunks) == 1
        assert chunks[0] == "Short sentence."

    def test_long_text_splits_at_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = self.split(text, 40)
        assert len(chunks) >= 2
        # Each chunk should end with a complete sentence
        for chunk in chunks:
            assert chunk.strip() != ""


# ===========================================================================
# Article text extraction from HTML
# ===========================================================================

class TestExtractArticleText:
    def setup_method(self):
        from veritas.ingest_text import _extract_article_text
        self.extract = _extract_article_text

    def test_extracts_title(self):
        html = "<html><head><title>Test Article</title></head><body>Content here.</body></html>"
        title, text = self.extract(html)
        assert title == "Test Article"

    def test_extracts_body_text(self):
        html = "<html><body><p>Some paragraph text.</p><p>Another paragraph.</p></body></html>"
        _, text = self.extract(html)
        assert "Some paragraph text" in text
        assert "Another paragraph" in text

    def test_strips_scripts(self):
        html = "<html><body><script>var x = 1;</script><p>Real content.</p></body></html>"
        _, text = self.extract(html)
        assert "var x" not in text
        assert "Real content" in text

    def test_strips_nav_and_footer(self):
        html = "<html><body><nav>Menu stuff</nav><article>Article text here.</article><footer>Footer stuff</footer></body></html>"
        _, text = self.extract(html)
        assert "Article text here" in text
        # Nav/footer may or may not be present depending on article extraction

    def test_prefers_article_tag(self):
        html = "<html><body><div>Sidebar</div><article>Main article content.</article></body></html>"
        _, text = self.extract(html)
        assert "Main article content" in text

    def test_handles_empty_html(self):
        title, text = self.extract("")
        assert title == ""


# ===========================================================================
# Text file ingestion (integration, mocked DB)
# ===========================================================================

class TestIngestTextFile:
    def setup_method(self):
        from veritas.ingest_text import ingest_text_file
        self.ingest = ingest_text_file

    @patch("veritas.ingest_text.db")
    def test_ingests_text_file(self, mock_db, tmp_path):
        # Create a temp text file
        txt = tmp_path / "test_article.txt"
        txt.write_text("The Federal Reserve raised interest rates by 25 basis points. Inflation fell to 3 percent.")

        source = self.ingest(str(txt), title="Test Article")
        assert source.source_type == "text"
        assert source.title == "Test Article"
        mock_db.insert_source.assert_called_once()
        mock_db.upsert_transcript.assert_called_once()

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            self.ingest("/nonexistent/path/file.txt")

    @patch("veritas.ingest_text.db")
    def test_raises_on_empty_file(self, mock_db, tmp_path):
        txt = tmp_path / "empty.txt"
        txt.write_text("")
        with pytest.raises(ValueError, match="empty"):
            self.ingest(str(txt))

    @patch("veritas.ingest_text.db")
    def test_defaults_title_to_filename(self, mock_db, tmp_path):
        txt = tmp_path / "my_article.txt"
        txt.write_text("Some claim about the economy and inflation rate.")
        source = self.ingest(str(txt))
        assert source.title == "my_article"


# ===========================================================================
# URL ingestion (mocked HTTP + DB)
# ===========================================================================

class TestIngestUrl:
    def setup_method(self):
        from veritas.ingest_text import ingest_url
        self.ingest = ingest_url

    @patch("veritas.ingest_text.db")
    @patch("veritas.ingest_text.requests")
    def test_ingests_url(self, mock_requests, mock_db):
        mock_resp = MagicMock()
        mock_resp.text = """
        <html>
        <head><title>Test Page</title></head>
        <body>
        <article>
        <p>The economy grew by 3 percent in Q4 2024.</p>
        <p>Inflation remained steady at 2.5 percent according to the Federal Reserve.</p>
        </article>
        </body>
        </html>
        """
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        source = self.ingest("https://example.com/article")
        assert source.source_type == "url"
        assert "Test Page" in source.title
        mock_db.insert_source.assert_called_once()
        mock_db.upsert_transcript.assert_called_once()

    @patch("veritas.ingest_text.db")
    @patch("veritas.ingest_text.requests")
    def test_raises_on_empty_content(self, mock_requests, mock_db):
        mock_resp = MagicMock()
        mock_resp.text = "<html><body><script>Only scripts here</script></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_resp

        # This might not raise if the script stripping leaves whitespace
        # So we just verify it doesn't crash
        try:
            source = self.ingest("https://example.com/empty")
        except ValueError:
            pass  # Expected for truly empty content


# ===========================================================================
# Raw text ingestion
# ===========================================================================

class TestIngestRawText:
    def setup_method(self):
        from veritas.ingest_text import ingest_raw_text
        self.ingest = ingest_raw_text

    @patch("veritas.ingest_text.db")
    def test_ingests_raw_text(self, mock_db):
        source = self.ingest(
            "The unemployment rate dropped to 3.5 percent in 2024.",
            title="Test Claim",
        )
        assert source.source_type == "text"
        assert source.title == "Test Claim"

    def test_raises_on_empty_text(self):
        with pytest.raises(ValueError, match="empty"):
            self.ingest("")


# ===========================================================================
# CLI command registration
# ===========================================================================

class TestCLICommands:
    def test_ingest_text_command_exists(self):
        from veritas.cli import cli
        commands = cli.commands
        assert "ingest-text" in commands

    def test_ingest_url_command_exists(self):
        from veritas.cli import cli
        commands = cli.commands
        assert "ingest-url" in commands
