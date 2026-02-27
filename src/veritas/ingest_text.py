"""Ingest text documents (plain text, PDF, web URL) into Veritas.

Converts text content into pseudo-segments compatible with the existing
claim extraction pipeline. No audio, no transcription — text goes straight
to segment format.

Supported intake paths:
  - Plain text file (.txt)
  - PDF file (.pdf) — requires PyMuPDF (fitz) or pdfplumber
  - Raw text string (inline)
  - Web URL (article extraction via requests + basic HTML parsing)
"""

from __future__ import annotations
import json
import re
import textwrap
from pathlib import Path
from typing import List, Optional

import requests

from .models import Source, TranscriptMeta, Segment, new_id
from .paths import source_raw_dir, source_export_dir
from . import db


# ------------------------------------------------------------------
# Text → pseudo-segments
# ------------------------------------------------------------------

_PARA_RE = re.compile(r'\n\s*\n')  # paragraph boundary
_SEGMENT_TARGET_CHARS = 200  # approximate target per segment


def _text_to_segments(text: str) -> List[dict]:
    """Split text into pseudo-segments (fake timestamps, real text).

    Mimics the transcript segment format so the existing claim
    extraction pipeline works unchanged.
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]

    segments = []
    fake_ts = 0.0

    for para in paragraphs:
        # Split long paragraphs into ~200-char chunks at sentence boundaries
        if len(para) <= _SEGMENT_TARGET_CHARS:
            chunks = [para]
        else:
            chunks = _split_into_chunks(para, _SEGMENT_TARGET_CHARS)

        for chunk in chunks:
            if len(chunk.strip()) < 20:
                continue
            duration = max(1.0, len(chunk) / 20.0)  # ~20 chars/sec reading speed
            segments.append({
                "start": round(fake_ts, 3),
                "end": round(fake_ts + duration, 3),
                "text": chunk.strip(),
            })
            fake_ts += duration

    return segments


def _split_into_chunks(text: str, target_chars: int) -> List[str]:
    """Split text at sentence boundaries into chunks of ~target_chars."""
    # Split at sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) > target_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ------------------------------------------------------------------
# Text file ingestion
# ------------------------------------------------------------------

def ingest_text_file(file_path: str, title: str = "", channel: str = "") -> Source:
    """Ingest a plain text file (.txt) as a Veritas source.

    Args:
        file_path: Path to the .txt file.
        title: Optional title (defaults to filename).
        channel: Optional channel/author.

    Returns: Source object.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"File is empty: {file_path}")

    if not title:
        title = path.stem  # filename without extension

    return _create_text_source(
        text=text,
        title=title,
        channel=channel,
        url=str(path.absolute()),
        source_type="text",
    )


# ------------------------------------------------------------------
# PDF ingestion
# ------------------------------------------------------------------

def ingest_pdf(file_path: str, title: str = "", channel: str = "") -> Source:
    """Ingest a PDF file as a Veritas source.

    Tries PyMuPDF (fitz) first, then pdfplumber as fallback.

    Args:
        file_path: Path to the .pdf file.
        title: Optional title (defaults to filename).
        channel: Optional channel/author.

    Returns: Source object.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = _extract_pdf_text(path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from PDF: {file_path}")

    if not title:
        title = path.stem

    return _create_text_source(
        text=text,
        title=title,
        channel=channel,
        url=str(path.absolute()),
        source_type="pdf",
    )


def _extract_pdf_text(path: Path) -> str:
    """Extract text from PDF using available library."""
    # Try PyMuPDF first (faster, better quality)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages)
    except ImportError:
        pass

    raise ImportError(
        "PDF ingestion requires PyMuPDF or pdfplumber. "
        "Install with: pip install PyMuPDF  or  pip install pdfplumber"
    )


# ------------------------------------------------------------------
# URL ingestion
# ------------------------------------------------------------------

def ingest_url(url: str, title: str = "", channel: str = "") -> Source:
    """Ingest a web article URL as a Veritas source.

    Fetches the page and extracts article text using basic HTML parsing.

    Args:
        url: Web URL to fetch.
        title: Optional title (auto-detected from page if empty).
        channel: Optional channel/publisher.

    Returns: Source object.
    """
    resp = requests.get(
        url,
        headers={"User-Agent": "Veritas/1.0 (local research tool)"},
        timeout=30,
    )
    resp.raise_for_status()

    page_title, text = _extract_article_text(resp.text)

    if not text.strip():
        raise ValueError(f"No article text could be extracted from: {url}")

    if not title:
        title = page_title or url

    return _create_text_source(
        text=text,
        title=title,
        channel=channel,
        url=url,
        source_type="url",
    )


def _extract_article_text(html: str) -> tuple[str, str]:
    """Extract title and article text from HTML.

    Basic approach: strip tags, extract <title>, get text from <article>
    or <main> or <body>. No external dependencies.

    Returns (title, text).
    """
    # Extract title
    title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""
    # Clean HTML entities in title
    title = re.sub(r'&amp;', '&', title)
    title = re.sub(r'&lt;', '<', title)
    title = re.sub(r'&gt;', '>', title)
    title = re.sub(r'&#\d+;', '', title)
    title = re.sub(r'&\w+;', '', title)

    # Try to find article/main content
    content_html = html
    for tag in ('article', 'main', '[role="main"]'):
        match = re.search(
            rf'<{tag}[^>]*>(.*?)</{tag.split("[")[0]}>',
            html, re.DOTALL | re.IGNORECASE,
        )
        if match:
            content_html = match.group(1)
            break

    # Strip scripts, styles, nav, header, footer
    for strip_tag in ('script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript'):
        content_html = re.sub(
            rf'<{strip_tag}[^>]*>.*?</{strip_tag}>',
            '', content_html, flags=re.DOTALL | re.IGNORECASE,
        )

    # Strip all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', content_html)

    # Clean up whitespace and HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'\s+', ' ', text)

    # Re-introduce paragraph breaks (approximate based on double spaces or periods)
    text = text.strip()

    return title[:200], text


# ------------------------------------------------------------------
# Inline text ingestion
# ------------------------------------------------------------------

def ingest_raw_text(text: str, title: str = "Inline Text", channel: str = "") -> Source:
    """Ingest raw text string as a Veritas source.

    Args:
        text: The text content.
        title: Title for this source.
        channel: Optional channel/author.

    Returns: Source object.
    """
    if not text.strip():
        raise ValueError("Text content is empty.")

    return _create_text_source(
        text=text,
        title=title,
        channel=channel,
        url="",
        source_type="text",
    )


# ------------------------------------------------------------------
# Shared: create source from text
# ------------------------------------------------------------------

def _create_text_source(
    text: str,
    title: str,
    channel: str,
    url: str,
    source_type: str,
) -> Source:
    """Create a Source + pseudo-transcript from text content.

    Stores the pseudo-segments as a transcript.json so the existing
    claim extraction pipeline can run on it unchanged.
    """
    source_id = new_id()

    # Create source
    source = Source(
        id=source_id,
        url=url,
        title=title,
        channel=channel,
        upload_date="",
        source_type=source_type,
        duration_seconds=0.0,
        local_audio_path="",
    )
    db.insert_source(source)

    # Convert text to pseudo-segments
    segments = _text_to_segments(text)
    if not segments:
        raise ValueError("No segments could be created from the text.")

    # Write transcript.json
    out_dir = source_export_dir(source_id)
    transcript_path = out_dir / "transcript.json"
    with open(transcript_path, "w", encoding="utf-8") as fh:
        json.dump({"segments": segments}, fh, indent=2, ensure_ascii=False)

    # Store transcript metadata
    tmeta = TranscriptMeta(
        source_id=source_id,
        engine="text-ingest",
        language="en",
        segment_count=len(segments),
        transcript_path=str(transcript_path),
    )
    db.upsert_transcript(tmeta)

    return source
