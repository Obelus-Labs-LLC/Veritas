"""Path helpers — create directories lazily and return safe paths."""

from pathlib import Path
from .config import RAW_DIR, TRANSCRIPTS_DIR, EXPORTS_DIR


def ensure_data_dirs() -> None:
    """Create all data sub-directories if they don't exist."""
    for d in (RAW_DIR, TRANSCRIPTS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def source_raw_dir(source_id: str) -> Path:
    """Return (and create) the raw-audio directory for a source."""
    p = RAW_DIR / source_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def source_transcript_dir(source_id: str) -> Path:
    """Return (and create) the transcript directory for a source."""
    p = TRANSCRIPTS_DIR / source_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def source_export_dir(source_id: str) -> Path:
    """Return (and create) the export directory for a source."""
    p = EXPORTS_DIR / source_id
    p.mkdir(parents=True, exist_ok=True)
    return p
