"""Data classes used across Veritas."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List
import uuid


def new_id() -> str:
    """Generate a short UUID (first 12 hex chars)."""
    return uuid.uuid4().hex[:12]


@dataclass
class Source:
    id: str = field(default_factory=new_id)
    url: str = ""
    title: str = ""
    channel: str = ""
    upload_date: str = ""
    source_type: str = "audio"  # audio|text|pdf|filing|url
    duration_seconds: float = 0.0
    local_audio_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TranscriptMeta:
    """Metadata row stored in DB.  Actual segments live in transcript.json on disk."""
    source_id: str = ""
    engine: str = "faster-whisper"
    language: str = ""
    segment_count: int = 0
    transcript_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Segment:
    """Single transcript segment — NOT stored in DB; kept in JSON file."""
    start: float = 0.0
    end: float = 0.0
    text: str = ""


@dataclass
class Claim:
    id: str = field(default_factory=new_id)
    source_id: str = ""
    text: str = ""
    ts_start: float = 0.0
    ts_end: float = 0.0
    speaker: Optional[str] = None
    confidence_language: str = "unknown"  # hedged | definitive | unknown
    status: str = "unknown"  # supported | contradicted | partial | unknown
    category: str = "general"  # finance|tech|politics|health|science|military|education|energy_climate|labor|general
    claim_date: str = ""  # extracted year/date from claim text (e.g. "2022", "2008", "1972")
    claim_hash: str = ""  # SHA256(source_id + normalised_text) — same-source dedup
    claim_hash_global: str = ""  # SHA256(normalised_text) — cross-source identity
    signals: str = ""  # pipe-delimited rule signals: "number|named_entity|assertion_verb"
    status_auto: str = "unknown"  # auto verification: supported|partial|unknown
    auto_confidence: float = 0.0  # 0.0-1.0
    status_human: Optional[str] = None  # human override (nullable)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def final_status(self) -> str:
        """Human override wins; otherwise auto; otherwise legacy 'status'."""
        if self.status_human:
            return self.status_human
        if self.status_auto != "unknown":
            return self.status_auto
        return self.status


@dataclass
class Evidence:
    id: str = field(default_factory=new_id)
    claim_id: str = ""
    url: str = ""
    title: str = ""
    evidence_type: str = "other"  # primary|secondary|dataset|filing|gov|paper|other
    strength: str = "medium"  # strong|medium|weak
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ClaimCluster:
    """Cross-source cluster: group of claims about the same fact."""
    id: str = field(default_factory=new_id)
    representative_text: str = ""
    category: str = "general"
    claim_count: int = 0
    source_count: int = 0
    best_status: str = "unknown"
    best_confidence: float = 0.0
    consensus_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class EvidenceSuggestion:
    """Auto-discovered evidence candidate (not human-verified)."""
    id: str = field(default_factory=new_id)
    claim_id: str = ""
    url: str = ""
    title: str = ""
    source_name: str = ""  # crossref|arxiv|pubmed|sec_edgar|yfinance|wikipedia|fred|google_factcheck
    evidence_type: str = "other"  # primary|secondary|dataset|filing|gov|paper|factcheck|other
    score: int = 0  # 0-100
    signals: str = ""  # pipe-delimited explainability signals
    snippet: str = ""  # optional short excerpt
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
