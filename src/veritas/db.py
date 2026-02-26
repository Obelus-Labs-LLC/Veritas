"""SQLite database initialisation and helpers for Veritas."""

from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any

from . import config
from .models import Source, TranscriptMeta, Claim, Evidence, EvidenceSuggestion

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    channel         TEXT NOT NULL DEFAULT '',
    upload_date     TEXT NOT NULL DEFAULT '',
    duration_seconds REAL NOT NULL DEFAULT 0,
    local_audio_path TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcripts (
    source_id       TEXT PRIMARY KEY REFERENCES sources(id),
    engine          TEXT NOT NULL DEFAULT 'faster-whisper',
    language        TEXT NOT NULL DEFAULT '',
    segment_count   INTEGER NOT NULL DEFAULT 0,
    transcript_path TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claims (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES sources(id),
    text            TEXT NOT NULL,
    ts_start        REAL NOT NULL DEFAULT 0,
    ts_end          REAL NOT NULL DEFAULT 0,
    speaker         TEXT,
    confidence_language TEXT NOT NULL DEFAULT 'unknown',
    status          TEXT NOT NULL DEFAULT 'unknown',
    category        TEXT NOT NULL DEFAULT 'general',
    claim_hash      TEXT NOT NULL DEFAULT '',
    claim_hash_global TEXT NOT NULL DEFAULT '',
    signals         TEXT NOT NULL DEFAULT '',
    status_auto     TEXT NOT NULL DEFAULT 'unknown',
    auto_confidence REAL NOT NULL DEFAULT 0.0,
    status_human    TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_source ON claims(source_id);
CREATE INDEX IF NOT EXISTS idx_claims_text   ON claims(text);
CREATE INDEX IF NOT EXISTS idx_claims_hash   ON claims(claim_hash);
CREATE INDEX IF NOT EXISTS idx_claims_ghash  ON claims(claim_hash_global);

CREATE TABLE IF NOT EXISTS evidence (
    id              TEXT PRIMARY KEY,
    claim_id        TEXT NOT NULL REFERENCES claims(id),
    url             TEXT NOT NULL DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    evidence_type   TEXT NOT NULL DEFAULT 'other',
    strength        TEXT NOT NULL DEFAULT 'medium',
    notes           TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_claim ON evidence(claim_id);

CREATE TABLE IF NOT EXISTS evidence_suggestions (
    id              TEXT PRIMARY KEY,
    claim_id        TEXT NOT NULL REFERENCES claims(id),
    url             TEXT NOT NULL DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    source_name     TEXT NOT NULL DEFAULT '',
    evidence_type   TEXT NOT NULL DEFAULT 'other',
    score           INTEGER NOT NULL DEFAULT 0,
    signals         TEXT NOT NULL DEFAULT '',
    snippet         TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evsug_claim ON evidence_suggestions(claim_id);
CREATE INDEX IF NOT EXISTS idx_evsug_score ON evidence_suggestions(score DESC);
"""

# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    p = config.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_db() -> None:
    """Create tables and indexes if they don't already exist."""
    with sqlite3.connect(str(_db_path())) as conn:
        _migrate_db(conn)
        conn.executescript(_SCHEMA)


def _migrate_db(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema (safe to re-run)."""
    # Check if claims table exists yet (it won't on first run)
    table_check = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='claims'"
    ).fetchone()
    if table_check is None:
        return  # Table doesn't exist yet — _SCHEMA will create it with all columns

    # Fetch existing column names for the claims table
    cursor = conn.execute("PRAGMA table_info(claims)")
    existing = {row[1] for row in cursor.fetchall()}

    if "category" not in existing:
        conn.execute("ALTER TABLE claims ADD COLUMN category TEXT NOT NULL DEFAULT 'general'")
    if "claim_hash" not in existing:
        conn.execute("ALTER TABLE claims ADD COLUMN claim_hash TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_hash ON claims(claim_hash)")
    if "claim_hash_global" not in existing:
        conn.execute("ALTER TABLE claims ADD COLUMN claim_hash_global TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_ghash ON claims(claim_hash_global)")
    if "signals" not in existing:
        conn.execute("ALTER TABLE claims ADD COLUMN signals TEXT NOT NULL DEFAULT ''")
    if "status_auto" not in existing:
        conn.execute("ALTER TABLE claims ADD COLUMN status_auto TEXT NOT NULL DEFAULT 'unknown'")
    if "auto_confidence" not in existing:
        conn.execute("ALTER TABLE claims ADD COLUMN auto_confidence REAL NOT NULL DEFAULT 0.0")
    if "status_human" not in existing:
        conn.execute("ALTER TABLE claims ADD COLUMN status_human TEXT")


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Yield a connection with WAL mode and foreign keys enabled."""
    init_db()
    conn = sqlite3.connect(str(_db_path()))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Source CRUD
# ---------------------------------------------------------------------------

def insert_source(s: Source) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sources (id, url, title, channel, upload_date, duration_seconds, local_audio_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (s.id, s.url, s.title, s.channel, s.upload_date, s.duration_seconds, s.local_audio_path, s.created_at),
        )


def get_source(source_id: str) -> Optional[Source]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
    if row is None:
        return None
    return Source(**dict(row))


def list_sources() -> List[Source]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM sources ORDER BY created_at DESC").fetchall()
    return [Source(**dict(r)) for r in rows]

# ---------------------------------------------------------------------------
# Transcript CRUD
# ---------------------------------------------------------------------------

def upsert_transcript(t: TranscriptMeta) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO transcripts (source_id, engine, language, segment_count, transcript_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (t.source_id, t.engine, t.language, t.segment_count, t.transcript_path, t.created_at),
        )


def get_transcript(source_id: str) -> Optional[TranscriptMeta]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM transcripts WHERE source_id = ?", (source_id,)).fetchone()
    if row is None:
        return None
    return TranscriptMeta(**dict(row))

# ---------------------------------------------------------------------------
# Claim CRUD
# ---------------------------------------------------------------------------

def delete_claims_for_source(source_id: str) -> int:
    """Delete all claims (and their evidence + suggestions) for a source. Returns count deleted."""
    with get_conn() as conn:
        # Delete evidence_suggestions linked to claims for this source
        conn.execute(
            "DELETE FROM evidence_suggestions WHERE claim_id IN (SELECT id FROM claims WHERE source_id = ?)",
            (source_id,),
        )
        # Delete evidence linked to claims for this source
        conn.execute(
            "DELETE FROM evidence WHERE claim_id IN (SELECT id FROM claims WHERE source_id = ?)",
            (source_id,),
        )
        cursor = conn.execute("DELETE FROM claims WHERE source_id = ?", (source_id,))
        return cursor.rowcount


def insert_claims(claims: List[Claim]) -> int:
    """Insert multiple claims.  Returns count inserted."""
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO claims (id, source_id, text, ts_start, ts_end, speaker, "
            "confidence_language, status, category, claim_hash, claim_hash_global, signals, "
            "status_auto, auto_confidence, status_human, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (c.id, c.source_id, c.text, c.ts_start, c.ts_end, c.speaker,
                 c.confidence_language, c.status, c.category, c.claim_hash,
                 c.claim_hash_global, c.signals, c.status_auto, c.auto_confidence,
                 c.status_human, c.created_at, c.updated_at)
                for c in claims
            ],
        )
    return len(claims)


def get_claims_for_source(source_id: str) -> List[Claim]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE source_id = ? ORDER BY ts_start", (source_id,)
        ).fetchall()
    return [Claim(**dict(r)) for r in rows]


def get_claim(claim_id: str) -> Optional[Claim]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
    if row is None:
        return None
    return Claim(**dict(row))


def update_claim_status(claim_id: str, status: str) -> None:
    from datetime import datetime, timezone
    with get_conn() as conn:
        conn.execute(
            "UPDATE claims SET status = ?, updated_at = ? WHERE id = ?",
            (status, datetime.now(timezone.utc).isoformat(), claim_id),
        )


def search_claims(query: str, limit: int = 20) -> List[Claim]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE text LIKE ? ORDER BY ts_start LIMIT ?",
            (f"%{query}%", limit),
        ).fetchall()
    return [Claim(**dict(r)) for r in rows]

# ---------------------------------------------------------------------------
# Evidence CRUD
# ---------------------------------------------------------------------------

def insert_evidence(e: Evidence) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO evidence (id, claim_id, url, title, evidence_type, strength, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (e.id, e.claim_id, e.url, e.title, e.evidence_type, e.strength, e.notes, e.created_at),
        )


def get_evidence_for_claim(claim_id: str) -> List[Evidence]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM evidence WHERE claim_id = ? ORDER BY created_at", (claim_id,)
        ).fetchall()
    return [Evidence(**dict(r)) for r in rows]

# ---------------------------------------------------------------------------
# Evidence Suggestions CRUD
# ---------------------------------------------------------------------------

def insert_evidence_suggestions(suggestions: List[EvidenceSuggestion]) -> int:
    """Insert multiple evidence suggestions. Returns count inserted."""
    if not suggestions:
        return 0
    with get_conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO evidence_suggestions "
            "(id, claim_id, url, title, source_name, evidence_type, score, signals, snippet, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (s.id, s.claim_id, s.url, s.title, s.source_name,
                 s.evidence_type, s.score, s.signals, s.snippet, s.created_at)
                for s in suggestions
            ],
        )
    return len(suggestions)


def get_suggestions_for_claim(claim_id: str, limit: int = 10) -> List[EvidenceSuggestion]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM evidence_suggestions WHERE claim_id = ? ORDER BY score DESC LIMIT ?",
            (claim_id, limit),
        ).fetchall()
    return [EvidenceSuggestion(**dict(r)) for r in rows]


def delete_suggestions_for_source(source_id: str) -> int:
    """Delete all evidence suggestions for claims belonging to a source."""
    with get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM evidence_suggestions WHERE claim_id IN "
            "(SELECT id FROM claims WHERE source_id = ?)",
            (source_id,),
        )
        return cursor.rowcount


def update_claim_auto_status(claim_id: str, status_auto: str, auto_confidence: float) -> None:
    """Set the auto verification status and confidence."""
    from datetime import datetime, timezone
    with get_conn() as conn:
        conn.execute(
            "UPDATE claims SET status_auto = ?, auto_confidence = ?, updated_at = ? WHERE id = ?",
            (status_auto, auto_confidence, datetime.now(timezone.utc).isoformat(), claim_id),
        )


def update_claim_human_status(claim_id: str, status_human: str) -> None:
    """Set human override status."""
    from datetime import datetime, timezone
    with get_conn() as conn:
        conn.execute(
            "UPDATE claims SET status_human = ?, status = ?, updated_at = ? WHERE id = ?",
            (status_human, status_human, datetime.now(timezone.utc).isoformat(), claim_id),
        )


def get_review_queue(limit: int = 20) -> List[Claim]:
    """Get claims sorted for review: unknown first, low confidence, high propagation."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM claims ORDER BY "
            "CASE WHEN status_auto = 'unknown' AND status_human IS NULL THEN 0 ELSE 1 END, "
            "auto_confidence ASC, "
            "ts_start ASC "
            "LIMIT ?",
            (limit,),
        ).fetchall()
    return [Claim(**dict(r)) for r in rows]


# ---------------------------------------------------------------------------
# Cross-source claim intelligence queries
# ---------------------------------------------------------------------------

def get_claim_spread(claim_hash_global: str) -> List[Dict[str, Any]]:
    """Find all occurrences of a claim across sources by its global hash.

    Returns list of dicts: {claim_id, source_id, source_title, text, ts_start,
    status_auto, auto_confidence, created_at}.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT c.id AS claim_id, c.source_id, s.title AS source_title, "
            "c.text, c.ts_start, c.status_auto, c.auto_confidence, c.category, "
            "s.created_at AS source_created "
            "FROM claims c JOIN sources s ON c.source_id = s.id "
            "WHERE c.claim_hash_global = ? "
            "ORDER BY s.created_at ASC, c.ts_start ASC",
            (claim_hash_global,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_claim_timeline(claim_hash_global: str) -> List[Dict[str, Any]]:
    """Chronological propagation of a claim across sources.

    Returns list of dicts with source date, source title, claim text,
    and verification status — ordered by source created_at.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT s.created_at AS source_date, s.title AS source_title, "
            "s.id AS source_id, c.id AS claim_id, c.text, c.ts_start, "
            "c.status_auto, c.auto_confidence, c.category "
            "FROM claims c JOIN sources s ON c.source_id = s.id "
            "WHERE c.claim_hash_global = ? "
            "ORDER BY s.created_at ASC",
            (claim_hash_global,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_top_claims(limit: int = 20) -> List[Dict[str, Any]]:
    """Most-repeated claims across all sources, ranked by frequency.

    Returns list of dicts: {claim_hash_global, frequency, text (from first occurrence),
    category, sources (comma-separated source_ids), best_status, best_confidence}.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT c.claim_hash_global, "
            "COUNT(*) AS frequency, "
            "COUNT(DISTINCT c.source_id) AS source_count, "
            "MIN(c.text) AS text, "
            "MIN(c.category) AS category, "
            "GROUP_CONCAT(DISTINCT c.source_id) AS source_ids, "
            "MAX(CASE WHEN c.status_auto = 'supported' THEN 2 "
            "         WHEN c.status_auto = 'partial' THEN 1 ELSE 0 END) AS best_status_rank, "
            "MAX(c.auto_confidence) AS best_confidence "
            "FROM claims c "
            "WHERE c.claim_hash_global != '' "
            "GROUP BY c.claim_hash_global "
            "HAVING COUNT(DISTINCT c.source_id) > 1 "
            "ORDER BY source_count DESC, frequency DESC "
            "LIMIT ?",
            (limit,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        # Decode best_status_rank back to label
        rank = d.pop("best_status_rank", 0)
        d["best_status"] = {2: "supported", 1: "partial"}.get(rank, "unknown")
        result.append(d)
    return result


def get_source_verification_stats() -> List[Dict[str, Any]]:
    """Per-source verification metrics for the enhanced sources command.

    Returns list of dicts: {source_id, title, channel, created_at, total_claims,
    supported, partial, unknown, verified_rate}.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT s.id AS source_id, s.title, s.channel, s.duration_seconds, "
            "s.created_at, "
            "COUNT(c.id) AS total_claims, "
            "SUM(CASE WHEN COALESCE(c.status_human, "
            "    CASE WHEN c.status_auto != 'unknown' THEN c.status_auto ELSE c.status END) "
            "    = 'supported' THEN 1 ELSE 0 END) AS supported, "
            "SUM(CASE WHEN COALESCE(c.status_human, "
            "    CASE WHEN c.status_auto != 'unknown' THEN c.status_auto ELSE c.status END) "
            "    = 'partial' THEN 1 ELSE 0 END) AS partial, "
            "SUM(CASE WHEN COALESCE(c.status_human, "
            "    CASE WHEN c.status_auto != 'unknown' THEN c.status_auto ELSE c.status END) "
            "    = 'unknown' THEN 1 ELSE 0 END) AS unknown_count "
            "FROM sources s LEFT JOIN claims c ON c.source_id = s.id "
            "GROUP BY s.id "
            "ORDER BY s.created_at DESC",
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        total = d["total_claims"]
        verified = d["supported"] + d["partial"]
        d["verified_rate"] = (verified / total * 100) if total > 0 else 0.0
        result.append(d)
    return result
