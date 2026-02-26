"""Smoke tests for Veritas — DB init, models, basic CLI import."""

import sys
import os
import sqlite3
import tempfile
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_models_creation():
    from veritas.models import Source, Claim, Evidence, TranscriptMeta, new_id

    s = Source(url="https://example.com", title="Test")
    assert len(s.id) == 12
    assert s.url == "https://example.com"

    c = Claim(source_id=s.id, text="Test claim")
    assert c.status == "unknown"
    assert c.confidence_language == "unknown"

    e = Evidence(claim_id=c.id, url="https://evidence.com")
    assert e.strength == "medium"


def test_db_init_and_roundtrip(tmp_path):
    """Create DB in a temp dir and verify CRUD."""
    import veritas.config as cfg

    # Redirect DB to temp
    original_db = cfg.DB_PATH
    cfg.DB_PATH = tmp_path / "test.sqlite"

    try:
        from veritas import db
        from veritas.models import Source, Claim, Evidence

        db.init_db()
        assert cfg.DB_PATH.exists()

        # Insert + retrieve source
        src = Source(url="https://example.com/video", title="Smoke Test Video")
        db.insert_source(src)
        got = db.get_source(src.id)
        assert got is not None
        assert got.title == "Smoke Test Video"

        # Insert + retrieve claims
        c1 = Claim(source_id=src.id, text="Test claim one")
        c2 = Claim(source_id=src.id, text="Test claim two")
        db.insert_claims([c1, c2])

        claims = db.get_claims_for_source(src.id)
        assert len(claims) == 2

        # Update status
        db.update_claim_status(c1.id, "supported")
        updated = db.get_claim(c1.id)
        assert updated.status == "supported"

        # Search
        results = db.search_claims("claim one")
        assert len(results) == 1

        # Evidence
        ev = Evidence(claim_id=c1.id, url="https://source.gov/report")
        db.insert_evidence(ev)
        evs = db.get_evidence_for_claim(c1.id)
        assert len(evs) == 1
        assert evs[0].url == "https://source.gov/report"

    finally:
        cfg.DB_PATH = original_db


def test_cli_imports():
    """Verify that the CLI module imports without error."""
    from veritas.cli import cli
    assert cli is not None


def test_config_paths():
    from veritas.config import PROJECT_ROOT, DATA_DIR, RAW_DIR
    assert PROJECT_ROOT.exists()
    assert "veritas-app" in str(PROJECT_ROOT) or "veritas" in str(PROJECT_ROOT).lower()


def test_review_list_retrieval(tmp_path):
    """Smoke test: verify the DB calls that power the review command."""
    import veritas.config as cfg
    original_db = cfg.DB_PATH
    cfg.DB_PATH = tmp_path / "review_test.sqlite"

    try:
        from veritas import db
        from veritas.models import Source, Claim

        db.init_db()

        src = Source(url="https://example.com/v", title="Review Test")
        db.insert_source(src)

        claims = [
            Claim(source_id=src.id, text="The Fed announced a rate cut.", ts_start=0.0, ts_end=5.0),
            Claim(source_id=src.id, text="Inflation dropped to 2.3 percent.", ts_start=5.0, ts_end=10.0),
        ]
        db.insert_claims(claims)

        # Retrieve claims as the review command would
        retrieved = db.get_claims_for_source(src.id)
        assert len(retrieved) == 2
        assert retrieved[0].ts_start < retrieved[1].ts_start  # ordered by timestamp

        # Verify one claim
        db.update_claim_status(claims[0].id, "supported")
        updated = db.get_claim(claims[0].id)
        assert updated.status == "supported"

    finally:
        cfg.DB_PATH = original_db
