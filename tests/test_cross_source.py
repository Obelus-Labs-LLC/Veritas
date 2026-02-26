"""Tests for cross-source claim intelligence: spread, timeline, top-claims, enhanced sources."""

import sys
import hashlib
import string
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _normalise(text: str) -> str:
    """Mirror the normalisation from claim_extract.py."""
    t = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(t.split())


def _global_hash(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode()).hexdigest()


def _setup_cross_source_db(tmp_path):
    """Create a temp DB with 3 sources, some claims sharing global hashes."""
    import veritas.config as cfg
    cfg.DB_PATH = tmp_path / "cross_source.sqlite"

    from veritas import db
    from veritas.models import Source, Claim

    db.init_db()

    # Source A — earliest
    src_a = Source(id="aaa000000001", url="https://example.com/a", title="Source Alpha",
                   channel="ChannelA", created_at="2025-01-01T00:00:00+00:00")
    # Source B — middle
    src_b = Source(id="bbb000000002", url="https://example.com/b", title="Source Beta",
                   channel="ChannelB", created_at="2025-02-01T00:00:00+00:00")
    # Source C — latest
    src_c = Source(id="ccc000000003", url="https://example.com/c", title="Source Gamma",
                   channel="ChannelC", created_at="2025-03-01T00:00:00+00:00")

    db.insert_source(src_a)
    db.insert_source(src_b)
    db.insert_source(src_c)

    # Shared claim text (appears in A and B) — same global hash
    shared_text = "Revenue grew 12 percent year over year."
    shared_ghash = _global_hash(shared_text)

    # Another shared claim (appears in all 3 sources)
    shared_text_2 = "The company reported 100 billion dollars in revenue."
    shared_ghash_2 = _global_hash(shared_text_2)

    claims = [
        # Source A claims
        Claim(id="claim_a1", source_id=src_a.id, text=shared_text, ts_start=10.0, ts_end=15.0,
              category="finance", claim_hash_global=shared_ghash,
              status_auto="supported", auto_confidence=0.85),
        Claim(id="claim_a2", source_id=src_a.id, text="Unique claim in source A.", ts_start=20.0, ts_end=25.0,
              category="general", claim_hash_global=_global_hash("Unique claim in source A.")),
        Claim(id="claim_a3", source_id=src_a.id, text=shared_text_2, ts_start=30.0, ts_end=35.0,
              category="finance", claim_hash_global=shared_ghash_2,
              status_auto="partial", auto_confidence=0.6),

        # Source B claims
        Claim(id="claim_b1", source_id=src_b.id, text=shared_text, ts_start=5.0, ts_end=10.0,
              category="finance", claim_hash_global=shared_ghash,
              status_auto="partial", auto_confidence=0.7),
        Claim(id="claim_b2", source_id=src_b.id, text=shared_text_2, ts_start=15.0, ts_end=20.0,
              category="finance", claim_hash_global=shared_ghash_2,
              status_auto="supported", auto_confidence=0.9),

        # Source C claims
        Claim(id="claim_c1", source_id=src_c.id, text=shared_text_2, ts_start=0.0, ts_end=5.0,
              category="finance", claim_hash_global=shared_ghash_2,
              status_auto="unknown", auto_confidence=0.0),
        Claim(id="claim_c2", source_id=src_c.id, text="Only in source C.", ts_start=10.0, ts_end=15.0,
              category="science", claim_hash_global=_global_hash("Only in source C.")),
    ]
    db.insert_claims(claims)

    return {
        "sources": [src_a, src_b, src_c],
        "claims": claims,
        "shared_ghash": shared_ghash,
        "shared_ghash_2": shared_ghash_2,
    }


# ──────────────────────────────────────────────────────────────────
# Tests for db.get_claim_spread
# ──────────────────────────────────────────────────────────────────

def test_spread_returns_all_occurrences(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        data = _setup_cross_source_db(tmp_path)
        from veritas import db
        results = db.get_claim_spread(data["shared_ghash"])
        assert len(results) == 2
        source_ids = {r["source_id"] for r in results}
        assert source_ids == {"aaa000000001", "bbb000000002"}
    finally:
        cfg.DB_PATH = original


def test_spread_three_sources(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        data = _setup_cross_source_db(tmp_path)
        from veritas import db
        results = db.get_claim_spread(data["shared_ghash_2"])
        assert len(results) == 3
        source_ids = {r["source_id"] for r in results}
        assert source_ids == {"aaa000000001", "bbb000000002", "ccc000000003"}
    finally:
        cfg.DB_PATH = original


def test_spread_nonexistent_hash(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        _setup_cross_source_db(tmp_path)
        from veritas import db
        results = db.get_claim_spread("nonexistent_hash_value")
        assert results == []
    finally:
        cfg.DB_PATH = original


def test_spread_ordered_by_source_date(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        data = _setup_cross_source_db(tmp_path)
        from veritas import db
        results = db.get_claim_spread(data["shared_ghash_2"])
        dates = [r["source_created"] for r in results]
        assert dates == sorted(dates)
    finally:
        cfg.DB_PATH = original


# ──────────────────────────────────────────────────────────────────
# Tests for db.get_claim_timeline
# ──────────────────────────────────────────────────────────────────

def test_timeline_chronological(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        data = _setup_cross_source_db(tmp_path)
        from veritas import db
        entries = db.get_claim_timeline(data["shared_ghash"])
        assert len(entries) == 2
        # First entry should be Source Alpha (earliest)
        assert entries[0]["source_title"] == "Source Alpha"
        assert entries[1]["source_title"] == "Source Beta"
    finally:
        cfg.DB_PATH = original


def test_timeline_includes_status(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        data = _setup_cross_source_db(tmp_path)
        from veritas import db
        entries = db.get_claim_timeline(data["shared_ghash"])
        statuses = [e["status_auto"] for e in entries]
        assert "supported" in statuses
        assert "partial" in statuses
    finally:
        cfg.DB_PATH = original


# ──────────────────────────────────────────────────────────────────
# Tests for db.get_top_claims
# ──────────────────────────────────────────────────────────────────

def test_top_claims_returns_cross_source_only(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        _setup_cross_source_db(tmp_path)
        from veritas import db
        results = db.get_top_claims(limit=10)
        # Only claims appearing in 2+ sources should be returned
        for r in results:
            assert r["source_count"] >= 2
    finally:
        cfg.DB_PATH = original


def test_top_claims_sorted_by_source_count(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        _setup_cross_source_db(tmp_path)
        from veritas import db
        results = db.get_top_claims(limit=10)
        assert len(results) == 2  # Two shared claims
        # shared_text_2 appears in 3 sources, shared_text in 2
        assert results[0]["source_count"] == 3
        assert results[1]["source_count"] == 2
    finally:
        cfg.DB_PATH = original


def test_top_claims_best_status(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        _setup_cross_source_db(tmp_path)
        from veritas import db
        results = db.get_top_claims(limit=10)
        # Both shared claims have at least one supported occurrence
        best_statuses = {r["best_status"] for r in results}
        assert "supported" in best_statuses
    finally:
        cfg.DB_PATH = original


def test_top_claims_limit(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        _setup_cross_source_db(tmp_path)
        from veritas import db
        results = db.get_top_claims(limit=1)
        assert len(results) == 1
    finally:
        cfg.DB_PATH = original


# ──────────────────────────────────────────────────────────────────
# Tests for db.get_source_verification_stats
# ──────────────────────────────────────────────────────────────────

def test_source_stats_all_sources(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        _setup_cross_source_db(tmp_path)
        from veritas import db
        stats = db.get_source_verification_stats()
        assert len(stats) == 3
    finally:
        cfg.DB_PATH = original


def test_source_stats_claim_counts(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        _setup_cross_source_db(tmp_path)
        from veritas import db
        stats = db.get_source_verification_stats()
        by_id = {s["source_id"]: s for s in stats}
        assert by_id["aaa000000001"]["total_claims"] == 3
        assert by_id["bbb000000002"]["total_claims"] == 2
        assert by_id["ccc000000003"]["total_claims"] == 2
    finally:
        cfg.DB_PATH = original


def test_source_stats_verified_rate(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    try:
        _setup_cross_source_db(tmp_path)
        from veritas import db
        stats = db.get_source_verification_stats()
        by_id = {s["source_id"]: s for s in stats}

        # Source A: claim_a1=supported, claim_a2=unknown, claim_a3=partial → 2/3 verified
        a_rate = by_id["aaa000000001"]["verified_rate"]
        assert abs(a_rate - 66.7) < 1.0  # ~66.7%

        # Source B: claim_b1=partial, claim_b2=supported → 2/2 verified
        b_rate = by_id["bbb000000002"]["verified_rate"]
        assert b_rate == 100.0

        # Source C: claim_c1=unknown, claim_c2=unknown → 0/2 verified
        c_rate = by_id["ccc000000003"]["verified_rate"]
        assert c_rate == 0.0
    finally:
        cfg.DB_PATH = original


def test_source_stats_empty_db(tmp_path):
    import veritas.config as cfg
    original = cfg.DB_PATH
    cfg.DB_PATH = tmp_path / "empty.sqlite"
    try:
        from veritas import db
        db.init_db()
        stats = db.get_source_verification_stats()
        assert stats == []
    finally:
        cfg.DB_PATH = original


# ──────────────────────────────────────────────────────────────────
# CLI integration smoke tests
# ──────────────────────────────────────────────────────────────────

def test_cli_spread_command_exists():
    from veritas.cli import cli
    assert "spread" in [c.name for c in cli.commands.values()]


def test_cli_timeline_command_exists():
    from veritas.cli import cli
    assert "timeline" in [c.name for c in cli.commands.values()]


def test_cli_top_claims_command_exists():
    from veritas.cli import cli
    assert "top-claims" in [c.name for c in cli.commands.values()]


def test_cli_sources_has_by_option():
    from veritas.cli import cli
    sources_cmd = cli.commands["sources"]
    param_names = [p.name for p in sources_cmd.params]
    assert "sort_by" in param_names
