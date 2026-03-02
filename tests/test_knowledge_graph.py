"""Tests for the knowledge graph layer — fingerprinting, clustering, consensus, DB, CLI."""

import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Step 1: Fingerprinting tests
# ---------------------------------------------------------------------------

class TestClaimFingerprint:
    """Tests for claim_fingerprint()."""

    def test_deterministic(self):
        """Same input always produces same fingerprint."""
        from veritas.knowledge_graph import claim_fingerprint
        fp1 = claim_fingerprint("Alphabet reported $350 billion revenue in 2024", "finance")
        fp2 = claim_fingerprint("Alphabet reported $350 billion revenue in 2024", "finance")
        assert fp1 == fp2
        assert len(fp1) > 0

    def test_case_insensitive(self):
        """Fingerprint ignores case."""
        from veritas.knowledge_graph import claim_fingerprint
        fp1 = claim_fingerprint("GDP grew by 3 percent", "finance")
        fp2 = claim_fingerprint("gdp grew by 3 percent", "finance")
        assert fp1 == fp2

    def test_captures_numbers(self):
        """Numbers appear in fingerprint."""
        from veritas.knowledge_graph import claim_fingerprint
        fp = claim_fingerprint("Revenue was $113.8 billion last year", "finance")
        assert "113.8" in fp.split("|")

    def test_unit_conversion(self):
        """Billion/million units get expanded."""
        from veritas.knowledge_graph import claim_fingerprint
        fp = claim_fingerprint("Revenue was $350 billion", "finance")
        parts = set(fp.split("|"))
        assert "350" in parts
        assert "350000" in parts  # billions → millions expansion

    def test_stopword_removal(self):
        """Common stopwords are excluded from fingerprint."""
        from veritas.knowledge_graph import claim_fingerprint
        fp = claim_fingerprint("The company is in the market", "general")
        parts = set(fp.split("|"))
        assert "the" not in parts
        assert "is" not in parts
        assert "in" not in parts
        assert "company" in parts
        assert "market" in parts

    def test_empty_input(self):
        """Empty or all-stopword input returns empty string."""
        from veritas.knowledge_graph import claim_fingerprint
        assert claim_fingerprint("", "general") == ""
        assert claim_fingerprint("the a an is", "general") == ""

    def test_category_terms_included(self):
        """Category-relevant terms appear in fingerprint."""
        from veritas.knowledge_graph import claim_fingerprint
        fp = claim_fingerprint("Revenue growth was strong at 15 percent", "finance")
        parts = set(fp.split("|"))
        # "revenue" is a finance category term
        assert "revenue" in parts

    def test_similar_facts_high_similarity(self):
        """Same fact with different wording produces high similarity."""
        from veritas.knowledge_graph import claim_fingerprint, fingerprint_similarity
        fp1 = claim_fingerprint("Alphabet reported $350 billion revenue in 2024", "finance")
        fp2 = claim_fingerprint("Google parent Alphabet had revenue of $350 billion for 2024", "finance")
        sim = fingerprint_similarity(fp1, fp2)
        assert sim >= 0.30  # should share key terms: 350, 350000, revenue, alphabet, 2024

    def test_different_facts_low_similarity(self):
        """Different facts produce low similarity."""
        from veritas.knowledge_graph import claim_fingerprint, fingerprint_similarity
        fp1 = claim_fingerprint("Alphabet reported $350 billion revenue in 2024", "finance")
        fp2 = claim_fingerprint("The climate changed significantly due to carbon emissions", "science")
        sim = fingerprint_similarity(fp1, fp2)
        assert sim < 0.15


class TestFingerprintSimilarity:
    """Tests for fingerprint_similarity()."""

    def test_identical(self):
        from veritas.knowledge_graph import fingerprint_similarity
        assert fingerprint_similarity("a|b|c", "a|b|c") == 1.0

    def test_no_overlap(self):
        from veritas.knowledge_graph import fingerprint_similarity
        assert fingerprint_similarity("a|b|c", "d|e|f") == 0.0

    def test_partial_overlap(self):
        from veritas.knowledge_graph import fingerprint_similarity
        sim = fingerprint_similarity("a|b|c|d", "a|b|e|f")
        # intersection: {a,b} = 2, union: {a,b,c,d,e,f} = 6 → 2/6 = 0.333
        assert abs(sim - 2 / 6) < 0.01

    def test_empty_returns_zero(self):
        from veritas.knowledge_graph import fingerprint_similarity
        assert fingerprint_similarity("", "a|b") == 0.0
        assert fingerprint_similarity("a|b", "") == 0.0
        assert fingerprint_similarity("", "") == 0.0


# ---------------------------------------------------------------------------
# Step 2: Blocking tests
# ---------------------------------------------------------------------------

class TestBuildBlocks:
    """Tests for build_blocks()."""

    def test_same_number_same_block(self):
        """Claims with the same number go to the same block."""
        from veritas.knowledge_graph import build_blocks, ClaimRecord, _extract_claim_numbers
        from veritas.scoring import _extract_claim_numbers

        c1 = ClaimRecord(claim_id="a", source_id="s1", text="Revenue was $350 billion", category="finance",
                         numbers=frozenset(_extract_claim_numbers("Revenue was $350 billion")))
        c2 = ClaimRecord(claim_id="b", source_id="s2", text="$350 billion in sales", category="finance",
                         numbers=frozenset(_extract_claim_numbers("$350 billion in sales")))

        blocks = build_blocks([c1, c2])
        # Both should appear in at least one common block
        found_together = False
        for block_claims in blocks.values():
            ids = {c.claim_id for c in block_claims}
            if "a" in ids and "b" in ids:
                found_together = True
                break
        assert found_together

    def test_different_category_different_block(self):
        """Claims with same number but different category go to different blocks."""
        from veritas.knowledge_graph import build_blocks, ClaimRecord

        c1 = ClaimRecord(claim_id="a", source_id="s1", text="100 troops deployed",
                         category="military", numbers=frozenset(["100"]))
        c2 = ClaimRecord(claim_id="b", source_id="s2", text="100 students enrolled",
                         category="education", numbers=frozenset(["100"]))

        blocks = build_blocks([c1, c2])
        # Should be in DIFFERENT blocks (military|100 vs education|100)
        for block_claims in blocks.values():
            ids = {c.claim_id for c in block_claims}
            assert not ("a" in ids and "b" in ids), "Different categories should not share blocks"

    def test_no_numbers_grouped(self):
        """Claims without numbers go to a no_numbers block."""
        from veritas.knowledge_graph import build_blocks, ClaimRecord

        c1 = ClaimRecord(claim_id="a", source_id="s1", text="Climate change is real", category="science")
        c2 = ClaimRecord(claim_id="b", source_id="s2", text="Global warming confirmed", category="science")

        blocks = build_blocks([c1, c2])
        assert "science|no_numbers" in blocks
        assert len(blocks["science|no_numbers"]) == 2


# ---------------------------------------------------------------------------
# Step 3: Union-Find tests
# ---------------------------------------------------------------------------

class TestUnionFind:
    """Tests for UnionFind data structure."""

    def test_find_new_element(self):
        from veritas.knowledge_graph import UnionFind
        uf = UnionFind()
        assert uf.find("a") == "a"

    def test_union_and_find(self):
        from veritas.knowledge_graph import UnionFind
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_transitivity(self):
        """A-B + B-C → A, B, C in same cluster."""
        from veritas.knowledge_graph import UnionFind
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_clusters_output(self):
        from veritas.knowledge_graph import UnionFind
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        clusters = uf.clusters()
        # Should have 2 clusters
        assert len(clusters) == 2
        members = [sorted(v) for v in clusters.values()]
        assert sorted(members) == [["a", "b"], ["c", "d"]]


class TestBuildClusters:
    """Tests for build_clusters() end-to-end."""

    def test_cross_source_clustering(self):
        """Claims about the same fact from different sources get clustered."""
        from veritas.knowledge_graph import build_clusters, ClaimRecord

        claims = [
            ClaimRecord(claim_id="a", source_id="s1",
                        text="Alphabet reported $350 billion revenue in 2024", category="finance"),
            ClaimRecord(claim_id="b", source_id="s2",
                        text="Google parent Alphabet had $350 billion revenue for 2024", category="finance"),
        ]
        clusters = build_clusters(claims, threshold=0.30)
        # Should form 1 cluster with both claims
        assert len(clusters) >= 1
        all_ids = set()
        for members in clusters.values():
            all_ids.update(m.claim_id for m in members)
        assert "a" in all_ids and "b" in all_ids

    def test_same_source_not_clustered(self):
        """Claims from the same source are never clustered together."""
        from veritas.knowledge_graph import build_clusters, ClaimRecord

        claims = [
            ClaimRecord(claim_id="a", source_id="s1",
                        text="Revenue was $350 billion", category="finance"),
            ClaimRecord(claim_id="b", source_id="s1",
                        text="Revenue reached $350 billion", category="finance"),
        ]
        clusters = build_clusters(claims, threshold=0.30)
        # Should have NO clusters (same source)
        assert len(clusters) == 0

    def test_singletons_excluded(self):
        """Claims that don't match anything are excluded from clusters."""
        from veritas.knowledge_graph import build_clusters, ClaimRecord

        claims = [
            ClaimRecord(claim_id="a", source_id="s1",
                        text="Alphabet reported $350 billion revenue in 2024", category="finance"),
            ClaimRecord(claim_id="b", source_id="s2",
                        text="Climate change caused extreme weather events", category="science"),
        ]
        clusters = build_clusters(claims, threshold=0.40)
        # These are completely different facts — no cluster
        assert len(clusters) == 0

    def test_threshold_respected(self):
        """Higher threshold means fewer clusters."""
        from veritas.knowledge_graph import build_clusters, ClaimRecord

        claims = [
            ClaimRecord(claim_id="a", source_id="s1",
                        text="Revenue was $350 billion in 2024", category="finance"),
            ClaimRecord(claim_id="b", source_id="s2",
                        text="$350 billion revenue reported for 2024", category="finance"),
        ]
        low = build_clusters(claims, threshold=0.10)
        high = build_clusters(claims, threshold=0.90)
        assert len(low) >= len(high)


# ---------------------------------------------------------------------------
# Step 4: DB tests
# ---------------------------------------------------------------------------

class TestClusterDB:
    """Tests for cluster DB CRUD functions."""

    def _setup_db(self, tmp_path):
        """Redirect DB to temp and initialize."""
        import veritas.config as cfg
        cfg.DB_PATH = tmp_path / "test_kg.sqlite"
        from veritas import db
        db.init_db()
        return db

    def test_table_creation(self, tmp_path):
        """Cluster tables are created on init."""
        import veritas.config as cfg
        cfg.DB_PATH = tmp_path / "test_tables.sqlite"
        from veritas import db
        db.init_db()

        import sqlite3
        conn = sqlite3.connect(str(cfg.DB_PATH))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "claim_clusters" in tables
        assert "cluster_members" in tables

    def test_upsert_and_get_cluster(self, tmp_path):
        """Insert and retrieve a cluster."""
        db = self._setup_db(tmp_path)
        from veritas.models import ClaimCluster

        cluster = ClaimCluster(
            id="test_c1",
            representative_text="Test claim about revenue",
            category="finance",
            claim_count=3,
            source_count=2,
            best_status="supported",
            best_confidence=0.85,
            consensus_score=0.92,
        )
        db.upsert_clusters([cluster])

        got = db.get_cluster("test_c1")
        assert got is not None
        assert got["representative_text"] == "Test claim about revenue"
        assert got["claim_count"] == 3
        assert got["consensus_score"] == 0.92

    def test_insert_and_get_members(self, tmp_path):
        """Insert cluster members and retrieve with claim details."""
        db = self._setup_db(tmp_path)
        from veritas.models import ClaimCluster, Source, Claim

        # Create source and claims first (FK constraints)
        src = Source(id="src1", url="https://example.com", title="Test Source")
        db.insert_source(src)
        claims = [
            Claim(id="cl1", source_id="src1", text="Revenue was $350 billion"),
            Claim(id="cl2", source_id="src1", text="$350B in revenue reported"),
        ]
        db.insert_claims(claims)

        # Create cluster
        cluster = ClaimCluster(id="c1", representative_text="Revenue $350B",
                               category="finance", claim_count=2, source_count=1)
        db.upsert_clusters([cluster])

        # Insert members
        members = [
            {"cluster_id": "c1", "claim_id": "cl1", "fingerprint": "350|revenue", "similarity_to_rep": 1.0},
            {"cluster_id": "c1", "claim_id": "cl2", "fingerprint": "350|revenue|reported", "similarity_to_rep": 0.8},
        ]
        db.insert_cluster_members(members)

        got = db.get_cluster_members("c1")
        assert len(got) == 2
        # Should be sorted by similarity DESC
        assert got[0]["similarity_to_rep"] >= got[1]["similarity_to_rep"]
        assert got[0]["text"] == "Revenue was $350 billion"

    def test_clear_clusters(self, tmp_path):
        """clear_clusters() removes all clusters and members."""
        db = self._setup_db(tmp_path)
        from veritas.models import ClaimCluster

        cluster = ClaimCluster(id="c1", representative_text="Test")
        db.upsert_clusters([cluster])
        assert db.get_cluster("c1") is not None

        db.clear_clusters()
        assert db.get_cluster("c1") is None

    def test_get_top_clusters(self, tmp_path):
        """get_top_clusters returns sorted results."""
        db = self._setup_db(tmp_path)
        from veritas.models import ClaimCluster

        clusters = [
            ClaimCluster(id="c1", representative_text="Low consensus",
                         consensus_score=0.3, source_count=1),
            ClaimCluster(id="c2", representative_text="High consensus",
                         consensus_score=0.9, source_count=3),
            ClaimCluster(id="c3", representative_text="Medium consensus",
                         consensus_score=0.6, source_count=2),
        ]
        db.upsert_clusters(clusters)

        top = db.get_top_clusters(limit=10, sort_by="consensus")
        assert len(top) == 3
        assert top[0]["id"] == "c2"  # highest consensus first
        assert top[2]["id"] == "c1"  # lowest last

    def test_get_cluster_for_claim(self, tmp_path):
        """get_cluster_for_claim returns the cluster a claim belongs to."""
        db = self._setup_db(tmp_path)
        from veritas.models import ClaimCluster, Source, Claim

        src = Source(id="src1", url="https://example.com", title="Test")
        db.insert_source(src)
        db.insert_claims([Claim(id="cl1", source_id="src1", text="Test claim")])

        cluster = ClaimCluster(id="c1", representative_text="Test")
        db.upsert_clusters([cluster])
        db.insert_cluster_members([
            {"cluster_id": "c1", "claim_id": "cl1", "fingerprint": "test", "similarity_to_rep": 1.0}
        ])

        got = db.get_cluster_for_claim("cl1")
        assert got is not None
        assert got["id"] == "c1"

        # Non-existent claim
        assert db.get_cluster_for_claim("nonexistent") is None

    def test_empty_operations(self, tmp_path):
        """Empty inputs return 0 / empty."""
        db = self._setup_db(tmp_path)
        assert db.upsert_clusters([]) == 0
        assert db.insert_cluster_members([]) == 0
        assert db.get_top_clusters() == []


# ---------------------------------------------------------------------------
# Step 5: Consensus scoring tests
# ---------------------------------------------------------------------------

class TestConsensus:
    """Tests for compute_consensus()."""

    def test_empty_members(self):
        from veritas.knowledge_graph import compute_consensus
        status, conf, consensus = compute_consensus([])
        assert status == "unknown"
        assert conf == 0.0
        assert consensus == 0.0

    def test_single_source_no_boost(self):
        """Single source gets no consensus boost."""
        from veritas.knowledge_graph import compute_consensus
        members = [
            {"source_id": "s1", "status_auto": "supported", "auto_confidence": 0.85},
        ]
        status, conf, consensus = compute_consensus(members)
        assert status == "supported"
        assert conf == 0.85
        assert consensus == 0.85  # no boost with just 1 source

    def test_two_verified_sources_boost(self):
        """Two independently verified sources get +0.10 boost."""
        from veritas.knowledge_graph import compute_consensus
        members = [
            {"source_id": "s1", "status_auto": "supported", "auto_confidence": 0.85},
            {"source_id": "s2", "status_auto": "partial", "auto_confidence": 0.72},
        ]
        status, conf, consensus = compute_consensus(members)
        assert status == "supported"
        assert conf == 0.85
        assert consensus == 0.95  # 0.85 + 0.10

    def test_three_verified_sources_extra_boost(self):
        """Three verified sources get +0.10 + 0.05 = +0.15 total."""
        from veritas.knowledge_graph import compute_consensus
        members = [
            {"source_id": "s1", "status_auto": "supported", "auto_confidence": 0.80},
            {"source_id": "s2", "status_auto": "partial", "auto_confidence": 0.70},
            {"source_id": "s3", "status_auto": "supported", "auto_confidence": 0.75},
        ]
        status, conf, consensus = compute_consensus(members)
        assert consensus == 0.95  # 0.80 + 0.10 + 0.05

    def test_unverified_sources_no_boost(self):
        """Sources with 'unknown' status don't trigger consensus boost."""
        from veritas.knowledge_graph import compute_consensus
        members = [
            {"source_id": "s1", "status_auto": "unknown", "auto_confidence": 0.40},
            {"source_id": "s2", "status_auto": "unknown", "auto_confidence": 0.30},
        ]
        status, conf, consensus = compute_consensus(members)
        assert status == "unknown"
        assert consensus == 0.40  # no boost

    def test_cap_at_one(self):
        """Consensus score never exceeds 1.0."""
        from veritas.knowledge_graph import compute_consensus
        members = [
            {"source_id": f"s{i}", "status_auto": "supported", "auto_confidence": 0.95}
            for i in range(10)
        ]
        status, conf, consensus = compute_consensus(members)
        assert consensus <= 1.0

    def test_best_status_hierarchy(self):
        """supported > partial > unknown."""
        from veritas.knowledge_graph import compute_consensus
        # Mix of statuses: supported wins
        members = [
            {"source_id": "s1", "status_auto": "unknown", "auto_confidence": 0.30},
            {"source_id": "s2", "status_auto": "partial", "auto_confidence": 0.70},
            {"source_id": "s3", "status_auto": "supported", "auto_confidence": 0.85},
        ]
        status, _, _ = compute_consensus(members)
        assert status == "supported"

        # Only partial and unknown: partial wins
        members2 = [
            {"source_id": "s1", "status_auto": "unknown", "auto_confidence": 0.30},
            {"source_id": "s2", "status_auto": "partial", "auto_confidence": 0.70},
        ]
        status2, _, _ = compute_consensus(members2)
        assert status2 == "partial"


# ---------------------------------------------------------------------------
# Step 6: CLI registration tests
# ---------------------------------------------------------------------------

class TestCLIRegistration:
    """Tests that CLI commands are registered correctly."""

    def test_build_graph_command_exists(self):
        from veritas.cli import cli
        assert "build-graph" in [cmd.name for cmd in cli.commands.values()]

    def test_clusters_command_exists(self):
        from veritas.cli import cli
        assert "clusters" in [cmd.name for cmd in cli.commands.values()]

    def test_cluster_command_exists(self):
        from veritas.cli import cli
        assert "cluster" in [cmd.name for cmd in cli.commands.values()]


# ---------------------------------------------------------------------------
# Integration: full pipeline test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Integration test for the full knowledge graph pipeline."""

    def test_build_clusters_end_to_end(self):
        """Full pipeline: fingerprint → block → cluster → consensus."""
        from veritas.knowledge_graph import build_clusters, compute_consensus, ClaimRecord

        claims = [
            # Same fact: Alphabet revenue $350B — from 3 different sources
            ClaimRecord(claim_id="a1", source_id="s1",
                        text="Alphabet reported $350 billion revenue in 2024",
                        category="finance", status_auto="supported", auto_confidence=0.85),
            ClaimRecord(claim_id="a2", source_id="s2",
                        text="Google parent Alphabet had revenue of $350 billion for 2024",
                        category="finance", status_auto="partial", auto_confidence=0.72),
            ClaimRecord(claim_id="a3", source_id="s3",
                        text="Alphabet total revenue reached $350 billion in year 2024",
                        category="finance", status_auto="supported", auto_confidence=0.80),
            # Different fact: climate
            ClaimRecord(claim_id="b1", source_id="s1",
                        text="Global temperatures rose 1.5 degrees Celsius",
                        category="science", status_auto="unknown", auto_confidence=0.20),
        ]

        clusters = build_clusters(claims, threshold=0.30)

        # Should find at least 1 cluster for the Alphabet revenue claims
        assert len(clusters) >= 1

        # Find the Alphabet cluster
        alphabet_cluster = None
        for members in clusters.values():
            ids = {m.claim_id for m in members}
            if "a1" in ids:
                alphabet_cluster = members
                break

        assert alphabet_cluster is not None
        assert len(alphabet_cluster) >= 2  # at least 2 of the 3 should cluster

        # Climate claim should NOT be in the Alphabet cluster
        alphabet_ids = {m.claim_id for m in alphabet_cluster}
        assert "b1" not in alphabet_ids

        # Compute consensus for the Alphabet cluster
        member_dicts = [
            {"source_id": m.source_id, "status_auto": m.status_auto,
             "auto_confidence": m.auto_confidence}
            for m in alphabet_cluster
        ]
        best_status, best_conf, consensus = compute_consensus(member_dicts)
        assert best_status == "supported"
        assert consensus > best_conf  # consensus should boost above individual best
