"""Knowledge graph layer — claim fingerprinting, clustering, and consensus scoring.

Links claims about the same fact across different sources using deterministic
fuzzy fingerprinting (zero LLM). When multiple independent sources verify the
same claim, consensus confidence increases.

Pipeline: fingerprint → block → cluster → consensus → store
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .scoring import _normalise, _tokenize, _extract_claim_numbers, _CAT_TERMS
from .models import new_id


# ---------------------------------------------------------------------------
# Stopwords — common English function words with zero semantic value for
# fact-matching. Intentionally small (~50 words).
# ---------------------------------------------------------------------------

STOPWORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "not", "no",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "or", "and", "but", "if", "so", "than", "then", "that", "this",
    "it", "its", "their", "there", "about", "also", "just", "more",
])


# ---------------------------------------------------------------------------
# Step 1: Claim fingerprinting
# ---------------------------------------------------------------------------

def claim_fingerprint(text: str, category: str = "general") -> str:
    """Generate a deterministic fingerprint for fuzzy claim matching.

    Extracts semantic components (numbers, category terms, significant tokens)
    and joins them sorted into a readable |-delimited string.

    Same underlying fact with different wording produces similar fingerprints.
    """
    # Get all tokens minus stopwords
    tokens = _tokenize(text) - STOPWORDS

    # Extract numbers with unit conversion ($350 billion → "350000")
    numbers = _extract_claim_numbers(text)

    # Get category-relevant terms
    cat_terms = _CAT_TERMS.get(category, set())
    if category == "general":
        # For general claims, scan all category dicts for any matches
        all_cat_terms: Set[str] = set()
        for terms in _CAT_TERMS.values():
            all_cat_terms |= terms
        cat_terms = all_cat_terms
    matched_cat = cat_terms & tokens

    # Combine: significant tokens + numbers + category terms
    # Numbers are the strongest signal, followed by category terms, then tokens
    components = tokens | numbers | matched_cat
    if not components:
        return ""

    return "|".join(sorted(components))


def fingerprint_similarity(fp1: str, fp2: str) -> float:
    """Jaccard similarity between two fingerprints.

    Returns 0.0-1.0 where 1.0 means identical token sets.
    """
    if not fp1 or not fp2:
        return 0.0
    set1 = set(fp1.split("|"))
    set2 = set(fp2.split("|"))
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


# ---------------------------------------------------------------------------
# Step 2: Blocking (avoid O(n²) comparisons)
# ---------------------------------------------------------------------------

@dataclass
class ClaimRecord:
    """Lightweight claim data for clustering."""
    claim_id: str
    source_id: str
    text: str
    category: str
    fingerprint: str = ""
    numbers: frozenset = frozenset()
    status_auto: str = "unknown"
    auto_confidence: float = 0.0


def build_blocks(claims: List[ClaimRecord]) -> Dict[str, List[ClaimRecord]]:
    """Group claims into comparison blocks by category + shared number.

    Only claims in the same block are compared pairwise, reducing
    O(n²) to manageable sizes.
    """
    blocks: Dict[str, List[ClaimRecord]] = defaultdict(list)

    for claim in claims:
        if claim.numbers:
            for num in claim.numbers:
                key = f"{claim.category}|{num}"
                blocks[key].append(claim)
        else:
            key = f"{claim.category}|no_numbers"
            blocks[key].append(claim)

    return dict(blocks)


# ---------------------------------------------------------------------------
# Step 3: Union-Find clustering
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set data structure with path compression and union by rank."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def clusters(self) -> Dict[str, List[str]]:
        """Return mapping of root → list of all members."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return dict(groups)


def build_clusters(
    claims: List[ClaimRecord],
    threshold: float = 0.40,
) -> Dict[str, List[ClaimRecord]]:
    """Build claim clusters via fingerprinting + blocking + Union-Find.

    Returns mapping of cluster_id → list of ClaimRecords (only clusters with 2+ members).
    """
    # 1. Compute fingerprints and numbers
    for claim in claims:
        claim.fingerprint = claim_fingerprint(claim.text, claim.category)
        claim.numbers = frozenset(_extract_claim_numbers(claim.text))

    # 2. Build blocks
    blocks = build_blocks(claims)

    # 3. Pairwise comparison within blocks → Union-Find
    uf = UnionFind()
    claim_map = {c.claim_id: c for c in claims}

    for block_claims in blocks.values():
        n = len(block_claims)
        if n < 2 or n > 500:  # skip huge blocks (noise)
            continue
        for i in range(n):
            for j in range(i + 1, n):
                a, b = block_claims[i], block_claims[j]
                # Skip same-source pairs (cross-source only)
                if a.source_id == b.source_id:
                    continue
                sim = fingerprint_similarity(a.fingerprint, b.fingerprint)
                if sim >= threshold:
                    uf.union(a.claim_id, b.claim_id)

    # 4. Extract clusters with 2+ members
    raw_clusters = uf.clusters()
    result: Dict[str, List[ClaimRecord]] = {}
    for members in raw_clusters.values():
        if len(members) < 2:
            continue
        cluster_id = new_id()
        result[cluster_id] = [claim_map[m] for m in members if m in claim_map]

    return result


# ---------------------------------------------------------------------------
# Step 5: Consensus scoring
# ---------------------------------------------------------------------------

def compute_consensus(
    members: List[Dict],
) -> Tuple[str, float, float]:
    """Compute consensus score for a cluster of claims.

    Args:
        members: List of dicts with keys: source_id, status_auto, auto_confidence

    Returns:
        (best_status, best_confidence, consensus_score)

    Consensus only boosts claims that already have evidence.
    Multiple independent verified sources → higher confidence.
    """
    if not members:
        return "unknown", 0.0, 0.0

    # Best individual evidence
    best_confidence = max(m.get("auto_confidence", 0.0) for m in members)

    # Count unique sources with meaningful verification
    source_ids = {m["source_id"] for m in members}
    source_count = len(source_ids)

    verified_sources = set()
    for m in members:
        if m.get("status_auto") in ("supported", "partial"):
            verified_sources.add(m["source_id"])

    # Consensus boost: each additional verified source adds diminishing confidence
    consensus = best_confidence
    if len(verified_sources) >= 2:
        consensus += 0.10
    if len(verified_sources) >= 3:
        extra = min(len(verified_sources) - 2, 4)  # cap at 6 sources
        consensus += 0.05 * extra
    consensus = min(1.0, round(consensus, 4))

    # Best status across all members
    statuses = [m.get("status_auto", "unknown") for m in members]
    if "supported" in statuses:
        best_status = "supported"
    elif "partial" in statuses:
        best_status = "partial"
    else:
        best_status = "unknown"

    return best_status, best_confidence, consensus


# ---------------------------------------------------------------------------
# Step 6: Full pipeline orchestration
# ---------------------------------------------------------------------------

def build_knowledge_graph(threshold: float = 0.40) -> Dict:
    """Run full knowledge graph pipeline: fingerprint → block → cluster → consensus → store.

    Returns summary dict with stats.
    """
    from . import db  # local import to avoid circular dependency

    t0 = time.time()

    # 1. Load all claims
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, source_id, text, category, status_auto, auto_confidence "
            "FROM claims"
        ).fetchall()
    claims = [
        ClaimRecord(
            claim_id=r["id"], source_id=r["source_id"], text=r["text"],
            category=r["category"],
            status_auto=r["status_auto"] or "unknown",
            auto_confidence=r["auto_confidence"] or 0.0,
        )
        for r in rows
    ]

    # 2. Build clusters
    clusters = build_clusters(claims, threshold=threshold)

    # 3. Compute consensus and prepare storage
    from .models import ClaimCluster
    cluster_objects = []
    member_rows = []

    for cluster_id, members in clusters.items():
        # Pick representative: claim with highest auto_confidence, or longest text
        rep = max(members, key=lambda m: (m.auto_confidence, len(m.text)))

        # Compute consensus
        member_dicts = [
            {"source_id": m.source_id, "status_auto": m.status_auto,
             "auto_confidence": m.auto_confidence}
            for m in members
        ]
        best_status, best_conf, consensus = compute_consensus(member_dicts)

        source_count = len({m.source_id for m in members})

        cluster_obj = ClaimCluster(
            id=cluster_id,
            representative_text=rep.text,
            category=rep.category,
            claim_count=len(members),
            source_count=source_count,
            best_status=best_status,
            best_confidence=best_conf,
            consensus_score=consensus,
        )
        cluster_objects.append(cluster_obj)

        # Member rows
        for m in members:
            sim = fingerprint_similarity(rep.fingerprint, m.fingerprint)
            member_rows.append({
                "cluster_id": cluster_id,
                "claim_id": m.claim_id,
                "fingerprint": m.fingerprint,
                "similarity_to_rep": round(sim, 4),
            })

    # 4. Store
    db.clear_clusters()
    db.upsert_clusters(cluster_objects)
    db.insert_cluster_members(member_rows)

    elapsed = time.time() - t0

    return {
        "total_claims": len(claims),
        "clusters_found": len(clusters),
        "claims_clustered": sum(len(m) for m in clusters.values()),
        "largest_cluster": max((len(m) for m in clusters.values()), default=0),
        "avg_cluster_size": (
            round(sum(len(m) for m in clusters.values()) / len(clusters), 1)
            if clusters else 0
        ),
        "elapsed_seconds": round(elapsed, 1),
    }
