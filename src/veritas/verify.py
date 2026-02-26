"""Verify command — update claim status and attach evidence."""

from __future__ import annotations
from typing import List, Optional

from .models import Evidence, new_id
from . import db


VALID_STATUSES = ("supported", "contradicted", "partial", "unknown")
VALID_EVIDENCE_TYPES = ("primary", "secondary", "dataset", "filing", "gov", "paper", "other")
VALID_STRENGTHS = ("strong", "medium", "weak")


def verify_claim(
    claim_id: str,
    status: str,
    evidence_urls: Optional[List[str]] = None,
    evidence_type: str = "other",
    strength: str = "medium",
    notes: str = "",
) -> None:
    """Set a claim's status and optionally attach evidence links."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_STATUSES}")

    claim = db.get_claim(claim_id)
    if claim is None:
        raise ValueError(f"Claim '{claim_id}' not found.")

    db.update_claim_status(claim_id, status)

    for url in (evidence_urls or []):
        ev = Evidence(
            id=new_id(),
            claim_id=claim_id,
            url=url,
            evidence_type=evidence_type if evidence_type in VALID_EVIDENCE_TYPES else "other",
            strength=strength if strength in VALID_STRENGTHS else "medium",
            notes=notes,
        )
        db.insert_evidence(ev)
