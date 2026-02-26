"""Full-text search across all claims in the Veritas DB."""

from __future__ import annotations
from typing import List

from .models import Claim
from . import db


def search(query: str, limit: int = 20) -> List[Claim]:
    """Search claim text.  Returns up to *limit* matching Claims."""
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    return db.search_claims(query.strip(), limit=limit)
