"""Export a source-cited brief as Markdown or JSON.

IMPORTANT: Never dumps the full transcript.  Exports contain:
  - source metadata
  - claim list with status + evidence
  - short timestamped quotes only (configurable max)
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from typing import List

from .config import DEFAULT_MAX_QUOTES
from .models import Source, Claim, Evidence
from .paths import source_export_dir
from . import db


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_brief_data(source_id: str, max_quotes: int = DEFAULT_MAX_QUOTES) -> dict:
    """Assemble the structured brief for a source."""
    source = db.get_source(source_id)
    if source is None:
        raise ValueError(f"Source '{source_id}' not found.")

    claims = db.get_claims_for_source(source_id)

    claims_data = []
    for c in claims[:max_quotes]:
        evidence = db.get_evidence_for_claim(c.id)
        suggestions = db.get_suggestions_for_claim(c.id, limit=3)
        claims_data.append({
            "id": c.id,
            "text": c.text,
            "timestamp": f"{_fmt_ts(c.ts_start)} - {_fmt_ts(c.ts_end)}",
            "confidence": c.confidence_language,
            "category": c.category,
            "final_status": c.final_status,
            "status_auto": c.status_auto,
            "auto_confidence": round(c.auto_confidence, 2),
            "status_human": c.status_human,
            "evidence": [
                {
                    "url": e.url,
                    "type": e.evidence_type,
                    "strength": e.strength,
                    "notes": e.notes,
                }
                for e in evidence
            ],
            "evidence_suggestions": [
                {
                    "url": s.url,
                    "title": s.title,
                    "source": s.source_name,
                    "score": s.score,
                }
                for s in suggestions
            ],
        })

    return {
        "title": source.title,
        "url": source.url,
        "channel": source.channel,
        "upload_date": source.upload_date,
        "duration": _fmt_ts(source.duration_seconds),
        "source_id": source.id,
        "total_claims": len(claims),
        "exported_claims": len(claims_data),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims": claims_data,
    }


def export_json(source_id: str, max_quotes: int = DEFAULT_MAX_QUOTES) -> str:
    """Write brief.json and return its path."""
    data = _build_brief_data(source_id, max_quotes)
    out = source_export_dir(source_id) / "brief.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    return str(out)


def export_markdown(source_id: str, max_quotes: int = DEFAULT_MAX_QUOTES) -> str:
    """Write brief.md and return its path."""
    d = _build_brief_data(source_id, max_quotes)

    lines = [
        f"# Veritas Brief: {d['title']}",
        "",
        f"**Source:** {d['url']}  ",
        f"**Channel:** {d['channel']}  ",
        f"**Uploaded:** {d['upload_date']}  ",
        f"**Duration:** {d['duration']}  ",
        f"**Source ID:** `{d['source_id']}`  ",
        f"**Total claims extracted:** {d['total_claims']}  ",
        f"**Generated:** {d['generated_at']}  ",
        "",
        "---",
        "",
        "## Claims",
        "",
    ]

    for i, c in enumerate(d["claims"], 1):
        final = c["final_status"]
        status_icon = {
            "supported": "✅",
            "contradicted": "❌",
            "partial": "⚠️",
            "unknown": "❓",
        }.get(final, "❓")

        # Show provenance: HUMAN override or AUTO
        provenance = "HUMAN" if c.get("status_human") else (
            f"AUTO ({c['auto_confidence']:.0%})" if c.get("status_auto", "unknown") != "unknown" else "UNVERIFIED"
        )

        lines.append(f"### {i}. {status_icon} [{final.upper()}] ({c['confidence']}) — {provenance}")
        lines.append("")
        lines.append(f"> \"{c['text']}\"")
        lines.append(f">")
        lines.append(f"> *Timestamp: {c['timestamp']}  |  Category: {c.get('category', 'general')}*")
        lines.append("")

        if c["evidence"]:
            lines.append("**Evidence (human-verified):**")
            for ev in c["evidence"]:
                lines.append(f"- [{ev['type']}] ({ev['strength']}) {ev['url']}")
                if ev["notes"]:
                    lines.append(f"  - {ev['notes']}")
            lines.append("")

        if c.get("evidence_suggestions"):
            lines.append("**Evidence suggestions (auto-discovered):**")
            for s in c["evidence_suggestions"]:
                lines.append(f"- [{s['source']}] (score: {s['score']}) {s['url']}")
                if s.get("title"):
                    lines.append(f"  - {s['title'][:100]}")
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by Veritas — local claim extraction engine.*")
    lines.append("")

    text = "\n".join(lines)
    out = source_export_dir(source_id) / "brief.md"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    return str(out)
