"""Batch re-run assisted verification on all sources.

Usage: python scripts/batch_assist.py [--dry-run]

Processes sources in order of lowest coverage first.
Logs progress to scripts/batch_assist.log
"""
import sys
import os
import time
import sqlite3
from datetime import datetime

# Ensure veritas package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veritas import db, assist

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "veritas.sqlite")
LOG_PATH = os.path.join(os.path.dirname(__file__), "batch_assist.log")

# SEC budget already re-run, and duplicate source with 0 claims
SKIP_IDS = {"c6961d7ad734", "31ff785bd56c"}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_sources_by_coverage():
    """Return source IDs sorted by lowest coverage first."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.title, COUNT(c.id) as total_claims,
               COUNT(DISTINCT CASE WHEN es.id IS NOT NULL THEN c.id END) as claims_with_ev
        FROM sources s
        JOIN claims c ON c.source_id = s.id
        LEFT JOIN evidence_suggestions es ON es.claim_id = c.id
        GROUP BY s.id
        HAVING total_claims > 0
        ORDER BY (CAST(COUNT(DISTINCT CASE WHEN es.id IS NOT NULL THEN c.id END) AS FLOAT)
                  / COUNT(c.id)) ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return [(r[0], r[1], r[2], r[3]) for r in rows if r[0] not in SKIP_IDS]


def run_batch(dry_run: bool = False):
    sources = get_sources_by_coverage()
    total_sources = len(sources)
    total_claims = sum(s[2] for s in sources)

    log(f"Batch assist: {total_sources} sources, {total_claims} total claims")
    log(f"Dry run: {dry_run}")
    log("")

    batch_start = time.time()
    completed = 0
    total_suggestions_all = 0

    for i, (sid, title, claim_count, prev_ev) in enumerate(sources, 1):
        cover = prev_ev / claim_count * 100 if claim_count > 0 else 0
        log(f"[{i}/{total_sources}] Starting: {title[:60]}")
        log(f"  Source ID: {sid} | Claims: {claim_count} | Previous coverage: {cover:.1f}%")

        try:
            t0 = time.time()
            result = assist.assist_source(
                source_id=sid,
                max_per_claim=5,
                budget_minutes=0,  # unlimited
                dry_run=dry_run,
            )
            elapsed = time.time() - t0

            stored = result.get("suggestions_stored", 0)
            found = result.get("suggestions_found", 0)
            processed = result.get("claims_processed", 0)
            skipped = result.get("claims_skipped_low_verifiability", 0)
            auto_sup = result.get("auto_supported", 0)
            auto_par = result.get("auto_partial", 0)
            auto_unk = result.get("auto_unknown", 0)

            total_suggestions_all += stored
            completed += 1

            log(f"  Done in {elapsed:.1f}s | Processed: {processed}/{claim_count} | "
                f"Skipped: {skipped} | Stored: {stored} | "
                f"SUP={auto_sup} PAR={auto_par} UNK={auto_unk}")

        except Exception as e:
            log(f"  ERROR: {e}")

        log("")

    batch_elapsed = time.time() - batch_start
    log(f"{'='*60}")
    log(f"Batch complete: {completed}/{total_sources} sources in {batch_elapsed:.1f}s "
        f"({batch_elapsed/60:.1f} min)")
    log(f"Total evidence suggestions stored: {total_suggestions_all}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    db.init_db()
    run_batch(dry_run=dry_run)
