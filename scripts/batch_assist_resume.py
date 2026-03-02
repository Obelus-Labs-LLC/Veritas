"""Resume batch assist for sources not completed in the v2 run.

The batch_assist.py run died at source 36/68. This script picks up
the remaining ~37 sources that weren't completed.

Usage: python scripts/batch_assist_resume.py
Logs to: scripts/batch_assist_v2.log
"""
import sys
import os
import re
import time
import sqlite3
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from veritas import db, assist

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "veritas.sqlite")
LOG_PATH = os.path.join(os.path.dirname(__file__), "batch_assist_v2.log")
MAIN_LOG = os.path.join(os.path.dirname(__file__), "batch_assist.log")

SKIP_IDS = {"c6961d7ad734", "31ff785bd56c"}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    safe_line = line.encode("ascii", errors="replace").decode()
    print(safe_line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_completed_from_log():
    """Parse source IDs that completed in the v2 batch run (started ~13:36)."""
    completed = set()
    try:
        with open(MAIN_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return completed

    # Find v2 run start (around 13:36 on 2026-03-01)
    in_v2 = False
    current_sid = None
    for line in lines:
        # The v2 run header line
        if "13:36" in line and "Batch assist:" in line:
            in_v2 = True
        if not in_v2:
            continue
        m = re.search(r"Source ID: ([a-f0-9]+)", line)
        if m:
            current_sid = m.group(1)
        if "Done in" in line and current_sid:
            completed.add(current_sid)
            current_sid = None
    return completed


def get_remaining_sources(completed_ids):
    """Return sources not yet completed, sorted by lowest coverage first."""
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
    return [
        (r[0], r[1], r[2], r[3])
        for r in rows
        if r[0] not in SKIP_IDS and r[0] not in completed_ids
    ]


def run_resume():
    completed_ids = get_completed_from_log()
    log(f"Sources already completed in v2 run: {len(completed_ids)}")

    sources = get_remaining_sources(completed_ids)
    total_sources = len(sources)
    total_claims = sum(s[2] for s in sources)

    log(f"Resuming batch: {total_sources} remaining sources, {total_claims} claims")
    log("")

    batch_start = time.time()
    completed = 0

    for i, (sid, title, claim_count, prev_ev) in enumerate(sources, 1):
        cover = prev_ev / claim_count * 100 if claim_count > 0 else 0
        log(f"[{i}/{total_sources}] Starting: {title[:60]}")
        log(f"  Source ID: {sid} | Claims: {claim_count} | Previous coverage: {cover:.1f}%")

        try:
            t0 = time.time()
            result = assist.assist_source(
                source_id=sid,
                max_per_claim=5,
                budget_minutes=0,
                dry_run=False,
            )
            elapsed = time.time() - t0

            stored = result.get("suggestions_stored", 0)
            processed = result.get("claims_processed", 0)
            skipped = result.get("claims_skipped_low_verifiability", 0)
            auto_sup = result.get("auto_supported", 0)
            auto_par = result.get("auto_partial", 0)
            auto_unk = result.get("auto_unknown", 0)

            completed += 1
            log(f"  Done in {elapsed:.1f}s | Processed: {processed}/{claim_count} | "
                f"Skipped: {skipped} | Stored: {stored} | "
                f"SUP={auto_sup} PAR={auto_par} UNK={auto_unk}")

        except Exception as e:
            log(f"  ERROR: {e}")

        log("")

    batch_elapsed = time.time() - batch_start
    log(f"{'='*60}")
    log(f"Resume complete: {completed}/{total_sources} sources in {batch_elapsed:.1f}s "
        f"({batch_elapsed/60:.1f} min)")


if __name__ == "__main__":
    db.init_db()
    run_resume()
