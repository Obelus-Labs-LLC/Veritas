"""Veritas CLI — local claim-extraction and evidence-tracking engine.

Usage:
    python -m veritas <command> [options]
"""

from __future__ import annotations
import sys
import click
from rich.console import Console
from rich.table import Table

console = Console()

# ──────────────────────────────────────────────────────────────────
# Root group
# ──────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="veritas")
def cli():
    """Veritas — local claim-extraction and evidence-tracking engine."""


# ──────────────────────────────────────────────────────────────────
# ingest
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("url")
def ingest(url: str):
    """Download audio from URL and register a new source."""
    from .ingest import ingest as do_ingest

    console.print(f"[bold cyan]Ingesting:[/] {url}")
    try:
        source = do_ingest(url)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Done![/]  source_id = [bold]{source.id}[/]")
    console.print(f"  Title   : {source.title}")
    console.print(f"  Channel : {source.channel}")
    console.print(f"  Duration: {source.duration_seconds:.0f}s")
    console.print(f"  Audio   : {source.local_audio_path}")


# ──────────────────────────────────────────────────────────────────
# ingest-text (text/PDF file ingestion)
# ──────────────────────────────────────────────────────────────────

@cli.command("ingest-text")
@click.argument("file_path")
@click.option("--title", default="", help="Source title (defaults to filename).")
@click.option("--channel", default="", help="Author or channel name.")
def ingest_text(file_path: str, title: str, channel: str):
    """Ingest a text or PDF file directly (no audio, no transcription).

    Supports: .txt, .pdf, .md, .csv, .html
    The text is split into pseudo-segments and fed through the claim
    extraction pipeline.
    """
    from .ingest_text import ingest_text_file, ingest_pdf

    console.print(f"[bold cyan]Ingesting text:[/] {file_path}")

    try:
        if file_path.lower().endswith(".pdf"):
            source = ingest_pdf(file_path, title=title, channel=channel)
        else:
            source = ingest_text_file(file_path, title=title, channel=channel)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Done![/]  source_id = [bold]{source.id}[/]")
    console.print(f"  Title : {source.title}")
    console.print(f"  Type  : {source.source_type}")
    console.print(f"\nNext: run [bold]veritas claims {source.id}[/] to extract claims.")


# ──────────────────────────────────────────────────────────────────
# ingest-url (web article ingestion)
# ──────────────────────────────────────────────────────────────────

@cli.command("ingest-url")
@click.argument("url")
@click.option("--title", default="", help="Source title (auto-detected from page).")
@click.option("--channel", default="", help="Publisher or channel name.")
def ingest_url_cmd(url: str, title: str, channel: str):
    """Ingest a web article URL (no audio, extracts article text).

    Fetches the page, extracts article text, and creates pseudo-segments
    for the claim extraction pipeline.
    """
    from .ingest_text import ingest_url

    console.print(f"[bold cyan]Ingesting URL:[/] {url}")

    try:
        source = ingest_url(url, title=title, channel=channel)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Done![/]  source_id = [bold]{source.id}[/]")
    console.print(f"  Title : {source.title}")
    console.print(f"  Type  : {source.source_type}")
    console.print(f"\nNext: run [bold]veritas claims {source.id}[/] to extract claims.")


# ──────────────────────────────────────────────────────────────────
# transcribe
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("source_id")
@click.option("--model", default="small", show_default=True,
              type=click.Choice(["tiny", "base", "small", "medium", "large-v2", "large-v3"]),
              help="Whisper model size.")
@click.option("--device", default="cuda", show_default=True,
              type=click.Choice(["cuda", "cpu"]),
              help="Compute device.")
def transcribe(source_id: str, model: str, device: str):
    """Transcribe audio for a source using faster-whisper."""
    from .transcribe import transcribe as do_transcribe

    console.print(f"[bold cyan]Transcribing[/] source [bold]{source_id}[/] "
                  f"(model={model}, device={device})")
    try:
        meta, segments = do_transcribe(source_id, model_size=model, device=device)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Done![/]  {meta.segment_count} segments, language={meta.language}")
    console.print(f"  Transcript: {meta.transcript_path}")


# ──────────────────────────────────────────────────────────────────
# claims
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("source_id")
def claims(source_id: str):
    """Extract candidate claims from a transcript (deterministic, no LLM)."""
    from .claim_extract import extract_claims

    console.print(f"[bold cyan]Extracting claims[/] from source [bold]{source_id}[/]")
    try:
        result = extract_claims(source_id)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Done![/]  {len(result)} claims extracted.\n")

    table = Table(title="Candidate Claims", show_lines=True)
    table.add_column("ID", style="dim", width=14)
    table.add_column("Timestamp", width=18)
    table.add_column("Conf.", width=12)
    table.add_column("Category", width=10)
    table.add_column("Claim Text", ratio=1)

    for c in result[:30]:  # show first 30 in terminal
        ts = f"{_fmt_ts(c.ts_start)}-{_fmt_ts(c.ts_end)}"
        table.add_row(c.id, ts, c.confidence_language, c.category, c.text[:120])

    console.print(table)
    if len(result) > 30:
        console.print(f"  ... and {len(result) - 30} more (see claims.json)")


# ──────────────────────────────────────────────────────────────────
# verify
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_id")
@click.option("--status", required=True,
              type=click.Choice(["supported", "contradicted", "partial", "unknown"]),
              help="Verification status.")
@click.option("--add-evidence", multiple=True, help="URL(s) for evidence.")
@click.option("--evidence-type", default="other",
              type=click.Choice(["primary", "secondary", "dataset", "filing", "gov", "paper", "other"]),
              help="Type of evidence.")
@click.option("--strength", default="medium",
              type=click.Choice(["strong", "medium", "weak"]),
              help="Evidence strength.")
@click.option("--notes", default="", help="Free-text notes.")
def verify(claim_id: str, status: str, add_evidence: tuple, evidence_type: str,
           strength: str, notes: str):
    """Update a claim's verification status and attach evidence."""
    from .verify import verify_claim

    try:
        verify_claim(
            claim_id=claim_id,
            status=status,
            evidence_urls=list(add_evidence) if add_evidence else None,
            evidence_type=evidence_type,
            strength=strength,
            notes=notes,
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Claim {claim_id}[/] → status=[bold]{status}[/]")
    if add_evidence:
        for url in add_evidence:
            console.print(f"  + evidence: {url}")


# ──────────────────────────────────────────────────────────────────
# review (interactive claim verification)
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("source_id")
def review(source_id: str):
    """Interactively review and verify claims for a source."""
    from . import db as _db
    from .verify import verify_claim, VALID_STATUSES

    source = _db.get_source(source_id)
    if source is None:
        console.print(f"[bold red]Error:[/] Source '{source_id}' not found.")
        sys.exit(1)

    claims = _db.get_claims_for_source(source_id)
    if not claims:
        console.print(f"[yellow]No claims for source '{source_id}'. Run `veritas claims` first.[/]")
        return

    console.print(f"\n[bold cyan]Review claims for:[/] {source.title}")
    console.print(f"  Source ID: {source_id}  |  {len(claims)} claims\n")

    # Show claim list
    table = Table(show_lines=False, padding=(0, 1))
    table.add_column("#", style="bold", width=4)
    table.add_column("ID", style="dim", width=14)
    table.add_column("Status", width=14)
    table.add_column("Time", width=18)
    table.add_column("Claim", ratio=1)

    for i, c in enumerate(claims, 1):
        status_icon = {"supported": "[green]supported[/]", "contradicted": "[red]contradicted[/]",
                       "partial": "[yellow]partial[/]", "unknown": "[dim]unknown[/]"}.get(c.status, c.status)
        ts = f"{_fmt_ts(c.ts_start)}-{_fmt_ts(c.ts_end)}"
        table.add_row(str(i), c.id, status_icon, ts, c.text[:100])

    console.print(table)
    console.print()

    # Interactive loop
    while True:
        console.print("[bold]Enter claim # to verify (or 'q' to quit):[/] ", end="")
        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice.lower() in ("q", "quit", "exit", ""):
            break

        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(claims):
                console.print(f"[red]Invalid number. Enter 1-{len(claims)}.[/]")
                continue
        except ValueError:
            console.print("[red]Enter a number or 'q'.[/]")
            continue

        claim = claims[idx]
        console.print(f"\n[bold]Claim #{idx + 1}[/] ({claim.id})")
        console.print(f"  [italic]\"{claim.text}\"[/]")
        console.print(f"  Timestamp: {_fmt_ts(claim.ts_start)} - {_fmt_ts(claim.ts_end)}")
        console.print(f"  Current status: {claim.status}")

        # Get evidence already attached
        evidence = _db.get_evidence_for_claim(claim.id)
        if evidence:
            console.print(f"  Evidence: {len(evidence)} item(s)")
            for ev in evidence:
                console.print(f"    [{ev.evidence_type}] {ev.url}")

        # Prompt for status
        console.print(f"\n  Status? (s)upported / (c)ontradicted / (p)artial / (u)nknown / Enter=skip: ", end="")
        try:
            status_input = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        status_map = {"s": "supported", "c": "contradicted", "p": "partial", "u": "unknown",
                      "supported": "supported", "contradicted": "contradicted",
                      "partial": "partial", "unknown": "unknown"}
        if not status_input:
            console.print("  [dim]Skipped.[/]\n")
            continue
        status = status_map.get(status_input)
        if not status:
            console.print(f"  [red]Invalid status. Skipped.[/]\n")
            continue

        # Prompt for evidence URL (optional)
        console.print("  Evidence URL (optional, Enter to skip): ", end="")
        try:
            ev_url = input().strip()
        except (EOFError, KeyboardInterrupt):
            ev_url = ""

        # Prompt for notes (optional)
        console.print("  Notes (optional, Enter to skip): ", end="")
        try:
            notes = input().strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""

        # Apply verification
        try:
            verify_claim(
                claim_id=claim.id,
                status=status,
                evidence_urls=[ev_url] if ev_url else None,
                notes=notes,
            )
        except Exception as exc:
            console.print(f"  [bold red]Error:[/] {exc}\n")
            continue

        console.print(f"  [bold green]Claim updated -> {status}[/]\n")

        # Refresh the claim in our local list
        updated = _db.get_claim(claim.id)
        if updated:
            claims[idx] = updated

    console.print("[dim]Review session ended.[/]")


# ──────────────────────────────────────────────────────────────────
# export
# ──────────────────────────────────────────────────────────────────

@cli.command("export")
@click.argument("source_id")
@click.option("--format", "fmt", default="md", type=click.Choice(["md", "json"]),
              show_default=True, help="Export format.")
@click.option("--max-quotes", default=10, show_default=True,
              help="Max claims/quotes in the brief.")
def export_cmd(source_id: str, fmt: str, max_quotes: int):
    """Generate a source-cited brief (Markdown or JSON)."""
    from .export import export_markdown, export_json

    try:
        if fmt == "md":
            path = export_markdown(source_id, max_quotes)
        else:
            path = export_json(source_id, max_quotes)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Brief exported:[/] {path}")


# ──────────────────────────────────────────────────────────────────
# search
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--limit", default=20, show_default=True, help="Max results.")
def search(query: str, limit: int):
    """Full-text search across all extracted claims."""
    from .search import search as do_search

    try:
        results = do_search(query, limit=limit)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    if not results:
        console.print("[yellow]No claims matched your query.[/]")
        return

    table = Table(title=f"Search: \"{query}\"", show_lines=True)
    table.add_column("Claim ID", style="dim", width=14)
    table.add_column("Source", width=14)
    table.add_column("Status", width=14)
    table.add_column("Claim Text", ratio=1)

    for c in results:
        table.add_row(c.id, c.source_id, c.status, c.text[:120])

    console.print(table)
    console.print(f"  {len(results)} result(s)")


# ──────────────────────────────────────────────────────────────────
# doctor
# ──────────────────────────────────────────────────────────────────

@cli.command()
def doctor():
    """Check runtime environment and dependencies."""
    from .doctor import run_checks

    console.print("[bold cyan]Veritas Doctor[/] — checking environment...\n")
    checks = run_checks()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", width=24)
    table.add_column("Status", width=8)
    table.add_column("Detail", ratio=1)

    all_ok = True
    for name, passed, detail in checks:
        icon = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        if not passed:
            all_ok = False
        table.add_row(name, icon, detail)

    console.print(table)
    console.print()
    if all_ok:
        console.print("[bold green]All checks passed.[/]")
    else:
        console.print("[bold yellow]Some checks failed — see above for details.[/]")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────
# sources (enhanced with verification metrics)
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--by", "sort_by", default=None,
              type=click.Choice(["verified_rate", "claims", "date"]),
              help="Sort sources by metric.")
def sources(sort_by: str | None):
    """List all ingested sources with verification metrics."""
    from . import db as _db

    stats = _db.get_source_verification_stats()
    if not stats:
        console.print("[yellow]No sources yet. Run `veritas ingest <url>` to add one.[/]")
        return

    # Sort if requested
    if sort_by == "verified_rate":
        stats.sort(key=lambda d: d["verified_rate"], reverse=True)
    elif sort_by == "claims":
        stats.sort(key=lambda d: d["total_claims"], reverse=True)
    # default (date) is already sorted by created_at DESC

    table = Table(title="Ingested Sources", show_lines=True)
    table.add_column("ID", style="bold", width=14)
    table.add_column("Title", ratio=1)
    table.add_column("Channel", width=16)
    table.add_column("Claims", width=7, justify="right")
    table.add_column("Sup.", width=5, justify="right")
    table.add_column("Part.", width=5, justify="right")
    table.add_column("Unk.", width=5, justify="right")
    table.add_column("Verified%", width=10, justify="right")

    for d in stats:
        rate = d["verified_rate"]
        rate_styled = f"[green]{rate:.1f}%[/]" if rate >= 20 else (
            f"[yellow]{rate:.1f}%[/]" if rate > 0 else f"[dim]{rate:.1f}%[/]"
        )
        table.add_row(
            d["source_id"], d["title"][:50], d["channel"][:16],
            str(d["total_claims"]),
            f"[green]{d['supported']}[/]",
            f"[yellow]{d['partial']}[/]",
            str(d["unknown_count"]),
            rate_styled,
        )

    console.print(table)


# ──────────────────────────────────────────────────────────────────
# spread (cross-source claim spread)
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_hash")
def spread(claim_hash: str):
    """Show where a claim appears across sources (by global hash or claim ID)."""
    from . import db as _db

    # Accept either a claim_hash_global directly, or a claim_id to resolve
    ghash = claim_hash
    if len(claim_hash) < 20:
        # Looks like a short claim ID — resolve to global hash
        claim = _db.get_claim(claim_hash)
        if claim is None:
            console.print(f"[bold red]Error:[/] Claim '{claim_hash}' not found.")
            sys.exit(1)
        ghash = claim.claim_hash_global
        if not ghash:
            console.print(f"[yellow]Claim '{claim_hash}' has no global hash (no cross-source matching possible).[/]")
            return

    occurrences = _db.get_claim_spread(ghash)
    if not occurrences:
        console.print(f"[yellow]No claims found with global hash '{ghash[:16]}...'[/]")
        return

    console.print(f"\n[bold cyan]Claim Spread[/] — global hash [dim]{ghash[:16]}...[/]")
    console.print(f"  Appears in [bold]{len(occurrences)}[/] occurrence(s) across "
                  f"[bold]{len({o['source_id'] for o in occurrences})}[/] source(s)\n")

    table = Table(show_lines=True)
    table.add_column("Source", ratio=1)
    table.add_column("Claim ID", style="dim", width=14)
    table.add_column("Time", width=10)
    table.add_column("Auto Status", width=12)
    table.add_column("Conf.", width=6)
    table.add_column("Claim Text", ratio=2)

    for o in occurrences:
        status_styled = {
            "supported": "[green]supported[/]",
            "partial": "[yellow]partial[/]",
            "unknown": "[dim]unknown[/]",
        }.get(o["status_auto"], o["status_auto"])
        conf = f"{o['auto_confidence']:.0%}" if o["auto_confidence"] > 0 else "-"
        table.add_row(
            o["source_title"][:40], o["claim_id"],
            _fmt_ts(o["ts_start"]), status_styled, conf, o["text"][:100],
        )

    console.print(table)


# ──────────────────────────────────────────────────────────────────
# timeline (chronological claim propagation)
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("claim_hash")
def timeline(claim_hash: str):
    """Show chronological propagation of a claim across sources."""
    from . import db as _db

    # Accept either a claim_hash_global directly, or a claim_id to resolve
    ghash = claim_hash
    if len(claim_hash) < 20:
        claim = _db.get_claim(claim_hash)
        if claim is None:
            console.print(f"[bold red]Error:[/] Claim '{claim_hash}' not found.")
            sys.exit(1)
        ghash = claim.claim_hash_global
        if not ghash:
            console.print(f"[yellow]Claim '{claim_hash}' has no global hash.[/]")
            return

    entries = _db.get_claim_timeline(ghash)
    if not entries:
        console.print(f"[yellow]No timeline data for hash '{ghash[:16]}...'[/]")
        return

    console.print(f"\n[bold cyan]Claim Timeline[/] — global hash [dim]{ghash[:16]}...[/]")
    console.print(f"  Representative text: [italic]\"{entries[0]['text'][:100]}\"[/]\n")

    for i, e in enumerate(entries):
        marker = "[green]FIRST[/]" if i == 0 else f"[dim]+{i}[/]"
        status_styled = {
            "supported": "[green]supported[/]",
            "partial": "[yellow]partial[/]",
            "unknown": "[dim]unknown[/]",
        }.get(e["status_auto"], e["status_auto"])
        date_part = e["source_date"][:10] if e["source_date"] else "unknown"
        console.print(f"  {marker}  {date_part}  {e['source_title'][:40]}")
        console.print(f"      Status: {status_styled}  |  Claim: {e['text'][:80]}")
        console.print()


# ──────────────────────────────────────────────────────────────────
# top-claims (most-repeated claims across sources)
# ──────────────────────────────────────────────────────────────────

@cli.command("top-claims")
@click.option("--by", "sort_by", default="frequency",
              type=click.Choice(["frequency", "confidence"]),
              help="Sort by frequency or best confidence.")
@click.option("--limit", default=20, show_default=True, help="Max results.")
def top_claims(sort_by: str, limit: int):
    """Show most-repeated claims across all sources."""
    from . import db as _db

    results = _db.get_top_claims(limit=limit)
    if not results:
        console.print("[yellow]No cross-source claims found. Need claims in 2+ sources with matching global hashes.[/]")
        return

    if sort_by == "confidence":
        results.sort(key=lambda d: d["best_confidence"], reverse=True)

    console.print(f"\n[bold cyan]Top Cross-Source Claims[/] — sorted by {sort_by}\n")

    table = Table(show_lines=True)
    table.add_column("#", width=4)
    table.add_column("Sources", width=8, justify="right")
    table.add_column("Freq.", width=6, justify="right")
    table.add_column("Best", width=10)
    table.add_column("Cat.", width=10)
    table.add_column("Claim Text", ratio=1)
    table.add_column("Hash", style="dim", width=12)

    for i, r in enumerate(results, 1):
        best_styled = {
            "supported": "[green]supported[/]",
            "partial": "[yellow]partial[/]",
            "unknown": "[dim]unknown[/]",
        }.get(r["best_status"], r["best_status"])
        table.add_row(
            str(i), str(r["source_count"]), str(r["frequency"]),
            best_styled, r["category"], r["text"][:100],
            r["claim_hash_global"][:12],
        )

    console.print(table)
    console.print(f"\n  {len(results)} cross-source claim(s) shown")


# ──────────────────────────────────────────────────────────────────
# assist (auto evidence discovery)
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("source_id")
@click.option("--max-per-claim", default=5, show_default=True,
              help="Max evidence suggestions per claim.")
@click.option("--budget-minutes", default=0, show_default=True,
              help="Time budget in minutes (0 = no limit, process all claims).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Search and score but don't store or update status.")
def assist(source_id: str, max_per_claim: int, budget_minutes: int, dry_run: bool):
    """Auto-discover evidence for claims using free APIs."""
    from .assist import assist_source

    if dry_run:
        console.print("[bold yellow]DRY RUN[/] — will search and score but not store anything.\n")

    console.print(f"[bold cyan]Assisted verification[/] for source [bold]{source_id}[/]")
    budget_label = f"{budget_minutes} min" if budget_minutes > 0 else "unlimited"
    console.print(f"  Budget: {budget_label}  |  Max per claim: {max_per_claim}\n")

    try:
        report = assist_source(
            source_id,
            max_per_claim=max_per_claim,
            budget_minutes=budget_minutes,
            dry_run=dry_run,
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    # Summary
    console.print(f"\n[bold green]Done![/]  Processed {report['claims_processed']}/{report['claims_total']} claims "
                  f"in {report['elapsed_seconds']}s\n")

    table = Table(title="Assist Summary", show_lines=False)
    table.add_column("Metric", width=30)
    table.add_column("Value", width=12)

    table.add_row("Evidence suggestions found", str(report["total_suggestions_found"]))
    table.add_row("Evidence suggestions stored", str(report["total_suggestions_stored"]))
    table.add_row("Auto -> SUPPORTED", f"[green]{report['auto_supported']}[/]")
    table.add_row("Auto -> PARTIAL", f"[yellow]{report['auto_partial']}[/]")
    table.add_row("Auto -> UNKNOWN", f"[dim]{report['auto_unknown']}[/]")
    skipped = report.get("claims_skipped_low_verifiability", 0)
    if skipped > 0:
        table.add_row("Skipped (low verifiability)", f"[dim]{skipped}[/]")
    console.print(table)

    # Top claims needing review
    needs_review = [r for r in report["claim_reports"] if r["status_auto"] == "unknown"]
    if needs_review:
        console.print(f"\n[bold]Top claims needing manual review:[/]")
        for r in needs_review[:10]:
            console.print(f"  {r['claim_id']}  [{r['category']:8s}]  "
                          f"best_score={r['best_score']:2d}  {r['text_excerpt']}")


# ──────────────────────────────────────────────────────────────────
# queue (review queue sorted by priority)
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--limit", default=20, show_default=True, help="Max items to show.")
def queue(limit: int):
    """Show claims needing review, sorted by priority."""
    from . import db as _db

    claims = _db.get_review_queue(limit)
    if not claims:
        console.print("[green]No claims in the review queue.[/]")
        return

    table = Table(title="Review Queue", show_lines=False, padding=(0, 1))
    table.add_column("ID", style="dim", width=14)
    table.add_column("Final", width=14)
    table.add_column("Auto", width=10)
    table.add_column("Conf", width=6)
    table.add_column("Cat", width=10)
    table.add_column("Claim", ratio=1)

    for c in claims:
        final = c.final_status
        final_styled = {
            "supported": "[green]supported[/]",
            "contradicted": "[red]contradicted[/]",
            "partial": "[yellow]partial[/]",
            "unknown": "[dim]unknown[/]",
        }.get(final, final)
        auto_styled = {
            "supported": "[green]auto:s[/]",
            "partial": "[yellow]auto:p[/]",
            "unknown": "[dim]auto:?[/]",
        }.get(c.status_auto, c.status_auto)
        conf = f"{c.auto_confidence:.0%}" if c.auto_confidence > 0 else "-"

        table.add_row(c.id, final_styled, auto_styled, conf, c.category, c.text[:80])

    console.print(table)
    console.print(f"\n  {len(claims)} claim(s) shown  |  Run [bold]veritas review <source_id>[/] to verify interactively")


# ──────────────────────────────────────────────────────────────────
# inspect-verified (inspect auto-verified claims)
# ──────────────────────────────────────────────────────────────────

@cli.command("inspect-verified")
@click.option("--status", default="", type=click.Choice(["", "supported", "partial"]),
              help="Filter by auto status (default: both).")
@click.option("--source", "source_id", default="", help="Filter by source ID.")
@click.option("--category", default="", help="Filter by category.")
@click.option("--limit", default=50, show_default=True, help="Max claims to show.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show full evidence details.")
def inspect_verified(status: str, source_id: str, category: str, limit: int, verbose: bool):
    """Inspect auto-verified claims with their evidence and scoring signals.

    Shows all claims that received SUPPORTED or PARTIAL auto-status,
    along with the evidence that triggered the verification and the
    scoring signals that contributed to the score.
    """
    from . import db as _db

    results = _db.get_verified_claims(
        status_filter=status,
        source_id=source_id,
        category=category,
        limit=limit,
    )

    if not results:
        console.print("[yellow]No auto-verified claims found matching filters.[/]")
        return

    # Count stats
    supported = sum(1 for r in results if r["status_auto"] == "supported")
    partial = sum(1 for r in results if r["status_auto"] == "partial")

    console.print(f"\n[bold cyan]Inspect Auto-Verified Claims[/]")
    console.print(f"  [green]SUPPORTED: {supported}[/]  |  [yellow]PARTIAL: {partial}[/]  |  Total: {len(results)}\n")

    if verbose:
        # Detailed view: one block per claim
        for i, r in enumerate(results, 1):
            status_styled = "[green]SUPPORTED[/]" if r["status_auto"] == "supported" else "[yellow]PARTIAL[/]"
            human = f" [bold](human: {r['status_human']})[/]" if r.get("status_human") else ""
            conf = f"{r['auto_confidence']:.0%}" if r.get("auto_confidence") else "-"

            console.print(f"[bold]#{i}[/]  {status_styled}{human}  score={r.get('best_score', 0)}  confidence={conf}")
            console.print(f"  [italic]\"{r['claim_text'][:150]}\"[/]")
            console.print(f"  Source: {r.get('source_title', '')[:60]}  |  Category: {r.get('category', '')}")

            if r.get("claim_date"):
                console.print(f"  Claim date: {r['claim_date']}")

            # Evidence details
            if r.get("best_url"):
                console.print(f"  Evidence: {r.get('best_evidence_title', '')[:80]}")
                console.print(f"  URL: {r['best_url']}")
                console.print(f"  Source API: {r.get('best_source', '')}")

            if r.get("best_signals"):
                console.print(f"  Signals: {r['best_signals']}")

            if r.get("best_snippet"):
                snippet = r['best_snippet'][:150].encode("ascii", errors="replace").decode("ascii")
                console.print(f"  Snippet: {snippet}")

            console.print()
    else:
        # Table view
        table = Table(show_lines=True)
        table.add_column("#", width=4)
        table.add_column("Status", width=10)
        table.add_column("Score", width=6, justify="right")
        table.add_column("Cat.", width=10)
        table.add_column("Claim Text", ratio=2)
        table.add_column("Best Source", width=14)
        table.add_column("Signals", ratio=1)

        for i, r in enumerate(results, 1):
            status_styled = "[green]SUPPORTED[/]" if r["status_auto"] == "supported" else "[yellow]PARTIAL[/]"
            table.add_row(
                str(i),
                status_styled,
                str(r.get("best_score", 0)),
                r.get("category", ""),
                r.get("claim_text", "")[:100],
                r.get("best_source", "")[:14],
                r.get("best_signals", "")[:60],
            )

        console.print(table)

    console.print(f"\n  {len(results)} claim(s) shown")
    console.print(f"  Tip: use [bold]--verbose[/] for full evidence details")


# ──────────────────────────────────────────────────────────────────
# build-graph (knowledge graph pipeline)
# ──────────────────────────────────────────────────────────────────

@cli.command("build-graph")
@click.option("--threshold", default=0.40, show_default=True,
              help="Jaccard similarity threshold for clustering (0.0-1.0).")
def build_graph(threshold: float):
    """Build the knowledge graph: fingerprint, cluster, and score claims across sources."""
    from .knowledge_graph import build_knowledge_graph

    console.print(f"[bold cyan]Building knowledge graph[/] (threshold={threshold})\n")

    try:
        result = build_knowledge_graph(threshold=threshold)
    except Exception as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[bold green]Done![/]  {result['elapsed_seconds']}s\n")

    table = Table(title="Knowledge Graph Summary", show_lines=False)
    table.add_column("Metric", width=24)
    table.add_column("Value", width=12)

    table.add_row("Total claims", str(result["total_claims"]))
    table.add_row("Clusters found", f"[bold]{result['clusters_found']}[/]")
    table.add_row("Claims clustered", str(result["claims_clustered"]))
    table.add_row("Largest cluster", str(result["largest_cluster"]))
    table.add_row("Avg cluster size", str(result["avg_cluster_size"]))

    console.print(table)
    console.print(f"\n  Run [bold]veritas clusters[/] to inspect top clusters.")


# ──────────────────────────────────────────────────────────────────
# clusters (list top clusters)
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--limit", default=20, show_default=True, help="Max clusters to show.")
@click.option("--by", "sort_by", default="consensus",
              type=click.Choice(["consensus", "sources", "claims"]),
              help="Sort clusters by metric.")
def clusters(limit: int, sort_by: str):
    """Show top claim clusters from the knowledge graph."""
    from . import db as _db

    results = _db.get_top_clusters(limit=limit, sort_by=sort_by)
    if not results:
        console.print("[yellow]No clusters found. Run `veritas build-graph` first.[/]")
        return

    console.print(f"\n[bold cyan]Claim Clusters[/] — sorted by {sort_by}\n")

    table = Table(show_lines=True)
    table.add_column("#", width=4)
    table.add_column("ID", style="dim", width=14)
    table.add_column("Sources", width=8, justify="right")
    table.add_column("Claims", width=8, justify="right")
    table.add_column("Status", width=10)
    table.add_column("Conf.", width=6, justify="right")
    table.add_column("Consensus", width=10, justify="right")
    table.add_column("Cat.", width=10)
    table.add_column("Representative Text", ratio=1)

    for i, r in enumerate(results, 1):
        status_styled = {
            "supported": "[green]supported[/]",
            "partial": "[yellow]partial[/]",
            "unknown": "[dim]unknown[/]",
        }.get(r["best_status"], r["best_status"])
        conf = f"{r['best_confidence']:.0%}" if r["best_confidence"] > 0 else "-"
        consensus = f"{r['consensus_score']:.0%}" if r["consensus_score"] > 0 else "-"
        table.add_row(
            str(i), r["id"], str(r["source_count"]), str(r["claim_count"]),
            status_styled, conf, consensus, r["category"],
            r["representative_text"][:80],
        )

    console.print(table)
    console.print(f"\n  {len(results)} cluster(s) shown  |  Run [bold]veritas cluster <id>[/] for details.")


# ──────────────────────────────────────────────────────────────────
# cluster (detail view of a single cluster)
# ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("cluster_id")
def cluster(cluster_id: str):
    """Show detailed view of a single cluster and its member claims."""
    from . import db as _db

    cluster_data = _db.get_cluster(cluster_id)
    if cluster_data is None:
        console.print(f"[bold red]Error:[/] Cluster '{cluster_id}' not found.")
        sys.exit(1)

    members = _db.get_cluster_members(cluster_id)

    status_styled = {
        "supported": "[green]SUPPORTED[/]",
        "partial": "[yellow]PARTIAL[/]",
        "unknown": "[dim]UNKNOWN[/]",
    }.get(cluster_data["best_status"], cluster_data["best_status"])

    console.print(f"\n[bold cyan]Cluster Detail[/] — {cluster_id}")
    console.print(f"  Representative: [italic]\"{cluster_data['representative_text'][:120]}\"[/]")
    console.print(f"  Category: {cluster_data['category']}  |  Status: {status_styled}")
    conf = f"{cluster_data['best_confidence']:.0%}" if cluster_data["best_confidence"] > 0 else "-"
    consensus = f"{cluster_data['consensus_score']:.0%}" if cluster_data["consensus_score"] > 0 else "-"
    console.print(f"  Confidence: {conf}  |  Consensus: {consensus}")
    console.print(f"  Sources: {cluster_data['source_count']}  |  Claims: {cluster_data['claim_count']}\n")

    if not members:
        console.print("[yellow]No member claims found (data may be stale — rebuild graph).[/]")
        return

    table = Table(title="Member Claims", show_lines=True)
    table.add_column("#", width=4)
    table.add_column("Source", ratio=1)
    table.add_column("Claim ID", style="dim", width=14)
    table.add_column("Auto", width=10)
    table.add_column("Conf.", width=6, justify="right")
    table.add_column("Similarity", width=10, justify="right")
    table.add_column("Claim Text", ratio=2)

    for i, m in enumerate(members, 1):
        auto_styled = {
            "supported": "[green]supported[/]",
            "partial": "[yellow]partial[/]",
            "unknown": "[dim]unknown[/]",
        }.get(m.get("status_auto", ""), m.get("status_auto", ""))
        conf = f"{m['auto_confidence']:.0%}" if m.get("auto_confidence", 0) > 0 else "-"
        sim = f"{m['similarity_to_rep']:.0%}" if m.get("similarity_to_rep", 0) > 0 else "-"
        table.add_row(
            str(i),
            m.get("source_title", "")[:30],
            m["claim_id"],
            auto_styled,
            conf,
            sim,
            m.get("text", "")[:100],
        )

    console.print(table)


# ──────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────

def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
