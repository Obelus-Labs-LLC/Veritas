# Veritas

**A deterministic claim extraction and fact-verification engine.**

Veritas extracts claims from audio, video, and text sources using NLP — no large language models — then verifies them against official data sources using rule-based methods. Every verification is transparent, auditable, and traceable to a primary source.

> Zero LLM dependency. Zero hallucination risk. Every claim maps to evidence or stays marked unknown.

---

## What It Does

Veritas takes any audio or video source (YouTube, podcasts, earnings calls, congressional hearings) and runs it through a fully deterministic pipeline:

1. **Ingest** — downloads audio via yt-dlp, extracts metadata
2. **Transcribe** — GPU-accelerated speech-to-text using faster-whisper (CTranslate2 / CUDA)
3. **Extract Claims** — rule-based NLP identifies checkable factual statements from the transcript. No LLM, no prompt engineering — uses sentence boundary detection, named entity recognition, assertion verb patterns, and signal scoring
4. **Categorize** — keyword-based classification (finance, health, science, tech, politics, military) routes each claim to the most relevant evidence sources
5. **Verify** — smart routing sends claims to free, structured APIs. Scoring uses token overlap, entity matching, number matching, keyphrase alignment, and evidence type weighting. Strict guardrails: a claim is only marked SUPPORTED when multiple scoring signals converge above threshold

The result: a structured database of claims, each linked to candidate evidence with full scoring transparency.

---

## Verification Approach

Veritas takes a fundamentally different approach from LLM-based fact-checkers:

- **Extraction is deterministic** — the same transcript always produces the same claims. No temperature, no sampling, no prompt sensitivity
- **Verification is rule-based** — scoring functions use token overlap, entity matching, exact number matching, and evidence type classification. No embeddings, no semantic similarity
- **Evidence comes from primary sources** — SEC filings, academic papers, government datasets. Not web search, not LLM-generated summaries
- **Unknown is the default** — if the evidence APIs return nothing relevant, the claim stays UNKNOWN. Veritas never guesses

### Auto Status Guardrails

| Status | Conditions |
|--------|-----------|
| **SUPPORTED** | Primary source + score ≥ 85 + keyphrase or exact number match |
| **PARTIAL** | Score 70–84, some evidence alignment |
| **UNKNOWN** | Everything else (the honest default) |

CONTRADICTED is never set automatically — too risky for an automated system.

---

## Evidence Sources

All free. No API keys required.

| Source | Type | Best For |
|--------|------|----------|
| **SEC EDGAR** | `filing` | Company financials, earnings, 10-K/10-Q/8-K filings |
| **yfinance** | `dataset` | Real-time market data, stock prices, market cap, revenue |
| **FRED** | `dataset` | Macroeconomic indicators — GDP, CPI, unemployment, federal funds rate |
| **Crossref** | `paper` | Academic papers across all fields (DOI-linked) |
| **arXiv** | `paper` | AI/ML, physics, mathematics, computer science preprints |
| **PubMed** | `paper` | Biomedical and health research (PMID-linked) |
| **Wikipedia** | `secondary` | Named entity context, background reference |

Smart routing analyzes claim content and boosts the most relevant sources. A claim mentioning "Alphabet revenue" gets routed to yfinance and SEC EDGAR first. A claim about "LDL cholesterol" gets routed to PubMed.

---

## Cross-Source Intelligence

Veritas doesn't just verify individual claims — it tracks how claims move across sources:

- **Spread Analysis** — identifies the same claim appearing across multiple videos or channels, scored by global content hash and fuzzy matching
- **Timeline Tracking** — maps when claims first appear and how they propagate
- **Top Claims Ranking** — surfaces the most-repeated claims across your entire corpus, ranked by cross-source frequency
- **Contradiction Detection** — flags cases where sources make conflicting factual assertions about the same topic

---

## Current Stats

| Metric | Value |
|--------|-------|
| Ingested sources | 21 |
| Extracted claims | 3,538 |
| Evidence suggestions gathered | 9,225 |
| Evidence sources | 7 (SEC EDGAR, yfinance, FRED, Crossref, arXiv, PubMed, Wikipedia) |
| Passing tests | 204 |
| Claim categories | 7 (finance, health, science, tech, politics, military, general) |

### Benchmark Results

| Source | Claims | Verified | Rate |
|--------|--------|----------|------|
| Alphabet Q4 2025 Earnings Call | 298 | 1 SUPPORTED + 8 PARTIAL | 3.0% |
| Trump State of the Union | 249 | 2 SUPPORTED + 8 PARTIAL | 4.0% |
| Ray Dalio: How The Economic Machine Works | 117 | 1 SUPPORTED + 4 PARTIAL | 4.3% |
| Veritasium: Game of Life | 185 | 7 PARTIAL | 3.8% |
| BlackRock Is Terrifying | 238 | 2 PARTIAL | 0.8% |

Verification rates are deliberately conservative. The scoring thresholds are strict because false positives are worse than false negatives in a verification system. A 4% rate against free APIs with no LLM and no web scraping represents genuine evidence alignment.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Transcription | faster-whisper (CTranslate2, CUDA-accelerated) |
| Audio download | yt-dlp |
| Database | SQLite (single-file, zero-config) |
| CLI | Click + Rich |
| NLP | Rule-based (no external NLP libraries) |
| Evidence APIs | requests (all free, no keys) |
| GPU support | NVIDIA CUDA 12 via pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) |
| Testing | pytest (204 tests) |

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/Obelus-Labs-LLC/Veritas.git
cd Veritas
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# (Recommended) Install CUDA-enabled PyTorch for GPU transcription
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Run the pipeline
python veritas.py ingest "https://www.youtube.com/watch?v=EXAMPLE"
python veritas.py transcribe <source_id>
python veritas.py claims <source_id>
python veritas.py assist <source_id>

# Check results
python veritas.py sources
python veritas.py queue
python veritas.py export <source_id> --format md
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `veritas ingest <url>` | Download audio, save metadata, register source |
| `veritas transcribe <id>` | Transcribe audio with faster-whisper (GPU) |
| `veritas claims <id>` | Extract candidate claims (deterministic, rule-based) |
| `veritas assist <id>` | Auto-discover evidence from 7 free APIs |
| `veritas queue` | Show claims needing review, sorted by priority |
| `veritas review <id>` | Interactively review and verify claims |
| `veritas verify <claim_id>` | Set status and attach evidence for a single claim |
| `veritas export <id>` | Generate Markdown or JSON brief with provenance labels |
| `veritas search "<query>"` | Full-text search across all claims |
| `veritas sources` | List all ingested sources |
| `veritas doctor` | Check environment, GPU, and dependency status |

---

## How Claim Extraction Works

Veritas uses deterministic rules — no AI, no API calls — to identify checkable statements:

1. **Segment Stitching** — Whisper outputs segments with arbitrary boundaries. Veritas merges adjacent segments into windows so complete sentences can be recovered across boundaries.

2. **Sentence Splitting** — The stitched window is split at punctuation boundaries. Fragments shorter than 7 words or 40 characters are rejected. Claims are capped at 240 characters.

3. **Candidate Detection** — A sentence becomes a candidate claim if it contains a signal (numbers, dates, named entities, or assertion verbs) AND has a subject-like anchor (proper noun, pronoun, or number).

4. **Fragment Filtering** — Dangling clauses starting with conjunctions are rejected. YouTube boilerplate is filtered out.

5. **Classification** — Each claim gets confidence language (hedged/definitive/unknown), a category, and a pipe-delimited signal log showing exactly which rules fired.

6. **Deduplication** — Two-layer dedup: SHA256 hash for exact matches (local and global), plus SequenceMatcher (0.85 threshold) for near-duplicates.

---

## Integration: WeThePeople

Veritas is being developed alongside [**WeThePeople**](https://github.com/Obelus-Labs-LLC/WeThePeople), a civic transparency platform. They are separate projects today with a planned integration path:

- **WeThePeople** collects and organizes public political content — congressional hearings, campaign speeches, policy debates
- **Veritas** provides the verification layer — extracting claims and checking them against primary sources
- The integration path: WeThePeople sends politician hearing clips and speech transcripts to Veritas for automated claim extraction and evidence verification, then surfaces the results to citizens

Together, they form a pipeline from raw political speech to verified, evidence-linked claims — with full transparency at every step.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

204 tests across 8 test files. Tests use fixture transcripts — no network calls or GPU required.

---

## Architecture

```
veritas-app/
├── src/veritas/
│   ├── cli.py              # Click CLI interface
│   ├── ingest.py           # Audio download (yt-dlp)
│   ├── transcribe.py       # Speech-to-text (faster-whisper)
│   ├── claim_extract.py    # Deterministic claim extraction
│   ├── assist.py           # Smart routing + evidence orchestration
│   ├── scoring.py          # Rule-based evidence scoring
│   ├── evidence_sources/   # 7 free API integrations
│   │   ├── crossref.py
│   │   ├── arxiv.py
│   │   ├── pubmed.py
│   │   ├── sec_edgar.py
│   │   ├── yfinance_source.py
│   │   ├── fred_source.py
│   │   └── wikipedia_source.py
│   ├── db.py               # SQLite schema + migrations
│   ├── models.py           # Data models
│   ├── config.py           # Constants and paths
│   ├── export.py           # Markdown/JSON brief generation
│   └── search.py           # Full-text claim search
├── tests/                  # 204 tests
├── data/                   # Local data (gitignored)
│   ├── raw/                # Downloaded audio
│   ├── transcripts/        # Whisper output
│   ├── exports/            # Generated briefs
│   └── veritas.sqlite      # Claim database
├── veritas.py              # Convenience runner
├── requirements.txt
└── pyproject.toml
```

### Design Principles

- **No external LLM** — all extraction and scoring is deterministic
- **No paid APIs** — runs entirely on local compute + free public APIs
- **Privacy first** — nothing leaves your machine except structured API queries to public endpoints
- **Unknown is honest** — Veritas never fabricates confidence. No evidence means UNKNOWN
- **Explainability** — every claim logs which rules fired; every evidence suggestion logs its scoring breakdown
- **AUTO vs HUMAN** — exports clearly separate machine suggestions from human verification

---

## License

MIT

---

Built and maintained by [**Obelus Labs LLC**](https://github.com/Obelus-Labs-LLC).

*Veritas — Latin for "truth."*
