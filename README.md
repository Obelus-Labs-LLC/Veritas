# Veritas

**A deterministic claim extraction and fact-verification engine.**

Veritas extracts claims from audio, video, text, and PDF sources using NLP — no large language models — then verifies them against 20 free structured APIs using rule-based methods. Every verification is transparent, auditable, and traceable to a primary source.

> Zero LLM dependency. Zero hallucination risk. Every claim maps to evidence or stays marked unknown.

---

## What It Does

Veritas takes any audio, video, text, or PDF source and runs it through a fully deterministic pipeline:

1. **Ingest** — downloads audio via yt-dlp, reads text/PDF files, or fetches web articles
2. **Transcribe** — GPU-accelerated speech-to-text using faster-whisper (CTranslate2 / CUDA), or direct text-to-segment conversion for document intake
3. **Extract Claims** — rule-based NLP identifies checkable factual statements from the transcript. No LLM, no prompt engineering — uses sentence boundary detection, named entity recognition, assertion verb patterns, and signal scoring
4. **Categorize** — context-aware keyword classification across 10 categories routes each claim to the most relevant evidence sources. Source metadata (title, channel) influences categorization so claims inherit context from their source
5. **Verify** — smart routing with 13 content-aware signals sends claims to free, structured APIs. Scoring uses token overlap, entity matching, number matching, keyphrase alignment, temporal awareness, and evidence type weighting. Strict guardrails prevent false positives
6. **Cluster** — a knowledge graph fingerprints claims, groups them by category and numeric content, then clusters related claims across sources using Union-Find with Jaccard similarity. Cross-source clusters get consensus scoring

The result: a structured database of claims, each linked to candidate evidence with full scoring transparency, plus a knowledge graph showing how claims relate across sources.

---

## Verification Approach

Veritas takes a fundamentally different approach from LLM-based fact-checkers:

- **Extraction is deterministic** — the same transcript always produces the same claims. No temperature, no sampling, no prompt sensitivity
- **Verification is rule-based** — scoring functions use token overlap, entity matching, exact number matching, and evidence type classification. No embeddings, no semantic similarity
- **Evidence comes from primary sources** — SEC filings, academic papers, government datasets, market data, fact-check organizations. Not web search, not LLM-generated summaries
- **Temporal awareness** — claims with dates are matched against time-relevant evidence; stale data is penalized
- **Unknown is the default** — if the evidence APIs return nothing relevant, the claim stays UNKNOWN. Veritas never guesses

### Auto Status Guardrails

| Status | Conditions |
|--------|-----------|
| **SUPPORTED** | Score >= 65 with strong evidence alignment (keyphrase, entity, or number match) |
| **PARTIAL** | Score >= 40, some evidence alignment |
| **UNKNOWN** | Everything else (the honest default) |

CONTRADICTED is never set automatically — too risky for an automated system. Finance claims have additional guardrails requiring specific financial metric matches, not just entity name overlap.

---

## Evidence Sources

20 free APIs. No API keys required.

| Source | Type | Best For |
|--------|------|----------|
| **SEC EDGAR** | `filing` | Company financials, earnings, 10-K/10-Q/8-K filings |
| **SEC Gov** | `gov` | SEC publications, reports, and regulatory documents |
| **yfinance** | `dataset` | Real-time market data, stock prices, market cap, revenue |
| **FRED** | `dataset` | Macroeconomic indicators — GDP, CPI, unemployment, federal funds rate |
| **BLS** | `gov` | Labor statistics — employment, wages, CPI, PPI |
| **CBO** | `gov` | Congressional Budget Office reports and projections |
| **USASpending** | `gov` | Federal government spending and contract awards |
| **Census** | `gov` | Population, demographics, housing, income statistics |
| **World Bank** | `dataset` | International development indicators across 200+ countries |
| **OpenFDA** | `gov` | Drug safety, adverse events, device recalls |
| **PatentsView** | `dataset` | USPTO patent and invention data |
| **Crossref** | `paper` | Academic papers across all fields (DOI-linked) |
| **arXiv** | `paper` | AI/ML, physics, mathematics, computer science preprints |
| **PubMed** | `paper` | Biomedical and health research (PMID-linked) |
| **Semantic Scholar** | `paper` | AI-curated academic search across all disciplines |
| **Wikipedia** | `secondary` | Named entity context, background reference |
| **Wikidata** | `dataset` | Structured knowledge base — entities, relationships, facts |
| **DuckDuckGo** | `search` | General web search fallback for uncategorized claims |
| **Google Fact Check** | `factcheck` | Verified fact-checks from PolitiFact, Snopes, Full Fact, AFP, Reuters, and IFCN-certified publishers |
| **Local Datasets** | `dataset` | Curated CSV datasets (FRED historical, corporate financials) for offline matching |

### Smart Routing

Smart routing uses 13 content-aware signals to optimize source ordering per claim:

- **Company mentions** boost yfinance + SEC EDGAR
- **Academic language** boosts arXiv + Crossref + Semantic Scholar
- **Health/clinical terms** boost PubMed + OpenFDA
- **Financial metrics** boost yfinance + SEC EDGAR + FRED
- **Drug/pharmaceutical terms** boost OpenFDA
- **Labor/employment terms** boost BLS
- **Budget/spending terms** boost CBO + USASpending
- **Demographics terms** boost Census
- **International indicators** boost World Bank
- **Patent/invention terms** boost PatentsView

---

## Document Ingestion

Veritas isn't limited to audio/video. It can ingest text documents directly:

| Intake Path | Description |
|-------------|-------------|
| **Plain text** | `.txt` files — read and segmented automatically |
| **PDF** | Requires PyMuPDF or pdfplumber — text extracted and segmented |
| **Web URL** | Article text extracted from HTML (strips nav, scripts, ads) |
| **Raw text** | Inline text string via CLI or API |

Text is converted to pseudo-segments (with synthetic timestamps) so the existing claim extraction pipeline works unchanged. No audio, no transcription — text goes straight to segments.

---

## Knowledge Graph

Veritas clusters related claims across sources to find consensus and contradiction:

- **Fingerprinting** — each claim is tokenized and numeric values extracted for comparison
- **Blocking** — claims are grouped by category and shared numbers to reduce comparison space
- **Clustering** — Union-Find with Jaccard similarity (0.40 threshold) groups related claims
- **Cross-source only** — clusters require claims from different sources (same-source repeats are filtered)
- **Consensus scoring** — clusters with 2+ sources get confidence boosts based on agreement

---

## Cross-Source Intelligence

Veritas tracks how claims move across sources:

- **Spread Analysis** — identifies the same claim appearing across multiple videos or channels, scored by global content hash and fuzzy matching
- **Timeline Tracking** — maps when claims first appear and how they propagate
- **Top Claims Ranking** — surfaces the most-repeated claims across your entire corpus, ranked by cross-source frequency
- **Contradiction Detection** — flags cases where sources make conflicting factual assertions about the same topic

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Transcription | faster-whisper (CTranslate2, CUDA-accelerated) |
| Audio download | yt-dlp |
| Database | SQLite (single-file, zero-config) |
| CLI | Click + Rich |
| NLP | Rule-based (no external NLP libraries) |
| HTTP client | httpx (async-capable, 20 free sources) |
| GPU support | NVIDIA CUDA 12 via pip (nvidia-cublas-cu12, nvidia-cudnn-cu12) |
| Testing | pytest |

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

# Run the pipeline on a YouTube video
python veritas.py ingest "https://www.youtube.com/watch?v=EXAMPLE"
python veritas.py transcribe <source_id>
python veritas.py claims <source_id>
python veritas.py assist <source_id>

# Or ingest text/PDF directly (no audio, no transcription)
python veritas.py ingest-text path/to/document.pdf
python veritas.py claims <source_id>
python veritas.py assist <source_id>

# Or ingest a web article
python veritas.py ingest-url "https://example.com/article"
python veritas.py claims <source_id>
python veritas.py assist <source_id>

# Build the knowledge graph
python veritas.py build-graph

# Check results
python veritas.py sources
python veritas.py queue
python veritas.py inspect-verified --status supported --verbose
python veritas.py clusters
python veritas.py export <source_id> --format md
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `veritas ingest <url>` | Download audio, save metadata, register source |
| `veritas ingest-text <path>` | Ingest a text or PDF file directly (no audio) |
| `veritas ingest-url <url>` | Ingest a web article URL (extracts article text) |
| `veritas transcribe <id>` | Transcribe audio with faster-whisper (GPU) |
| `veritas claims <id>` | Extract candidate claims (deterministic, rule-based) |
| `veritas assist <id>` | Auto-discover evidence from 20 free APIs |
| `veritas build-graph` | Build the knowledge graph (fingerprint, cluster, consensus) |
| `veritas clusters` | Show top claim clusters from the knowledge graph |
| `veritas cluster <id>` | Show detailed view of a single cluster and its members |
| `veritas queue` | Show claims needing review, sorted by priority |
| `veritas review <id>` | Interactively review and verify claims |
| `veritas verify <claim_id>` | Set status and attach evidence for a single claim |
| `veritas inspect-verified` | Inspect auto-verified claims with evidence and signals |
| `veritas export <id>` | Generate Markdown or JSON brief with provenance labels |
| `veritas search "<query>"` | Full-text search across all claims |
| `veritas sources` | List all ingested sources with verification metrics |
| `veritas spread <claim_hash>` | Show where a claim appears across sources |
| `veritas timeline <claim_hash>` | Chronological propagation of a claim |
| `veritas top-claims` | Most-repeated claims across all sources |
| `veritas doctor` | Check environment, GPU, and dependency status |

---

## How Claim Extraction Works

Veritas uses deterministic rules — no AI, no API calls — to identify checkable statements:

1. **Segment Stitching** — Whisper outputs segments with arbitrary boundaries. Veritas merges adjacent segments into windows so complete sentences can be recovered across boundaries.

2. **Sentence Splitting** — The stitched window is split at punctuation boundaries. Fragments shorter than 7 words or 40 characters are rejected. Claims are capped at 240 characters.

3. **Candidate Detection** — A sentence becomes a candidate claim if it contains a signal (numbers, dates, named entities, or assertion verbs) AND has a subject-like anchor (proper noun, pronoun, or number).

4. **Fragment Filtering** — Dangling clauses starting with conjunctions are rejected. YouTube boilerplate is filtered out.

5. **Classification** — Each claim gets confidence language (hedged/definitive/unknown), a category (10 categories), and a pipe-delimited signal log showing exactly which rules fired.

6. **Deduplication** — Two-layer dedup: SHA256 hash for exact matches (local and global), plus SequenceMatcher (0.85 threshold) for near-duplicates.

---

## Claim Categories

| Category | Routes To |
|----------|-----------|
| **finance** | yfinance, SEC EDGAR, FRED, BLS, CBO, USASpending, Local Datasets |
| **health** | PubMed, OpenFDA, Google Fact Check |
| **science** | arXiv, Crossref, PubMed, Semantic Scholar |
| **tech** | arXiv, Crossref, PatentsView, Semantic Scholar |
| **politics** | Google Fact Check, CBO, USASpending, Wikipedia |
| **military** | Google Fact Check, USASpending, Wikipedia |
| **education** | Census, World Bank, Crossref |
| **energy_climate** | World Bank, Crossref, arXiv |
| **labor** | BLS, FRED, Census |
| **general** | DuckDuckGo, Google Fact Check, Wikipedia, Wikidata, Crossref, arXiv, BLS, Census |

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

Tests use fixture transcripts and mocked APIs — no network calls or GPU required.

---

## Architecture

```
veritas-app/
├── src/veritas/
│   ├── cli.py              # Click CLI (21 commands)
│   ├── ingest.py           # Audio download (yt-dlp)
│   ├── ingest_text.py      # Text/PDF/URL document ingestion
│   ├── transcribe.py       # Speech-to-text (faster-whisper)
│   ├── claim_extract.py    # Deterministic claim extraction (10 categories)
│   ├── assist.py           # Smart routing + evidence orchestration (13 signals)
│   ├── scoring.py          # Rule-based evidence scoring (0-100)
│   ├── knowledge_graph.py  # Fingerprinting, clustering, consensus scoring
│   ├── evidence_sources/   # 20 free API integrations
│   │   ├── crossref.py
│   │   ├── arxiv.py
│   │   ├── pubmed.py
│   │   ├── semantic_scholar.py
│   │   ├── sec_edgar.py
│   │   ├── sec_gov.py
│   │   ├── yfinance_source.py
│   │   ├── fred_source.py
│   │   ├── wikipedia_source.py
│   │   ├── wikidata.py
│   │   ├── duckduckgo.py
│   │   ├── google_factcheck.py
│   │   ├── openfda.py
│   │   ├── bls.py
│   │   ├── cbo.py
│   │   ├── usaspending.py
│   │   ├── census.py
│   │   ├── worldbank.py
│   │   ├── patentsview.py
│   │   └── local_datasets.py
│   ├── db.py               # SQLite schema + migrations
│   ├── models.py           # Data models
│   ├── config.py           # Constants and paths
│   ├── export.py           # Markdown/JSON brief generation
│   └── search.py           # Full-text claim search
├── tests/                  # pytest suite
├── data/                   # Local data (gitignored)
│   ├── raw/                # Downloaded audio
│   ├── transcripts/        # Whisper output
│   ├── datasets/           # Curated CSV datasets
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
- **Temporal awareness** — claims with dates are scored against time-relevant evidence

---

## License

MIT

---

Built and maintained by [**Obelus Labs LLC**](https://github.com/Obelus-Labs-LLC).

*Veritas — Latin for "truth."*
