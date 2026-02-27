"""Local dataset evidence source — verify claims against user-provided data files.

Loads XLSX and CSV files from data/datasets/ directory and matches claim text
against column values. Follows the BLS/Census pattern: keyword matching first,
then precise data lookup.

Use case: SEC budget documents, statistical data files, government spreadsheets.
No API calls — all local. Zero latency.
"""

from __future__ import annotations
import csv
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from .. import config

# Cache of loaded datasets: {file_hash: {filename, headers, rows, text_index}}
_DATASET_CACHE: Dict[str, Dict[str, Any]] = {}

# Directory for user-provided data files
_DATASETS_DIR: Optional[Path] = None


def _get_datasets_dir() -> Path:
    """Lazy-init datasets directory under data/datasets/."""
    global _DATASETS_DIR
    if _DATASETS_DIR is None:
        _DATASETS_DIR = config.DATA_DIR / "datasets"
        _DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    return _DATASETS_DIR


def _file_hash(path: Path) -> str:
    """Quick hash based on path + mtime + size for cache invalidation."""
    stat = path.stat()
    key = f"{path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _load_csv(path: Path) -> Dict[str, Any]:
    """Load a CSV file into a searchable structure."""
    rows = []
    headers = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    headers = [h.strip() for h in row]
                    continue
                if i > 10000:  # safety cap
                    break
                rows.append(row)
    except Exception:
        return {"filename": path.name, "headers": [], "rows": [], "text_index": ""}

    # Build text index: all cell values concatenated for fast substring search
    all_text = " ".join(headers)
    for row in rows:
        all_text += " " + " ".join(str(c).strip() for c in row)

    return {
        "filename": path.name,
        "headers": headers,
        "rows": rows,
        "text_index": all_text.lower(),
        "row_count": len(rows),
        "path": str(path),
    }


def _load_xlsx(path: Path) -> Dict[str, Any]:
    """Load an XLSX file into a searchable structure."""
    try:
        import openpyxl
    except ImportError:
        return {"filename": path.name, "headers": [], "rows": [], "text_index": ""}

    rows = []
    headers = []
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        if ws is None:
            return {"filename": path.name, "headers": [], "rows": [], "text_index": ""}

        for i, row in enumerate(ws.iter_rows(values_only=True)):
            str_row = [str(c).strip() if c is not None else "" for c in row]
            if i == 0:
                headers = str_row
                continue
            if i > 10000:
                break
            rows.append(str_row)
        wb.close()
    except Exception:
        return {"filename": path.name, "headers": [], "rows": [], "text_index": ""}

    all_text = " ".join(headers)
    for row in rows:
        all_text += " " + " ".join(row)

    return {
        "filename": path.name,
        "headers": headers,
        "rows": rows,
        "text_index": all_text.lower(),
        "row_count": len(rows),
        "path": str(path),
    }


def _load_all_datasets() -> List[Dict[str, Any]]:
    """Load (or return cached) all datasets from data/datasets/."""
    datasets_dir = _get_datasets_dir()
    datasets = []

    for path in sorted(datasets_dir.iterdir()):
        if path.suffix.lower() == ".csv":
            fhash = _file_hash(path)
            if fhash not in _DATASET_CACHE:
                _DATASET_CACHE[fhash] = _load_csv(path)
            datasets.append(_DATASET_CACHE[fhash])
        elif path.suffix.lower() == ".xlsx":
            fhash = _file_hash(path)
            if fhash not in _DATASET_CACHE:
                _DATASET_CACHE[fhash] = _load_xlsx(path)
            datasets.append(_DATASET_CACHE[fhash])

    return datasets


def _extract_numbers(text: str) -> set:
    """Extract significant numbers from text."""
    nums = set(re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', text))
    # Normalize: remove commas
    return {n.replace(",", "") for n in nums if len(n.replace(",", "").replace(".", "")) >= 2}


def _find_matching_rows(
    dataset: Dict[str, Any],
    claim_text: str,
    claim_numbers: set,
) -> List[Dict[str, Any]]:
    """Find rows in a dataset that match claim numbers or key terms."""
    matches = []
    headers = dataset.get("headers", [])
    rows = dataset.get("rows", [])

    if not rows or not headers:
        return []

    claim_lower = claim_text.lower()

    # Extract key terms from claim (skip very common words)
    _STOP = frozenset([
        "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
        "be", "been", "to", "of", "in", "for", "on", "at", "by", "with",
        "from", "as", "and", "but", "or", "not", "that", "this", "it",
    ])
    claim_words = set()
    for w in claim_text.split():
        w_clean = w.strip(".,!?;:\"'()[]$%").lower()
        if w_clean and w_clean not in _STOP and len(w_clean) > 2:
            claim_words.add(w_clean)

    for row_idx, row in enumerate(rows):
        row_text = " ".join(str(c) for c in row).lower()
        row_numbers = _extract_numbers(" ".join(str(c) for c in row))

        # Score this row
        score = 0

        # Exact number matches (strongest signal)
        num_matches = claim_numbers & row_numbers
        if num_matches:
            score += len(num_matches) * 20

        # Term overlap
        for word in claim_words:
            if word in row_text:
                score += 3

        if score >= 10:  # minimum threshold
            # Build a snippet from this row
            snippet_parts = []
            for h, v in zip(headers, row):
                v_str = str(v).strip()
                if v_str and v_str.lower() != "none":
                    snippet_parts.append(f"{h}: {v_str}")
            snippet = " | ".join(snippet_parts)

            matches.append({
                "row_idx": row_idx,
                "score": score,
                "snippet": snippet[:2000],
                "num_matches": num_matches,
            })

    # Sort by score descending
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:5]


def search_local_datasets(claim_text: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Search local dataset files for data matching a claim.

    Standard evidence source signature.
    Returns list of dicts with keys: url, title, source_name, evidence_type, snippet.
    """
    datasets = _load_all_datasets()
    if not datasets:
        return []

    claim_numbers = _extract_numbers(claim_text)
    claim_lower = claim_text.lower()

    results = []

    for ds in datasets:
        # Quick pre-filter: does this dataset's text index contain ANY claim numbers?
        text_idx = ds.get("text_index", "")
        has_number_hit = False
        for num in claim_numbers:
            if num in text_idx:
                has_number_hit = True
                break

        # Also check for key term overlap (at least 2 non-trivial terms)
        claim_words = [w.strip(".,!?;:\"'()[]$%").lower() for w in claim_text.split()]
        claim_words = [w for w in claim_words if len(w) > 3]
        term_hits = sum(1 for w in claim_words if w in text_idx)

        if not has_number_hit and term_hits < 2:
            continue  # skip datasets with no relevance

        # Deep search: find matching rows
        row_matches = _find_matching_rows(ds, claim_text, claim_numbers)

        for match in row_matches:
            filename = ds.get("filename", "unknown")
            row_count = ds.get("row_count", 0)

            title = f"Local Dataset: {filename} ({row_count} rows)"
            snippet = match["snippet"]

            if match["num_matches"]:
                nums_str = ", ".join(sorted(match["num_matches"])[:3])
                snippet = f"[Exact number match: {nums_str}] {snippet}"

            results.append({
                "url": f"file://{ds.get('path', '')}",
                "title": title[:200],
                "source_name": "local_dataset",
                "evidence_type": "dataset",
                "snippet": snippet[:4000],
            })

    # Sort by snippet length as proxy for match quality (more data = better)
    results.sort(key=lambda r: len(r.get("snippet", "")), reverse=True)
    return results[:max_results]
