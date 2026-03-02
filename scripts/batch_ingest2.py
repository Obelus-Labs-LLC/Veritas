"""Batch ingest + transcribe + extract claims from a list of URLs (batch 2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.ingest import ingest
from veritas.transcribe import transcribe
from veritas.claim_extract import extract_claims
from veritas import db

URLS = [
    "https://www.youtube.com/watch?v=shmaTeJvGpk",
    "https://www.youtube.com/watch?v=TQuxVz52w2w",
    "https://www.youtube.com/watch?v=cWE8a_Ge4aU",
    "https://www.youtube.com/watch?v=kK9_3sXFuYw",
    "https://www.youtube.com/watch?v=rT7DMb3ZucU",
    "https://www.youtube.com/watch?v=0S35WwkQ6gQ",
    "https://www.youtube.com/watch?v=MPTNHrq_4LU",
    "https://www.youtube.com/watch?v=KZX_c6zfJ2w",
    "https://www.youtube.com/watch?v=yMKlkJNmWTI",
    "https://www.youtube.com/watch?v=tdv7r2JSokI",
]


def main():
    total = len(URLS)
    results = []

    for i, url in enumerate(URLS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {url}")
        print(f"{'='*60}")

        # Ingest
        try:
            source = ingest(url)
            name = source.title.encode("ascii", errors="replace").decode()[:55]
            print(f"  Ingested: {source.id} | {name} | {source.duration_seconds:.0f}s")
        except Exception as e:
            print(f"  INGEST ERROR: {e}")
            results.append((url, "INGEST_ERROR", str(e)))
            continue

        # Transcribe
        try:
            meta, _ = transcribe(source.id, model_size="small", device="cuda")
            print(f"  Transcribed: {meta.segment_count} segments")
        except Exception as e:
            print(f"  TRANSCRIBE ERROR: {e}")
            results.append((url, source.id, f"TRANSCRIBE_ERROR: {e}"))
            continue

        # Extract claims
        try:
            claims = extract_claims(source.id)
            print(f"  Claims: {len(claims)}")
            results.append((url, source.id, f"{len(claims)} claims"))
        except Exception as e:
            print(f"  EXTRACT ERROR: {e}")
            results.append((url, source.id, f"EXTRACT_ERROR: {e}"))

    # Summary
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"{'='*60}")
    for url, sid, status in results:
        print(f"  {sid:14s} | {status}")

    # Overall stats
    sources = db.list_sources()
    total_claims = sum(len(db.get_claims_for_source(s.id)) for s in sources)
    print(f"\nTotal sources: {len(sources)}")
    print(f"Total claims: {total_claims}")


if __name__ == "__main__":
    main()
