"""Batch transcribe + extract claims for all untranscribed sources."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.transcribe import transcribe
from veritas.claim_extract import extract_claims
from veritas import db


def main():
    sources = db.list_sources()
    todo = [
        s for s in sources
        if db.get_transcript(s.id) is None and s.local_audio_path
    ]
    print(f"Sources needing transcription: {len(todo)}")

    for i, s in enumerate(todo, 1):
        name = s.title.encode("ascii", errors="replace").decode()[:50]
        print(f"\n[{i}/{len(todo)}] {name}")

        # Transcribe
        print("  Transcribing...")
        try:
            meta, _ = transcribe(s.id, model_size="small", device="cuda")
            print(f"  -> {meta.segment_count} segments")
        except Exception as e:
            print(f"  -> TRANSCRIBE ERROR: {e}")
            continue

        # Extract claims
        print("  Extracting claims...")
        try:
            claims = extract_claims(s.id)
            print(f"  -> {len(claims)} claims")
        except Exception as e:
            print(f"  -> EXTRACT ERROR: {e}")

    # Summary
    print("\n=== SUMMARY ===")
    all_sources = db.list_sources()
    total_claims = 0
    for s in all_sources:
        c = db.get_claims_for_source(s.id)
        total_claims += len(c)
    print(f"Total sources: {len(all_sources)}")
    print(f"Total claims: {total_claims}")


if __name__ == "__main__":
    main()
