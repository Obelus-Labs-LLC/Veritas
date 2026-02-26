"""Transcribe audio using faster-whisper and persist results."""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import List

from .config import DEFAULT_WHISPER_MODEL, DEFAULT_DEVICE, DEFAULT_COMPUTE_TYPE
from .models import TranscriptMeta, Segment
from .paths import source_transcript_dir
from . import db


def _load_nvidia_dll_paths() -> None:
    """Add pip-installed nvidia-* DLL directories to the DLL search path (Windows).

    This allows CTranslate2 to find cublas64_12.dll, cudnn*.dll, etc. that ship
    in the nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip packages without needing
    a system-wide CUDA Toolkit install.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return  # nvidia packages not installed — nothing to do
    nvidia_root = Path(nvidia.__path__[0])
    for subdir in nvidia_root.iterdir():
        bin_dir = subdir / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            # Also prepend to PATH so CTranslate2's native code can find the DLLs
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def transcribe(
    source_id: str,
    model_size: str = DEFAULT_WHISPER_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str | None = None,
) -> tuple[TranscriptMeta, List[Segment]]:
    """Run faster-whisper on the audio for *source_id*.

    Returns the metadata row and a list of Segment objects.
    """
    _load_nvidia_dll_paths()
    from faster_whisper import WhisperModel  # defer import so CLI loads fast

    source = db.get_source(source_id)
    if source is None:
        raise ValueError(f"Source '{source_id}' not found in database.")

    audio_path = Path(source.local_audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Resolve compute type
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    # Try requested device, fallback to cpu
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception:
        if device != "cpu":
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            device = "cpu"
        else:
            raise

    segments_iter, info = model.transcribe(str(audio_path), beam_size=5)

    segments: List[Segment] = []
    for seg in segments_iter:
        segments.append(Segment(start=round(seg.start, 3), end=round(seg.end, 3), text=seg.text.strip()))

    # Write transcript JSON to disk
    out_dir = source_transcript_dir(source_id)
    transcript_path = out_dir / "transcript.json"
    payload = {
        "source_id": source_id,
        "engine": "faster-whisper",
        "model": model_size,
        "device": device,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in segments],
    }
    with open(transcript_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    meta = TranscriptMeta(
        source_id=source_id,
        engine="faster-whisper",
        language=info.language,
        segment_count=len(segments),
        transcript_path=str(transcript_path),
    )
    db.upsert_transcript(meta)

    return meta, segments
