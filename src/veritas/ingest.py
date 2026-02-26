"""Download audio from a URL via yt-dlp and register a Source in the DB."""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

from .models import Source, new_id
from .paths import source_raw_dir
from . import db


def ingest(url: str) -> Source:
    """Download best audio from *url*, create a Source row, return it."""
    source_id = new_id()
    out_dir = source_raw_dir(source_id)

    # yt-dlp: extract audio as m4a (or best fallback), write info-json
    output_template = str(out_dir / "audio.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "-x",                          # extract audio
        "--audio-format", "m4a",       # prefer m4a
        "--audio-quality", "0",        # best quality
        "--write-info-json",
        "--no-write-playlist-metafiles",
        "-o", output_template,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed (exit {result.returncode}):\n{result.stderr}")

    # Find the downloaded audio file
    audio_files = [f for f in out_dir.iterdir() if f.suffix in (".m4a", ".mp3", ".opus", ".wav", ".webm") and "info" not in f.name]
    if not audio_files:
        raise FileNotFoundError(f"No audio file found in {out_dir} after yt-dlp download.")
    audio_path = audio_files[0]

    # Parse info json for metadata
    info_files = list(out_dir.glob("*.info.json"))
    title = ""
    channel = ""
    upload_date = ""
    duration = 0.0
    if info_files:
        with open(info_files[0], "r", encoding="utf-8") as fh:
            info = json.load(fh)
        title = info.get("title", "")
        channel = info.get("channel", info.get("uploader", ""))
        upload_date = info.get("upload_date", "")
        duration = float(info.get("duration", 0) or 0)

    source = Source(
        id=source_id,
        url=url,
        title=title,
        channel=channel,
        upload_date=upload_date,
        duration_seconds=duration,
        local_audio_path=str(audio_path),
    )
    db.insert_source(source)
    return source
