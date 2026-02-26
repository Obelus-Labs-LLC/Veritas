"""Diagnostics command — check that all runtime dependencies are available."""

from __future__ import annotations
import sys
import shutil
from typing import List, Tuple


def _load_nvidia_dll_paths() -> None:
    """Register pip-installed nvidia DLL directories on Windows."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
        import os
        from pathlib import Path
        nvidia_root = Path(nvidia.__path__[0])
        for subdir in nvidia_root.iterdir():
            bin_dir = subdir / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass


def run_checks() -> List[Tuple[str, bool, str]]:
    """Return a list of (check_name, passed, detail) tuples."""
    _load_nvidia_dll_paths()
    results: List[Tuple[str, bool, str]] = []

    # 1. Python version
    v = sys.version
    ok = sys.version_info >= (3, 9)
    results.append(("Python version", ok, v if ok else f"{v} — Python >=3.9 required"))

    # 2. yt-dlp importable
    try:
        import yt_dlp
        results.append(("yt-dlp import", True, f"version {yt_dlp.version.__version__}"))
    except ImportError:
        results.append(("yt-dlp import", False, "NOT FOUND — pip install yt-dlp"))

    # 3. yt-dlp on PATH
    on_path = shutil.which("yt-dlp") is not None
    results.append(("yt-dlp on PATH", on_path, shutil.which("yt-dlp") or "NOT FOUND"))

    # 4. faster-whisper import
    try:
        from faster_whisper import WhisperModel
        results.append(("faster-whisper import", True, "OK"))
    except ImportError:
        results.append(("faster-whisper import", False, "NOT FOUND — pip install faster-whisper"))

    # 5. CUDA available (faster-whisper / CTranslate2)
    try:
        from faster_whisper import WhisperModel
        # Try loading tiny model on CUDA to verify GPU works
        _m = WhisperModel("tiny", device="cuda", compute_type="float16")
        results.append(("faster-whisper CUDA", True, "GPU acceleration available"))
        del _m
    except Exception as exc:
        results.append(("faster-whisper CUDA", False, f"Not available — {exc}"))

    # 6. torch + CUDA (informational)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        detail = f"torch {torch.__version__}"
        if cuda_ok:
            detail += f" — CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}"
        else:
            detail += " — CUDA NOT available (CPU-only torch installed)"
        results.append(("torch CUDA", cuda_ok, detail))
    except ImportError:
        results.append(("torch CUDA", False, "torch not installed (optional for Veritas)"))

    # 7. SQLite
    try:
        import sqlite3
        results.append(("SQLite", True, f"version {sqlite3.sqlite_version}"))
    except Exception as exc:
        results.append(("SQLite", False, str(exc)))

    return results
