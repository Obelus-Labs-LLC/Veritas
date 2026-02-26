#!/usr/bin/env python
"""Convenience runner — allows `python veritas.py <command>` from project root."""
import sys
from pathlib import Path

# Add src/ to Python path so veritas package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from veritas.cli import main

if __name__ == "__main__":
    main()
