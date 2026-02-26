"""Global configuration constants for Veritas."""

from pathlib import Path

# Project root is two levels up from this file (src/veritas/config.py -> veritas-app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "veritas.sqlite"

# Whisper defaults
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"

# Claim extraction
ASSERTION_VERBS = frozenset([
    "is", "are", "was", "were", "has", "have", "had",
    "shows", "show", "confirm", "confirms", "confirmed",
    "found", "reveals", "reveal", "means", "meant",
    "will", "causes", "cause", "caused",
    "leads", "led", "announced", "released",
    "proved", "proves", "demonstrates", "established",
])

HEDGE_WORDS = frozenset([
    "might", "may", "could", "possibly", "likely",
    "appears", "suggests", "suggest", "perhaps",
    "probably", "seemingly", "reportedly", "allegedly",
])

DEFINITIVE_WORDS = frozenset([
    "confirms", "confirm", "confirmed",
    "proves", "prove", "proved",
    "is", "are", "will", "has", "have",
    "definitely", "certainly", "absolutely",
    "establishes", "demonstrates",
])

# Dedup similarity threshold (0-1, higher = stricter)
DEDUP_THRESHOLD = 0.85

# Export defaults
DEFAULT_MAX_QUOTES = 10
