"""Test the improved categorization with source context."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from veritas.claim_extract import _classify_category, _score_all_categories
from veritas import db

# Test context scoring for various source titles
print("=== Source Title Context Scores ===")
titles = [
    "Apple Q1 FY26 Earnings Call AAPL",
    "Palantir Q4 FY25 Earnings Call PLTR",
    "Nvidia Q4 FY26 Earnings Call NVDA",
    "The Most Misunderstood Concept in Physics",
    "Sean Carroll: General Relativity",
    "BlackRock Is Terrifying.",
    "How The Economic Machine Works by Ray Dalio",
]
for t in titles:
    scores = _score_all_categories(t)
    top = max(scores.items(), key=lambda x: x[1]) if scores else ("none", 0)
    print(f"  {top[0]:12s} (score {top[1]:2d}) | {t[:55]}")

print()

# Test key claims that should benefit from context
print("=== Claims with vs without context ===")
test_cases = [
    ("we are well positioned for the year ahead", "Palantir Q4 FY25 Earnings Call PLTR"),
    ("Adding to that about our charging network expansion", "Live: Tesla Q4 Earnings Call 2025"),
    ("I am pleased that we are collaborating with them", "Alphabet 2025 Q4 Earnings Call"),
    ("neurons fire electrical signals across synapses", "The Most Misunderstood Concept in Physics"),
]
for text, title in test_cases:
    without = _classify_category(text)
    with_ctx = _classify_category(text, source_title=title)
    tag = "  IMPROVED!" if without == "general" and with_ctx != "general" else ""
    print(f"  [{without:10s} -> {with_ctx:10s}]{tag}  \"{text[:50]}...\"")

print()

# Test re-categorization on real sources
print("=== Re-categorization Impact on Real Sources ===")
source_ids = [
    ("3984720e7522", "Apple"),
    ("4d003833aaa2", "Palantir"),
    ("f53d366f4562", "NVIDIA"),
    ("3b394fd44414", "Tesla"),
    ("a69a8ca7e251", "Meta"),
    ("2352a611dfbe", "Microsoft"),
    ("9abd7307e8a1", "Physics"),
    ("cdc7e1008fb0", "Sean Carroll"),
    ("b6ac70f07811", "Alphabet"),
]

total_before = 0
total_changed = 0
total_claims = 0

for sid, name in source_ids:
    s = db.get_source(sid)
    if not s:
        continue
    claims = db.get_claims_for_source(sid)
    before = sum(1 for c in claims if c.category == "general")
    changed = 0
    for c in claims:
        if c.category == "general":
            new = _classify_category(c.text, s.title, s.channel)
            if new != "general":
                changed += 1
    after = before - changed
    total_before += before
    total_changed += changed
    total_claims += len(claims)
    pct_before = before / len(claims) * 100 if claims else 0
    pct_after = after / len(claims) * 100 if claims else 0
    print(f"  {name:12s}: {len(claims):4d} claims | general: {before:4d} ({pct_before:.1f}%) -> {after:4d} ({pct_after:.1f}%)")

print()
print(f"  Total claims checked: {total_claims}")
print(f"  Total would re-categorize: {total_changed}")
print(f"  General before: {total_before} -> {total_before - total_changed}")
