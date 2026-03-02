"""Test re-categorization with expanded science terms."""
import sys
sys.path.insert(0, "src")
from veritas.claim_extract import _classify_category, _score_all_categories
from veritas import db

# Test with the galaxy source title
title = "Why Galaxies Are the Universe's Impossible Walls | Brian Cox."
scores = _score_all_categories(title)
print(f"Galaxy title scores: {scores}")

# Test some claims from that source with context
test_claims = [
    "these galaxies are moving away from us",
    "the universe is expanding at an accelerating rate",
    "dark matter makes up about 27 percent of the universe",
    "light takes millions of years to reach us from distant galaxies",
    "gravity holds the stars together in a galaxy",
]
print()
for c in test_claims:
    without = _classify_category(c)
    with_ctx = _classify_category(c, source_title=title)
    tag = "  IMPROVED!" if without == "general" and with_ctx != "general" else ""
    print(f"  [{without:10s} -> {with_ctx:10s}]{tag}  \"{c[:55]}\"")

# Test overall impact
print()
test_sources = [
    ("30ba5680bf33", "Galaxy/Brian Cox"),
    ("9abd7307e8a1", "Physics"),
    ("cdc7e1008fb0", "Sean Carroll"),
    ("f70ac313be1c", "Wormhole/Science"),
    ("1250b305a66c", "Theory of Everything"),
    ("3984720e7522", "Apple"),
    ("4d003833aaa2", "Palantir"),
]
print("=== Re-categorization Impact ===")
total_before = 0
total_changed = 0
total_claims = 0
for sid, name in test_sources:
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
    total = len(claims)
    total_before += before
    total_changed += changed
    total_claims += total
    pct_before = before / total * 100 if total else 0
    pct_after = after / total * 100 if total else 0
    print(f"  {name:20s}: {total:4d} claims | general: {before:4d} ({pct_before:.0f}%) -> {after:4d} ({pct_after:.0f}%)")

print()
print(f"  Total: {total_claims} claims | general: {total_before} -> {total_before - total_changed}")
print(f"  Would re-categorize: {total_changed}")
