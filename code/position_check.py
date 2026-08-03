#!/usr/bin/env python3
"""
position_check.py — 30-second sanity check.

Reads the preprocessing metadata JSONs and reports where, within the
15-slice stack, the annotated slice ends up after sorting by depth.

Usage:
    python position_check.py /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15/metadata

If the distribution is spread across positions 0..14, "no positional
signal" is defensible. If it's clustered at one or two indices, we
soften the wording in the response and move on.
"""
import json, sys
from pathlib import Path
from collections import Counter

if len(sys.argv) < 2:
    print("usage: python position_check.py <metadata_root>")
    print("example: /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15/metadata")
    sys.exit(1)

root = Path(sys.argv[1])
if not root.exists():
    print(f"not found: {root}")
    sys.exit(1)

for split in ["train", "validation", "test"]:
    split_dir = root / split
    if not split_dir.exists():
        continue
    idxs = []
    n_files = 0
    for cls in ["Benign", "Cancer"]:
        cls_dir = split_dir / cls
        if not cls_dir.exists(): continue
        for jf in cls_dir.glob("*.json"):
            with open(jf) as f:
                meta = json.load(f)
            for i in meta.get("annotated_slice_indices", []):
                idxs.append(int(i))
            n_files += 1
    if not idxs:
        continue
    print(f"\n{split}: {n_files} cases, {len(idxs)} annotated-slice positions")
    print(f"  min={min(idxs)}, max={max(idxs)}, mean={sum(idxs)/len(idxs):.2f}")
    print(f"  distribution (index -> count):")
    c = Counter(idxs)
    for i in range(15):
        bar = "#" * c.get(i, 0)
        print(f"    {i:2d} | {bar} ({c.get(i, 0)})")