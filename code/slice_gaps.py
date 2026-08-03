#!/usr/bin/env python3
"""
slice_gaps.py  ->  depth-continuity statistics for the selected slices.

Checks whether gradient-based slice selection preserves depth order
or produces irregular gaps. Two separate claims are needed:

  1. ORDER: selected indices are returned sorted, so the 15-slice sequence
     preserves anatomical depth ordering. This is verified directly here.
  2. SPACING: the gaps between consecutive selected slices are NOT uniform,
     because selection is by gradient rank. The distribution is what the
     should be reported, rather than claiming regular spacing.

READ-ONLY, CPU, seconds. Reads only the preprocessing metadata JSONs.

If the metadata does not record the original slice indices, the script says so
and prints the keys it did find, rather than guessing.

USAGE
    python slice_gaps.py
    python slice_gaps.py --split train
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

DATA_ROOT = '/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15'

# Keys that might hold the list of selected original slice indices.
CANDIDATES = ['selected_slices', 'slice_indices', 'original_slices',
              'chosen_slices', 'slices', 'kept_slices', 'slice_map']


def find_indices(meta):
    for k in CANDIDATES:
        v = meta.get(k)
        if isinstance(v, list) and len(v) > 1 and all(isinstance(i, (int, float)) for i in v):
            return k, [int(i) for i in v]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--classes', nargs='+', default=['Benign', 'Cancer', 'Normal'])
    args = ap.parse_args()

    files = []
    for cls in args.classes:
        d = Path(args.data_root) / 'metadata' / args.split / cls
        if d.exists():
            files.extend(sorted(d.glob('*.json')))
    if not files:
        print(f"No metadata JSONs under {args.data_root}/metadata/{args.split}/")
        return

    print("=" * 70)
    print(f"INTER-SLICE GAP DISTRIBUTION  ({args.split}, {len(files)} volumes)")
    print("=" * 70)

    all_gaps, n_ok, n_sorted, depths, key_used = [], 0, 0, [], None
    sample_keys = None

    for fp in files:
        try:
            with open(fp) as f:
                meta = json.load(f)
        except Exception:
            continue
        if sample_keys is None:
            sample_keys = sorted(meta.keys())
        k, idx = find_indices(meta)
        if idx is None:
            continue
        key_used = k
        n_ok += 1
        if idx == sorted(idx):
            n_sorted += 1
        g = np.diff(sorted(idx))
        all_gaps.extend(g.tolist())
        if 'num_slices_original' in meta:
            depths.append(int(meta['num_slices_original']))
        elif 'original_depth' in meta:
            depths.append(int(meta['original_depth']))

    if n_ok == 0:
        print("\n  The metadata does not record original slice indices.")
        print(f"  Keys present in a sample file: {sample_keys}")
        print("\n  Without them the gap distribution cannot be computed from metadata.")
        print("  It would have to come from the preprocessing script instead, by")
        print("  logging the selected indices at the point of selection.")
        return

    g = np.array(all_gaps)
    print(f"\n  metadata key used      : '{key_used}'")
    print(f"  volumes parsed         : {n_ok}/{len(files)}")
    print(f"  ascending (depth order): {n_sorted}/{n_ok}"
          f"{'   <- order preserved in all volumes' if n_sorted == n_ok else ''}")
    if depths:
        print(f"  original depth         : mean {np.mean(depths):.1f}, "
              f"range {min(depths)}-{max(depths)}")

    print(f"\n  gaps between consecutive selected slices (n = {len(g):,})")
    print(f"    mean   {g.mean():.2f}    SD {g.std():.2f}")
    print(f"    median {np.median(g):.1f}     min {g.min()}   max {g.max()}")
    for q in (25, 75, 90, 99):
        print(f"    p{q:<3d}   {np.percentile(g, q):.1f}")

    print("\n  gap histogram (top 12):")
    for val, cnt in Counter(g.tolist()).most_common(12):
        bar = '#' * max(1, int(40 * cnt / len(g)))
        print(f"    gap {val:3d}: {cnt:6d} ({100*cnt/len(g):5.1f}%) {bar}")

    frac1 = float((g == 1).mean())
    print(f"\n  adjacent (gap = 1): {100*frac1:.1f}% of consecutive pairs")
    print("\n  Selection preserves depth ORDER but not uniform")
    print("  SPACING. State both. Claiming regular spacing would be wrong, and")
    print("  irregular gaps are quantified above.")
    print("=" * 70)


if __name__ == '__main__':
    main()
