#!/usr/bin/env python3
"""
position_leakage_test.py  ->  tests whether detection depends on the sequence
position of the annotated slice.

THE CONCERN, STATED PRECISELY
    position_check.py shows the annotated slice lands at sequence position 0
    in most volumes (81 of 133 test cases). Mamba is a sequence model, so
    position 0 is where its recurrence initialises. If the model learned
    "the first element is the important one", that is a shortcut, and 's
    leakage concern is real.

    Position 0 being common is NOT itself evidence of leakage. It is a
    consequence of the selector: annotated slices are seeded first, gradient
    fills the remaining slots, and the result is sorted ascending. If gradient
    scoring favours deeper slices, the annotated slice ends up lowest-indexed.

WHAT THIS SCRIPT ACTUALLY TESTS
    If the model exploits position, detection should be markedly better on
    volumes where the annotated slice sits at position 0 than on volumes where
    it sits elsewhere. If the two groups perform comparably, the model is
    finding lesions by appearance, not by position, and the shortcut is not
    being used.

    It also reports classification separately, since a position shortcut would
    plausibly help detection more than benign-versus-cancer.

    Nothing here needs a GPU. It joins metadata already on disk to per-case
    dumps already computed.

READING THE RESULT
    similar detection across groups  ->  no evidence of a position shortcut;
                                        summary table
    much better at position 0        ->  the shortcut is real, and the
                                        pure_grad variant becomes the number
                                        that must be reported

USAGE
    python position_leakage_test.py \
      --meta_root /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15/metadata/validation \
      --dump 'dumps_val_all_locked_20260708/per_case_BASELINE_(v5.2).jsonl'

    python position_leakage_test.py \
      --meta_root /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15/metadata/test \
      --dump 'dumps_test_all_locked_20260708/per_case_BASELINE_(v5.2).jsonl'
"""

import json
import argparse
from pathlib import Path

import numpy as np


def load_positions(meta_root):
    """case_id -> sorted list of annotated slice positions within the stack."""
    out = {}
    for p in Path(meta_root).rglob('*.json'):
        try:
            m = json.load(open(p))
        except Exception:
            continue
        idx = m.get('annotated_slice_indices')
        if idx is None:
            continue
        out[p.stem] = sorted(int(i) for i in idx)
    return out


def load_dump(path):
    rows = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[r['case_id']] = r
    return rows


def summarize(tag, group):
    if not group:
        print(f'  {tag:<34} (none)')
        return None
    n_gt = sum(r['n_gt'] for r in group)
    n_match = sum(r['n_matched'] for r in group)
    n_fp = sum(r['n_fp'] for r in group)
    y = np.array([1 if r['gt_label'] == 'Cancer' else 0 for r in group])
    p = np.array([float(r['cancer_prob']) for r in group])
    auc = float('nan')
    if len(np.unique(y)) > 1:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
    det = n_match / n_gt if n_gt else float('nan')
    a = 'n/a' if np.isnan(auc) else f'{auc:.4f}'
    print(f'  {tag:<34} vols {len(group):>4}  det {n_match:>4}/{n_gt:<4} '
          f'= {det:.3f}   FP/vol {n_fp/len(group):.2f}   vol-AUC {a}')
    return {'n_volumes': len(group), 'n_gt': n_gt, 'n_matched': n_match,
            'det_rate': det, 'fp_per_vol': n_fp / len(group),
            'volume_auc': None if np.isnan(auc) else auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_root', required=True)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pos = load_positions(args.meta_root)
    dump = load_dump(args.dump)
    if not pos:
        raise SystemExit(f'no annotated_slice_indices found under {args.meta_root}')

    matched = [c for c in dump if c in pos]
    print('=' * 78)
    print(f'  POSITION LEAKAGE TEST')
    print(f'  metadata {Path(args.meta_root).name} | dump {Path(args.dump).name}')
    print('=' * 78)
    print(f'  {len(dump)} cases in dump, {len(pos)} with metadata, '
          f'{len(matched)} joined')
    if len(matched) < 0.8 * len(dump):
        print(f'  [warn] many cases failed to join. Check that metadata file '
              f'stems match case_id.')

    at_zero = [dump[c] for c in matched if pos[c] and min(pos[c]) == 0]
    elsewhere = [dump[c] for c in matched if pos[c] and min(pos[c]) > 0]

    print(f'\n  Annotated slice at sequence position 0: {len(at_zero)} volumes')
    print(f'  Annotated slice elsewhere (1-14):        {len(elsewhere)} volumes')

    print(f'\n  DETECTION AND CLASSIFICATION BY POSITION GROUP')
    a = summarize('annotated slice at position 0', at_zero)
    b = summarize('annotated slice at position 1-14', elsewhere)

    if a and b and not np.isnan(a['det_rate']) and not np.isnan(b['det_rate']):
        d = a['det_rate'] - b['det_rate']
        print(f'\n  Detection difference (pos 0 minus elsewhere): {d:+.3f}')
        if abs(d) < 0.10:
            print(f'  -> Comparable. No evidence that the model exploits')
            print(f'     sequence position. This supports the response.')
        else:
            better = 'position 0' if d > 0 else 'elsewhere'
            print(f'  -> Substantially better at {better}. If position 0 is')
            print(f'     favoured, a position shortcut cannot be ruled out and')
            print(f'     the annotation-independent (pure_grad) result should')
            print(f'     carry the response instead of this table.')

    # finer breakdown, in case the effect is graded rather than binary
    print(f'\n  DETECTION BY EXACT POSITION')
    by_pos = {}
    for c in matched:
        if not pos[c]:
            continue
        k = min(pos[c])
        by_pos.setdefault(k, []).append(dump[c])
    for k in sorted(by_pos):
        g = by_pos[k]
        n_gt = sum(r['n_gt'] for r in g)
        n_m = sum(r['n_matched'] for r in g)
        rate = n_m / n_gt if n_gt else float('nan')
        bar = '#' * int(round(rate * 40))
        print(f'    pos {k:>2} | n={len(g):>3} | det {rate:.3f} | {bar}')

    if args.out:
        json.dump({'meta_root': args.meta_root, 'dump': args.dump,
                   'at_position_0': a, 'elsewhere': b}, open(args.out, 'w'),
                  indent=2, default=float)
        print(f'\n  written: {args.out}')


if __name__ == '__main__':
    main()
