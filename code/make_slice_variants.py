#!/usr/bin/env python3
"""
make_slice_variants.py  ->  builds the slice-selection ablation datasets for


WHAT IT IS
    A thin wrapper around the production preprocessing pipeline
    ~/DBT/trial1/preprcessing_pipeline_gradient_based_normals.py
    (note the typo in that filename; it is the real one, KEEP_SLICES = 15,
    BBOX_EXPANSION = 1.2, OUTPUT_DIR = DBT_CancerBenignNormal_Gradient_1024_15).

    It swaps ONE function, the slice selector, and writes to a new output
    directory. Everything else -- Duke laterality, windowing, 1024x1024
    resize, 1.2x box expansion, metadata format, folder layout -- runs
    unchanged, so the variant datasets are comparable to the main dataset
    by construction rather than by hope.

WHY A WRAPPER AND NOT AN EDITED COPY
    An edited copy drifts. This imports the production module and monkeypatches
    the module-level `select_annotated_plus_gradient_based`, which the class
    calls as a global, so the patch takes effect at call time. If the pipeline
    is ever changed, this wrapper inherits the change.

WHY THE EXISTING .npy FILES CANNOT BE REUSED
    They already contain only the 15 selected slices. Choosing 15 DIFFERENT
    slices needs the other 35-85, which exist only in the DICOMs. So this has
    to re-read from the manifest. Only Cancer and Benign are processed, about
    275 volumes, not the normals.

MODES
    gradient    the control. Delegates to the ORIGINAL function untouched.
                Must reproduce the existing dataset. Verify before trusting
                anything else.
    random      annotated slices retained, remaining slots filled at random
    uniform     annotated slices retained, remaining slots evenly spaced
    contiguous  a contiguous block of 15 slices centred on the annotated slice
    pure_grad   gradient scoring with NO forced retention of the annotated
                slice. This is 's "lesion-independent" sampling. Detection
                is only scoreable on the subset where the annotated slice
                happens to survive, so treat it as classification-only.

USAGE
    source ~/tomomamba/bin/activate

    # 1. VERIFY FIRST. Rebuilds a few cases with the control selector and
    #    checks them against the existing metadata. Writes nothing else.
    python make_slice_variants.py --mode gradient --verify_only

    # 2. Then build a variant.
    python make_slice_variants.py --mode random
    python make_slice_variants.py --mode uniform

    Output goes to /mnt/e/DBT_SliceAblation_<mode>_1024_15 by default.
"""

import sys
import json
import random
import argparse
from pathlib import Path
from typing import List

import numpy as np

PIPELINE_DIR = Path.home() / 'DBT' / 'trial1'
PIPELINE_MODULE = 'preprcessing_pipeline_gradient_based_normals'   # sic
PROD_DATA = Path('/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15')


# --------------------------------------------------------------- selectors

def _finish(chosen, keep):
    """Guarantee exactly `keep` sorted unique indices."""
    out = sorted(set(int(c) for c in chosen))
    return out[:keep]


def sel_random(volume, annotated_slices, keep=15, min_sep=1, seed=0):
    S = volume.shape[0]
    keep = max(1, min(keep, S))
    chosen = sorted(set(int(a) for a in annotated_slices if 0 <= a < S))
    if len(chosen) >= keep:
        return _finish(chosen, keep)
    pool = [i for i in range(S) if i not in chosen]
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen.extend(pool[:keep - len(chosen)])
    return _finish(chosen, keep)


def sel_uniform(volume, annotated_slices, keep=15, min_sep=1, seed=0):
    S = volume.shape[0]
    keep = max(1, min(keep, S))
    chosen = sorted(set(int(a) for a in annotated_slices if 0 <= a < S))
    if len(chosen) >= keep:
        return _finish(chosen, keep)
    need = keep - len(chosen)
    grid = np.linspace(0, S - 1, num=keep, dtype=int).tolist()
    for g in grid:                      # walk the even grid, skip collisions
        if need == 0:
            break
        if g not in chosen:
            chosen.append(g); need -= 1
    i = 0                               # top up if the grid collided a lot
    while need > 0 and i < S:
        if i not in chosen:
            chosen.append(i); need -= 1
        i += 1
    return _finish(chosen, keep)


def sel_contiguous(volume, annotated_slices, keep=15, min_sep=1, seed=0):
    S = volume.shape[0]
    keep = max(1, min(keep, S))
    ann = sorted(set(int(a) for a in annotated_slices if 0 <= a < S))
    centre = int(np.median(ann)) if ann else S // 2
    start = int(np.clip(centre - keep // 2, 0, max(0, S - keep)))
    block = list(range(start, min(start + keep, S)))
    for a in ann:                       # keep ground truth scoreable
        if a not in block:
            block[-1] = a
            block = sorted(set(block))
    i = 0
    while len(block) < keep and i < S:
        if i not in block:
            block.append(i)
        i += 1
    return _finish(block, keep)


def make_pure_grad(module):
    """Gradient selection with NO annotated-slice retention."""
    def sel(volume, annotated_slices, keep=15, min_sep=1, seed=0):
        S = volume.shape[0]
        keep = max(1, min(keep, S))
        scores = [(i, module._gradient_score(volume[i])) for i in range(S)]
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = []
        for idx, _ in scores:
            if len(chosen) >= keep:
                break
            if all(abs(idx - c) >= min_sep for c in chosen):
                chosen.append(idx)
        i = 0
        while len(chosen) < keep and i < S:
            if i not in chosen:
                chosen.append(i)
            i += 1
        return _finish(chosen, keep)
    return sel


# --------------------------------------------------------------- verify

def verify(module, n_check):
    """Rebuild the control selection from metadata and compare."""
    meta_dirs = sorted(PROD_DATA.glob('*/metadata/*')) or \
                sorted(PROD_DATA.glob('metadata/*/*'))
    metas = []
    for pat in ('**/metadata/**/*.json', '**/*.json'):
        metas = sorted(PROD_DATA.glob(pat))
        if metas:
            break
    if not metas:
        print(f'  [verify] no metadata json found under {PROD_DATA}')
        print(f'  [verify] cannot verify; inspect the layout manually')
        return False

    print(f'  [verify] found {len(metas)} metadata files, checking '
          f'{min(n_check, len(metas))}')
    ok = bad = 0
    for p in metas[:n_check]:
        try:
            m = json.load(open(p))
        except Exception:
            continue
        sel = m.get('selected_slices')
        if sel is None:
            continue
        if len(sel) != module.DBTCancerBenignNormalPreprocessor.KEEP_SLICES:
            print(f'    {p.name}: {len(sel)} slices, expected '
                  f'{module.DBTCancerBenignNormalPreprocessor.KEEP_SLICES}')
            bad += 1
            continue
        if sel != sorted(sel):
            print(f'    {p.name}: selected_slices NOT ascending')
            bad += 1
            continue
        ann = m.get('annotated_slice_indices', [])
        if ann and not all(0 <= a < len(sel) for a in ann):
            print(f'    {p.name}: annotated_slice_indices out of range {ann}')
            bad += 1
            continue
        ok += 1
    print(f'  [verify] {ok} consistent, {bad} problems')
    print(f'  [verify] KEEP_SLICES={module.DBTCancerBenignNormalPreprocessor.KEEP_SLICES} '
          f'MIN_SEP={module.DBTCancerBenignNormalPreprocessor.MIN_SEP}')
    return bad == 0


# --------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['gradient', 'random', 'uniform', 'contiguous',
                             'pure_grad'])
    ap.add_argument('--manifest_dir', default='/mnt/e/manifest-1617905855234')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--verify_only', action='store_true')
    ap.add_argument('--n_check', type=int, default=25)
    args = ap.parse_args()

    if not PIPELINE_DIR.exists():
        raise SystemExit(f'pipeline dir not found: {PIPELINE_DIR}')
    sys.path.insert(0, str(PIPELINE_DIR))
    import importlib
    module = importlib.import_module(PIPELINE_MODULE)

    cls = module.DBTCancerBenignNormalPreprocessor
    print('=' * 74)
    print(f'  SLICE VARIANT BUILDER  |  mode = {args.mode}')
    print(f'  pipeline : {PIPELINE_DIR / (PIPELINE_MODULE + ".py")}')
    print(f'  settings : KEEP_SLICES={cls.KEEP_SLICES}  MIN_SEP={cls.MIN_SEP}')
    print('=' * 74)

    if cls.KEEP_SLICES != 15:
        raise SystemExit(f'ABORT: KEEP_SLICES is {cls.KEEP_SLICES}, expected 15. '
                         f'Wrong pipeline file.')

    if args.verify_only:
        verify(module, args.n_check)
        print('\n  verify only, nothing written.')
        return

    original = module.select_annotated_plus_gradient_based
    if args.mode == 'gradient':
        patched = original                      # control, untouched
    elif args.mode == 'pure_grad':
        patched = make_pure_grad(module)
    else:
        base = {'random': sel_random, 'uniform': sel_uniform,
                'contiguous': sel_contiguous}[args.mode]
        seed = args.seed

        def patched(volume, annotated_slices, keep=15, min_sep=1, _b=base,
                    _s=seed):
            return _b(volume, annotated_slices, keep=keep, min_sep=min_sep,
                      seed=_s)

    module.select_annotated_plus_gradient_based = patched
    print(f'  selector patched -> {args.mode}\n')

    out = args.out_dir or f'/mnt/e/DBT_SliceAblation_{args.mode}_1024_15'
    pre = cls(manifest_dir=args.manifest_dir,
              output_dir=out,
              target_size=(1024, 1024),
              skip_existing=False,
              bbox_expansion=1.2)

    pre.run(include_cancer_benign=True,
            include_normals=False,
            normal_sample_n=0)

    print(f'\n  dataset written: {out}')
    print(f'  next: train with --data_root {out}, then evaluate at '
          f'--score_thresh 0.16')


if __name__ == '__main__':
    main()
