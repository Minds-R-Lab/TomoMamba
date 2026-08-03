#!/usr/bin/env python3
"""
failure_cases.py  ->  failure-mode breakdown for missed and misclassified cases.

Failure-case analysis, separating missed cancers
and false positives". This produces the whole answer from the per-case dumps
you already have. CPU only, seconds, no GPU contention with the seed sweep.

WHAT IT SEPARATES
    Failure is not one thing here, and lumping it together hides the story.
    A cancer can be missed three different ways:

      D  DETECTION MISS      the detector found no box matching the lesion,
                             so the ROI classifier never saw it
      C  CLASSIFICATION MISS the lesion WAS localised but the ROI classifier
                             called it benign. This is the interesting failure
      F  FALLBACK MISS       no detections at all in the volume, so the
                             global-average-pooling fallback decided

    And a benign case can fail one way: called cancer (false positive).

    Separating D from C matters because they point at different
    components. D is a detector problem, C is a classifier problem.

OUTPUT
    a summary table, per-volume and per-patient
    the case IDs of every missed cancer, grouped by failure mode, ready to
      feed to visualize.py for the figure
    false-positive burden per volume, split by class
    the largest and smallest missed lesions, since lesion size is the usual
      the first explanation worth testing

USAGE
    python failure_cases.py \
        --dump 'dumps_val_all_locked_20260708/per_case_BASELINE_(v5.2).jsonl' \
        --out failures_val.json

    python failure_cases.py \
        --dump 'dumps_test_all_locked_20260708/per_case_BASELINE_(v5.2).jsonl' \
        --out failures_test.json

    Add --meta_root to pull lesion sizes from the preprocessing metadata:
    --meta_root /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15/metadata/validation
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def load(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r['y'] = 1 if r['gt_label'] == 'Cancer' else 0
        r['patient'] = r['case_id'].split('_')[0]
        r['no_det'] = (r['n_matched'] == 0 and r['n_fp'] == 0)
        rows.append(r)
    return rows


def lesion_areas(meta_root, case_id):
    """Return GT box areas as a fraction of the image, if metadata is there."""
    if not meta_root:
        return []
    for cls in ('Cancer', 'Benign'):
        p = Path(meta_root) / cls / f'{case_id}.json'
        if p.exists():
            try:
                meta = json.load(open(p))
            except Exception:
                return []
            out = []
            for b in meta.get('boxes', []):
                w, h = float(b.get('width', 0)), float(b.get('height', 0))
                if w > 1.0 or h > 1.0:           # pixel coords, normalise
                    shape = meta.get('volume_shape', [1024, 1024, 1024])
                    w /= float(shape[-1]); h /= float(shape[0])
                out.append(w * h)
            return out
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--tau', type=float, default=0.50)
    ap.add_argument('--meta_root', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = load(args.dump)
    cancer = [r for r in rows if r['y'] == 1]
    benign = [r for r in rows if r['y'] == 0]

    print('=' * 78)
    print(f'  FAILURE CASE ANALYSIS  |  {Path(args.dump).name}')
    print('=' * 78)
    print(f'  {len(rows)} volumes  ({len(cancer)} cancer, {len(benign)} benign)')

    # ---------------- volume-level failure modes on cancer ----------------
    det_miss, cls_miss, fb_miss, correct = [], [], [], []
    for r in cancer:
        called_cancer = r['cancer_prob'] > args.tau
        if r['no_det']:
            (correct if called_cancer else fb_miss).append(r)
        elif r['n_matched'] == 0:
            (correct if called_cancer else det_miss).append(r)
        else:
            (correct if called_cancer else cls_miss).append(r)

    n = len(cancer)
    print(f'\n  CANCER VOLUMES: how they fail')
    print(f'    correctly called cancer                  {len(correct)}/{n}')
    print(f'    D  lesion not localised, called benign   {len(det_miss)}/{n}')
    print(f'    C  lesion localised, called benign       {len(cls_miss)}/{n}')
    print(f'    F  no detections at all, fallback wrong   {len(fb_miss)}/{n}')

    for tag, label, group in (('D', 'DETECTION misses', det_miss),
                              ('C', 'CLASSIFICATION misses', cls_miss),
                              ('F', 'FALLBACK misses', fb_miss)):
        if not group:
            continue
        print(f'\n  {tag}  {label} ({len(group)}) '
              f'-- use these for the figure')
        for r in sorted(group, key=lambda z: z['cancer_prob']):
            areas = lesion_areas(args.meta_root, r['case_id'])
            asz = (f'  lesion area {100 * min(areas):.3f}% of image'
                   if areas else '')
            print(f'    {r["case_id"]:<38} p={r["cancer_prob"]:.3f}  '
                  f'det {r["n_matched"]}/{r["n_gt"]}  FP {r["n_fp"]}{asz}')

    # ---------------- lesion size, missed versus found ----------------
    if args.meta_root:
        missed_a, found_a = [], []
        for r in cancer:
            a = lesion_areas(args.meta_root, r['case_id'])
            if not a:
                continue
            (missed_a if r['n_matched'] == 0 else found_a).extend(a)
        if missed_a and found_a:
            print(f'\n  LESION SIZE (percent of image area)')
            print(f'    localised lesions   median {100*np.median(found_a):.3f}'
                  f'   n={len(found_a)}')
            print(f'    missed lesions      median {100*np.median(missed_a):.3f}'
                  f'   n={len(missed_a)}')
            ratio = np.median(found_a) / max(np.median(missed_a), 1e-12)
            print(f'    localised lesions are {ratio:.2f}x larger by median '
                  f'area')

    # ---------------- false positives on benign ----------------
    fp_vol = [r for r in benign if r['cancer_prob'] > args.tau]
    print(f'\n  BENIGN VOLUMES called cancer: {len(fp_vol)}/{len(benign)}')
    for r in sorted(fp_vol, key=lambda z: -z['cancer_prob'])[:12]:
        print(f'    {r["case_id"]:<38} p={r["cancer_prob"]:.3f}  '
              f'det {r["n_matched"]}/{r["n_gt"]}  FP boxes {r["n_fp"]}')
    if len(fp_vol) > 12:
        print(f'    ... and {len(fp_vol) - 12} more')

    fp_c = sum(r['n_fp'] for r in cancer)
    fp_b = sum(r['n_fp'] for r in benign)
    print(f'\n  SPURIOUS DETECTION BOXES (not matched to any GT lesion)')
    print(f'    on cancer volumes   {fp_c} over {len(cancer)} volumes '
          f'= {fp_c / max(len(cancer), 1):.2f} per volume')
    print(f'    on benign volumes   {fp_b} over {len(benign)} volumes '
          f'= {fp_b / max(len(benign), 1):.2f} per volume')

    # ---------------- patient level ----------------
    by = defaultdict(list); lab = {}
    for r in rows:
        by[r['patient']].append(r['cancer_prob']); lab[r['patient']] = r['y']
    fn_pat = [p for p in by if lab[p] == 1 and max(by[p]) <= args.tau]
    fp_pat = [p for p in by if lab[p] == 0 and max(by[p]) > args.tau]
    print(f'\n  PATIENT LEVEL (max over views, tau_cls = {args.tau})')
    print(f'    {len(by)} patients | false negatives {len(fn_pat)} | '
          f'false positives {len(fp_pat)}')
    if fn_pat:
        print(f'    missed cancer patients: {", ".join(sorted(fn_pat))}')

    if args.out:
        json.dump({
            'dump': args.dump, 'n_volumes': len(rows),
            'cancer_volumes': len(cancer), 'benign_volumes': len(benign),
            'detection_misses': [r['case_id'] for r in det_miss],
            'classification_misses': [r['case_id'] for r in cls_miss],
            'fallback_misses': [r['case_id'] for r in fb_miss],
            'benign_called_cancer': [r['case_id'] for r in fp_vol],
            'fp_boxes_per_cancer_volume': fp_c / max(len(cancer), 1),
            'fp_boxes_per_benign_volume': fp_b / max(len(benign), 1),
            'patient_false_negatives': sorted(fn_pat),
            'patient_false_positives': sorted(fp_pat),
        }, open(args.out, 'w'), indent=2)
        print(f'\n  written: {args.out}')


if __name__ == '__main__':
    main()
