#!/usr/bin/env python3
"""
fallback_analysis.py  ->  characterises the global-average-pooling fallback path.

asks for four things:
    (a) how often the no-detection fallback classifier fires
    (b) classification performance on ROI-based cases
    (c) classification performance on fallback cases
    (d) the effect of removing the fallback classifier

All four come out of the per-case dumps you already have. CPU only, seconds.

HOW A FALLBACK VOLUME IS IDENTIFIED
    The dump records n_matched (detections matched to a GT box) and n_fp
    (detections matched to nothing). A volume where BOTH are zero produced no
    detections at all, so the global-average-pooling fallback path ran. This
    reproduces the 10/75 = 13.3% that analyze_checkpoint.py measured directly
    from the model, which is the cross-check that the inference is sound.

ON (d)
    You cannot literally delete the fallback head without retraining, so this
    reports the honest proxy: patient-level metrics when fallback volumes are
    excluded from the max-pool. Patients left with no ROI-based view at all
    are reported separately rather than silently dropped, because dropping
    them is what would inflate the number. Treat the fallback as a
    leave-out analysis, not as an ablation.

USAGE
    python fallback_analysis.py \
        --dump dumps_val_all_locked_20260708/"per_case_BASELINE_(v5.2).jsonl"
    python fallback_analysis.py \
        --dump dumps_test_all_locked_20260708/"per_case_BASELINE_(v5.2).jsonl"
"""

import json
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def patient_of(cid):
    return cid.split('_')[0]


def pat_metrics(pairs, tau=0.50):
    """pairs: list of (label, prob). Returns dict."""
    if not pairs:
        return None
    y = np.array([p[0] for p in pairs])
    p = np.array([p[1] for p in pairs])
    out = {'n': len(y), 'n_cancer': int(y.sum())}
    out['auc'] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float('nan')
    pred = (p > tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    out['sens'] = tp / (tp + fn) if (tp + fn) else float('nan')
    out['spec'] = tn / (tn + fp) if (tn + fp) else float('nan')
    out['J'] = out['sens'] + out['spec'] - 1
    out['acc'] = (tp + tn) / len(y)
    return out


def show(title, m):
    if m is None or m['n'] == 0:
        print(f'  {title:<34} (none)')
        return
    auc = 'n/a' if np.isnan(m['auc']) else f'{m["auc"]:.4f}'
    print(f'  {title:<34} n={m["n"]:>3} ({m["n_cancer"]} cancer)  '
          f'AUC {auc:>6}  sens {m["sens"]:.3f}  spec {m["spec"]:.3f}  '
          f'J {m["J"]:+.3f}  acc {m["acc"]:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--tau', type=float, default=0.50)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dump) if l.strip()]
    for r in rows:
        r['fallback'] = (r['n_matched'] == 0 and r['n_fp'] == 0)
        r['y'] = 1 if r['gt_label'] == 'Cancer' else 0

    n = len(rows)
    fb = [r for r in rows if r['fallback']]
    roi = [r for r in rows if not r['fallback']]

    print('=' * 78)
    print(f'  FALLBACK ANALYSIS  |  {args.dump}')
    print('=' * 78)

    print(f'\n(a) How often the fallback fires')
    print(f'  volumes total                      {n}')
    print(f'  no detection -> fallback used      {len(fb)}  '
          f'({100 * len(fb) / n:.1f}%)')
    for cls, lab in ((1, 'Cancer'), (0, 'Benign')):
        tot = sum(1 for r in rows if r['y'] == cls)
        got = sum(1 for r in fb if r['y'] == cls)
        print(f'    among {lab:<7} volumes            {got}/{tot} '
              f'({100 * got / tot:.1f}%)' if tot else '')

    print(f'\n(b,c) Volume-level classification, ROI path vs fallback path')
    show('ROI-based volumes', pat_metrics([(r['y'], r['cancer_prob'])
                                           for r in roi], args.tau))
    show('fallback volumes', pat_metrics([(r['y'], r['cancer_prob'])
                                          for r in fb], args.tau))

    # ---- patient level ----
    def pool(subset):
        by = defaultdict(list); lab = {}
        for r in subset:
            pid = patient_of(r['case_id'])
            by[pid].append(r['cancer_prob']); lab[pid] = r['y']
        return [(lab[p], max(v)) for p, v in sorted(by.items())]

    all_pairs = pool(rows)
    roi_pairs = pool(roi)
    roi_pids = {patient_of(r['case_id']) for r in roi}
    all_pids = {patient_of(r['case_id']) for r in rows}
    lost = sorted(all_pids - roi_pids)

    print(f'\n(d) Effect of removing the fallback path (patient level, '
          f'max over views)')
    show('all views (as published)', pat_metrics(all_pairs, args.tau))
    show('ROI views only', pat_metrics(roi_pairs, args.tau))
    print(f'  patients with NO ROI-based view    {len(lost)}'
          + (f'  -> {", ".join(lost)}' if lost else ''))
    if lost:
        print('  These patients would have no prediction at all if the '
              'fallback were removed.')
        print('  Report that count; do not quietly drop them from the '
              'denominator.')

    if args.out:
        json.dump({
            'dump': args.dump, 'n_volumes': n, 'n_fallback': len(fb),
            'fallback_rate': len(fb) / n,
            'volume_roi': pat_metrics([(r['y'], r['cancer_prob']) for r in roi]),
            'volume_fallback': pat_metrics([(r['y'], r['cancer_prob']) for r in fb]),
            'patient_all': pat_metrics(all_pairs),
            'patient_roi_only': pat_metrics(roi_pairs),
            'patients_without_roi_view': lost,
        }, open(args.out, 'w'), indent=2, default=str)
        print(f'\n  written: {args.out}')


if __name__ == '__main__':
    main()
