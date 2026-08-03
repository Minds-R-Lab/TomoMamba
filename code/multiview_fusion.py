#!/usr/bin/env python3
"""
multiview_fusion.py  ->  comparison of CC/MLO view-aggregation rules.

WHY IT EXISTS
    asks for the impact of multi-view fusion to be isolated. The only
    figures previously available came from POOLED CROSS-FOLD predictions, which
    are not citable: each fold's model has its own probability scale, so pooling
    raw probabilities destroys cross-fold ranking. This recomputes the same
    comparison on the validation and test dumps, where all predictions come
    from one model on one scale.

WHAT IT COMPARES
    Each patient contributes 2 to 4 volumes (CC and MLO per breast). The
    patient-level score is formed by aggregating those volume scores:

      max        the published choice
      mean       average across views
      top2       mean of the two highest
      noisy-or   1 - prod(1 - p), the probabilistic union

    Reports patient-level AUC with bootstrap CI for each, plus a paired
    bootstrap of each alternative against max.

USAGE
    python multiview_fusion.py \
        --dump 'dumps_val_all_locked_20260708/per_case_BASELINE_(v5.2).jsonl'
    python multiview_fusion.py \
        --dump 'dumps_test_all_locked_20260708/per_case_BASELINE_(v5.2).jsonl'
"""

import json
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


def agg(scores, how):
    s = np.asarray(scores, dtype=float)
    if how == 'max':
        return float(s.max())
    if how == 'mean':
        return float(s.mean())
    if how == 'top2':
        k = min(2, len(s))
        return float(np.sort(s)[-k:].mean())
    if how == 'noisy_or':
        return float(1.0 - np.prod(1.0 - s))
    raise ValueError(how)


def load(path):
    by, lab = defaultdict(list), {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pid = r['case_id'].split('_')[0]
        by[pid].append(float(r['cancer_prob']))
        lab[pid] = 1 if r['gt_label'] == 'Cancer' else 0
    pids = sorted(by)
    return pids, np.array([lab[p] for p in pids]), [by[p] for p in pids]


def boot_ci(y, p, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n, v = len(y), []
    for _ in range(n_boot):
        i = rng.choice(n, n, replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def paired(y, a, b, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    n, d = len(y), []
    for _ in range(n_boot):
        i = rng.choice(n, n, replace=True)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    if not len(d):
        return None
    lo, hi = np.percentile(d, [2.5, 97.5])
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.mean(d)), float(lo), float(hi), float(min(pv, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--tau', type=float, default=0.50)
    ap.add_argument('--n_boot', type=int, default=4000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pids, y, views = load(args.dump)
    nv = [len(v) for v in views]
    print('=' * 74)
    print(f'  MULTI-VIEW FUSION  |  {args.dump.split("/")[-1]}')
    print('=' * 74)
    print(f'  {len(pids)} patients ({int(y.sum())} cancer), '
          f'{sum(nv)} volumes, {min(nv)}-{max(nv)} views per patient')

    methods = ['max', 'mean', 'top2', 'noisy_or']
    scores, rows = {}, {}
    print(f'\n  {"aggregation":<12}{"AUC":>9}{"95% CI":>20}{"sens":>8}{"spec":>8}')
    for how in methods:
        p = np.array([agg(v, how) for v in views])
        scores[how] = p
        auc = roc_auc_score(y, p)
        lo, hi = boot_ci(y, p, args.n_boot)
        pred = (p > args.tau).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        se = tp / (tp + fn) if (tp + fn) else np.nan
        sp = tn / (tn + fp) if (tn + fp) else np.nan
        rows[how] = {'auc': float(auc), 'ci': [lo, hi],
                     'sens': float(se), 'spec': float(sp)}
        star = '  <- published' if how == 'max' else ''
        print(f'  {how:<12}{auc:>9.4f}{f"[{lo:.3f}, {hi:.3f}]":>20}'
              f'{se:>8.3f}{sp:>8.3f}{star}')

    print(f'\n  Paired bootstrap against max ({args.n_boot} resamples)')
    print(f'  {"aggregation":<12}{"dAUC":>9}{"95% CI":>22}{"p":>9}')
    pairs = {}
    for how in methods[1:]:
        r = paired(y, scores['max'], scores[how], args.n_boot)
        if r is None:
            continue
        m, lo, hi, pv = r
        pairs[how] = {'mean_diff': m, 'ci': [lo, hi], 'p': pv}
        print(f'  {how:<12}{m:>+9.4f}{f"[{lo:+.3f}, {hi:+.3f}]":>22}{pv:>9.3f}')

    if args.out:
        json.dump({'dump': args.dump, 'n_patients': len(pids),
                   'per_method': rows, 'vs_max': pairs},
                  open(args.out, 'w'), indent=2)
        print(f'\n  written: {args.out}')


if __name__ == '__main__':
    main()
