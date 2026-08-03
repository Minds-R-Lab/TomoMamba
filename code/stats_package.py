#!/usr/bin/env python3
"""
stats_package.py  ->  bootstrap CIs, maximum Youden's J, Brier scores, matched
operating points and paired DeLong tests, computed from per-case prediction dumps.

READ-ONLY, CPU-only, runs in seconds. No retraining, no GPU, no checkpoints.

WHAT IT PRODUCES, PER MODEL
    patient-level AUC with bootstrap 95% CI
    max Youden's J from the ROC curve, and the threshold that attains it
    sensitivity / specificity at tau=0.50 AND at the model's own tau*
    balanced accuracy at tau*
    Brier score, calibration slope and intercept, ECE
    sensitivity at the reference model's specificity  (matched specificity)
    specificity at the reference model's sensitivity  (matched sensitivity)
    detection rate, total false positives, FP per volume

WHAT IT PRODUCES, PER PAIR
    DeLong paired test for AUC difference vs the reference model,
    with the AUC difference, its standard error, z and two-sided p.

WHY THIS EXISTS
    says the J=0.45 vs J=0.05 comparison is unfair because every model
    is cut at the fixed 0.50 threshold, and BiGRU's low specificity there may
    be miscalibration rather than poor discrimination. The honest answer is
    to compare each model at its OWN best threshold, show the calibration
    numbers, and test whether the AUC gaps are significant at all. On 40
    patients they very likely are not, and saying so plainly is stronger
    than defending a difference the data cannot support.

USAGE
    python stats_package.py --dump_dir dumps_val_all_locked_20260708
    python stats_package.py --dump_dir dumps_test_all_locked_20260708 \
                            --ref "BASELINE_(v5.2)" --out stats_test.json

    Add BiGRU / slice-independent by dropping their per_case_*.jsonl files
    into the same directory. The script picks up every per_case_*.jsonl it
    finds, so no code change is needed.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_curve, roc_auc_score


# ---------------------------------------------------------------------------
# DeLong. Fast midrank implementation (Sun & Xu 2014, IEEE SPL 21(11):1389).
# ---------------------------------------------------------------------------

def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: (k, n) with the m positives first. Returns (aucs, cov)."""
    n = preds_sorted.shape[1] - m
    k = preds_sorted.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _midrank(preds_sorted[r, :m])
        ty[r, :] = _midrank(preds_sorted[r, m:])
        tz[r, :] = _midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    return aucs, sx / m + sy / n


def delong_test(labels, prob_a, prob_b):
    """Two-sided paired DeLong test. Returns dict with auc_a, auc_b, z, p."""
    labels = np.asarray(labels)
    order = (-labels).argsort(kind='mergesort')   # positives first, stable
    m = int(labels.sum())
    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        return {'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]),
                'diff': float(diff), 'se': 0.0, 'z': None, 'p': None}
    se = float(np.sqrt(var))
    z = float(diff / se)
    p = float(2 * (1 - sps.norm.cdf(abs(z))))
    return {'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]),
            'diff': float(diff), 'se': se, 'z': z, 'p': p}


# ---------------------------------------------------------------------------
# Loading and patient aggregation
# ---------------------------------------------------------------------------

def patient_of(case_id):
    return case_id.split('_')[0]


def load_dump(path, agg='max'):
    """Returns (patient_ids, labels, probs, detection totals)."""
    per_pt_prob = defaultdict(list)
    per_pt_label = {}
    n_gt = n_matched = n_fp = 0
    n_vol = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pid = patient_of(r['case_id'])
        per_pt_prob[pid].append(float(r['cancer_prob']))
        per_pt_label[pid] = 1 if r['gt_label'] == 'Cancer' else 0
        n_gt += int(r.get('n_gt', 0))
        n_matched += int(r.get('n_matched', 0))
        n_fp += int(r.get('n_fp', 0))
        n_vol += 1
    pids = sorted(per_pt_prob)
    fn = {'max': np.max, 'mean': np.mean}[agg]
    probs = np.array([fn(per_pt_prob[p]) for p in pids])
    labels = np.array([per_pt_label[p] for p in pids])
    det = {'n_gt': n_gt, 'n_matched': n_matched, 'n_fp': n_fp, 'n_vol': n_vol,
           'det_rate': n_matched / n_gt if n_gt else float('nan'),
           'fp_per_vol': n_fp / n_vol if n_vol else float('nan')}
    return pids, labels, probs, det


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def boot_ci(labels, probs, fn, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        try:
            vals.append(fn(labels[idx], probs[idx]))
        except Exception:
            pass
    if not vals:
        return (float('nan'), float('nan'))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def sens_spec_at(labels, probs, tau):
    pred = (probs >= tau).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn_ = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    sens = tp / (tp + fn_) if (tp + fn_) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return sens, spec, (tn, fp, fn_, tp)


def max_youden(labels, probs):
    fpr, tpr, thr = roc_curve(labels, probs)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(j[i]), float(thr[i]), float(tpr[i]), float(1 - fpr[i])


def calibration(labels, probs, n_bins=5):
    brier = float(np.mean((probs - labels) ** 2))
    eps = 1e-6
    p = np.clip(probs, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    try:
        import sklearn.linear_model as lm
        m = lm.LogisticRegression(penalty=None, solver='lbfgs')
        m.fit(logit.reshape(-1, 1), labels)
        slope = float(m.coef_[0][0])
        intercept = float(m.intercept_[0])
    except Exception:
        slope = intercept = float('nan')
    edges = np.linspace(probs.min(), probs.max() + 1e-9, n_bins + 1)
    ece = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (probs >= a) & (probs < b)
        if sel.sum() == 0:
            continue
        ece += sel.sum() / len(probs) * abs(probs[sel].mean() - labels[sel].mean())
    return {'brier': brier, 'cal_slope': slope, 'cal_intercept': intercept,
            'ece': float(ece)}


def sens_at_spec(labels, probs, target_spec):
    fpr, tpr, thr = roc_curve(labels, probs)
    ok = (1 - fpr) >= target_spec - 1e-9
    return float(tpr[ok].max()) if ok.any() else float('nan')


def spec_at_sens(labels, probs, target_sens):
    fpr, tpr, thr = roc_curve(labels, probs)
    ok = tpr >= target_sens - 1e-9
    return float((1 - fpr)[ok].max()) if ok.any() else float('nan')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_dir', required=True)
    ap.add_argument('--ref', default='BASELINE_(v5.2)',
                    help='Reference model for matched operating points and DeLong.')
    ap.add_argument('--agg', default='max', choices=['max', 'mean'])
    ap.add_argument('--tau_cls', type=float, default=0.50)
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    d = Path(args.dump_dir)
    files = sorted(d.glob('per_case_*.jsonl'))
    if not files:
        raise SystemExit(f'no per_case_*.jsonl under {d}')

    models = {}
    for f in files:
        name = f.stem.replace('per_case_', '')
        models[name] = load_dump(f, agg=args.agg)

    if args.ref not in models:
        args.ref = list(models)[0]
    _, ref_labels, ref_probs, _ = models[args.ref]
    ref_sens, ref_spec, _ = sens_spec_at(ref_labels, ref_probs, args.tau_cls)

    print('=' * 78)
    print(f'  STATISTICS PACKAGE  |  {d.name}  |  agg={args.agg}  '
          f'|  reference = {args.ref}')
    print('=' * 78)

    results = {}
    for name, (pids, labels, probs, det) in models.items():
        auc = roc_auc_score(labels, probs)
        lo, hi = boot_ci(labels, probs, roc_auc_score, args.n_boot)
        s50, p50, cm = sens_spec_at(labels, probs, args.tau_cls)
        jmax, tstar, s_star, p_star = max_youden(labels, probs)
        jlo, jhi = boot_ci(labels, probs,
                           lambda l, p: max_youden(l, p)[0], args.n_boot)
        cal = calibration(labels, probs)
        row = {
            'n_patients': len(labels), 'n_cancer': int(labels.sum()),
            'auc': float(auc), 'auc_ci': [lo, hi],
            'sens@0.50': s50, 'spec@0.50': p50,
            'J@0.50': s50 + p50 - 1,
            'confusion@0.50': {'tn': cm[0], 'fp': cm[1], 'fn': cm[2], 'tp': cm[3]},
            'J_max': jmax, 'J_max_ci': [jlo, jhi], 'tau_star': tstar,
            'sens@tau_star': s_star, 'spec@tau_star': p_star,
            'balanced_acc@tau_star': 0.5 * (s_star + p_star),
            'prob_range': [float(probs.min()), float(probs.max())],
            'sens@ref_spec': sens_at_spec(labels, probs, ref_spec),
            'spec@ref_sens': spec_at_sens(labels, probs, ref_sens),
            'detection': det,
        }
        row.update(cal)
        results[name] = row

        print(f'\n  {name}')
        print(f'    patients {row["n_patients"]} ({row["n_cancer"]} cancer)  '
              f'| prob range [{row["prob_range"][0]:.3f}, {row["prob_range"][1]:.3f}]')
        print(f'    AUC            {auc:.4f}  95% CI [{lo:.3f}, {hi:.3f}]')
        print(f'    at tau=0.50    sens {s50:.3f}  spec {p50:.3f}  '
              f'J {s50 + p50 - 1:.3f}')
        print(f'    at tau*={tstar:.3f}  sens {s_star:.3f}  spec {p_star:.3f}  '
              f'J_max {jmax:.3f}  95% CI [{jlo:.3f}, {jhi:.3f}]  '
              f'balAcc {0.5 * (s_star + p_star):.3f}')
        print(f'    calibration    Brier {cal["brier"]:.4f}  slope '
              f'{cal["cal_slope"]:.3f}  intercept {cal["cal_intercept"]:.3f}  '
              f'ECE {cal["ece"]:.4f}')
        print(f'    matched        sens@ref_spec({ref_spec:.2f}) '
              f'{row["sens@ref_spec"]:.3f}   spec@ref_sens({ref_sens:.2f}) '
              f'{row["spec@ref_sens"]:.3f}')
        print(f'    detection      {det["n_matched"]}/{det["n_gt"]} '
              f'= {det["det_rate"]:.3f}   FP {det["n_fp"]} '
              f'({det["fp_per_vol"]:.2f}/volume over {det["n_vol"]} volumes)')

    print('\n' + '=' * 78)
    print(f'  DeLong paired AUC tests vs {args.ref}')
    print('=' * 78)
    print(f'  {"model":<28}{"dAUC":>9}{"SE":>9}{"z":>8}{"p":>10}   verdict')
    pairs = {}
    for name, (pids, labels, probs, det) in models.items():
        if name == args.ref:
            continue
        assert len(labels) == len(ref_labels), 'patient sets differ'
        t = delong_test(ref_labels, ref_probs, probs)
        pairs[name] = t
        verdict = ('n.s.' if (t['p'] is None or t['p'] >= 0.05)
                   else 'significant')
        pstr = 'n/a' if t['p'] is None else f'{t["p"]:.4f}'
        zstr = 'n/a' if t['z'] is None else f'{t["z"]:.3f}'
        print(f'  {name:<28}{t["diff"]:>+9.4f}{t["se"]:>9.4f}'
              f'{zstr:>8}{pstr:>10}   {verdict}')

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'per_model': results, 'delong_vs_ref': pairs,
                       'reference': args.ref, 'dump_dir': str(d)}, f, indent=2)
        print(f'\n  written: {args.out}')


if __name__ == '__main__':
    main()
