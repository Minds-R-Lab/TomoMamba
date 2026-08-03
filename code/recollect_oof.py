#!/usr/bin/env python3
"""
recollect_oof.py  ->  regenerate each CV fold's out-of-fold predictions at the
DETECTION threshold actually reported, without retraining anything.

THE PROBLEM THIS FIXES
    train_stage2.py line 465 hardcodes score_thresh=0.20 inside eval_epoch, and
    cross_validate_resumable.py defaults --oof_score_thresh to 0.20. The reported
    reports tau_det = 0.16. So every cross-validation number produced so far
    was computed at 0.20. Reporting a CV at 0.20 next to a main result at 0.16
    is inconsistent, so this script recollects the out-of-fold predictions at
    the reported threshold.

WHY IT COSTS NOTHING
    The relaunched CV was started with --save_fold_checkpoints, so each fold's
    best_model.pt survives on disk. The detection threshold only affects
    INFERENCE, never training, so the same weights can be re-scored at any
    threshold. This is a forward pass over each fold's held-out patients:
    minutes, not hours.

WHAT IT DOES
    Rebuilds the identical fold assignment (deterministic given data_root,
    cv_splits, folds and seed), loads each fold's checkpoint, re-runs the
    detection -> ROI-classification path at --score_thresh, and writes
    fold_N/oof_tau{X}.json plus a pooled summary. It does NOT overwrite
    fold_result.json, so nothing you already have is destroyed.

USAGE  (after the CV finishes)
    source ~/tomomamba/bin/activate
    cd ~/DBT/MambaCenterNet_v5.2
    python recollect_oof.py --score_thresh 0.16

    Then point the operating-point analysis at the new files:
    python sweep_oof.py --cv_dir /mnt/e/DBT_Stage2_MambaCenterNet_v5.2_cv \
        --oof_file oof_tau0.16.json      # if sweep_oof.py supports it
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cross_validate_resumable import (index_all_cases, make_folds, FoldDataset,
                                      patient_id_of, bootstrap_auc_ci,
                                      collect_oof_patient_predictions)
from train_stage2 import collate_fn
from centernet_models import MambaCenterNet, compute_patient_metrics


def find_ckpt(fold_dir, prefer):
    order = ([prefer] if prefer else []) + ['best_model.pt',
                                            'best_patient_auc.pt', 'last.pt']
    for name in order:
        p = fold_dir / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',
                    default='/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15')
    ap.add_argument('--save_dir',
                    default='/mnt/e/DBT_Stage2_MambaCenterNet_v5.2_cv')
    ap.add_argument('--cv_splits', nargs='+', default=['train', 'validation'])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--spatial_size', type=int, default=384)
    ap.add_argument('--val_batch_size', type=int, default=4)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_boxes', type=int, default=20)
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--score_thresh', type=float, default=0.16,
                    help="operating tau_det")
    ap.add_argument('--ckpt_name', default=None,
                    help='force a specific checkpoint filename per fold')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    tag = f'{args.score_thresh:g}'

    print('=' * 74)
    print(f'  RECOLLECT OOF at tau_det = {args.score_thresh}')
    print(f'  save_dir = {save_dir}')
    print('=' * 74)

    cases, _ = index_all_cases(args.data_root, args.cv_splits)
    if not cases:
        raise SystemExit('no cases found; check --data_root')
    folds, groups, labels, method, _ = make_folds(cases, args.folds, args.seed)
    print(f'  {len(cases)} cases | '
          f'{len(set(patient_id_of(c["case_id"]) for c in cases))} patients | '
          f'split method {method}\n')

    merged, per_fold = {}, {}
    for k, (tr_idx, va_idx) in enumerate(folds):
        fold_dir = save_dir / f'fold_{k + 1}'
        ckpt = find_ckpt(fold_dir, args.ckpt_name)
        if ckpt is None:
            print(f'  fold {k + 1}: no checkpoint on disk, SKIPPED '
                  f'(was --save_fold_checkpoints passed?)')
            continue

        val_cases = [cases[i] for i in va_idx]
        ds = FoldDataset(val_cases, 'validation', args.spatial_size,
                         max_boxes=args.max_boxes, augment=False)
        loader = DataLoader(ds, args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            collate_fn=collate_fn, pin_memory=True)

        model = MambaCenterNet(num_classes=2, dropout=0.7, use_mamba=True,
                              spatial_size=args.spatial_size,
                              backbone=args.backbone).to(device)
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        state = ck.get('model_state_dict', ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f'  fold {k + 1}: [warn] {len(missing)} missing / '
                  f'{len(unexpected)} unexpected tensors')

        oof = collect_oof_patient_predictions(
            model, loader, device, score_thresh=args.score_thresh)

        out = fold_dir / f'oof_tau{tag}.json'
        json.dump({'score_thresh': args.score_thresh,
                   'checkpoint': str(ckpt),
                   'checkpoint_epoch': ck.get('epoch'),
                   'n_patients': len(oof),
                   'oof_patients': oof}, open(out, 'w'), indent=2, default=str)

        pm = compute_patient_metrics(oof) if oof else None
        per_fold[k + 1] = pm
        merged.update(oof)
        auc = f'{pm["auc"]:.4f}' if pm else 'n/a'
        print(f'  fold {k + 1}: {ckpt.name} (epoch {ck.get("epoch", "?")}) -> '
              f'{len(oof)} patients, AUC {auc}  [{out.name}]')

        del model, loader, ds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not merged:
        raise SystemExit('\n  nothing collected; no fold checkpoints found.')

    y = [d['label'] for d in merged.values()]
    p = [d['prob'] for d in merged.values()]
    pm = compute_patient_metrics(merged)
    ci = bootstrap_auc_ci(y, p, seed=args.seed)

    print(f'\n{"=" * 74}')
    print(f'  POOLED OOF at tau_det = {args.score_thresh}  '
          f'({len(merged)} patients, {len(per_fold)} folds)')
    print(f'{"=" * 74}')
    print(f'  AUC          {pm["auc"]:.4f}'
          + (f'   95% CI [{ci[1]:.3f}, {ci[2]:.3f}]' if ci else ''))
    print(f'  Sensitivity  {pm["sensitivity"]:.4f}')
    print(f'  Specificity  {pm["specificity"]:.4f}')
    print(f'  J            {pm["sensitivity"] + pm["specificity"] - 1:+.4f}')

    aucs = [m['auc'] for m in per_fold.values() if m]
    if len(aucs) > 1:
        print(f'\n  Per-fold AUC mean +/- SD  {np.mean(aucs):.4f} '
              f'+/- {np.std(aucs, ddof=1):.4f}   '
              f'({", ".join(f"{a:.3f}" for a in aucs)})')

    summary = save_dir / f'oof_pooled_tau{tag}.json'
    json.dump({'score_thresh': args.score_thresh,
               'n_patients': len(merged), 'n_folds': len(per_fold),
               'pooled': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                          for k, v in pm.items()},
               'auc_ci95': [ci[1], ci[2]] if ci else None,
               'per_fold_auc': {k: (m['auc'] if m else None)
                                for k, m in per_fold.items()}},
              open(summary, 'w'), indent=2, default=str)
    print(f'\n  written: {summary}')


if __name__ == '__main__':
    main()
