#!/usr/bin/env python3
"""
dump_baselines.py  ->  per-case prediction dumps for the three controlled
baselines in Table 4, so stats_package.py can run DeLong / max-J /
calibration on them.

WHY THIS SCRIPT EXISTS
    eval_ablations.py can only build MambaCenterNet, so it cannot touch the
    baselines. eval_baselines.py can build them but only prints aggregate
    metrics and never writes per-case probabilities. and both need
    per-case probabilities for BiGRU and the slice-independent model, so this
    fills the gap.

    It reuses eval_ablations.eval_case unchanged, which means the detection
    matcher, autocast scope, box expansion and ROI classification path are
    byte-for-byte the ones that produced the reported numbers. Nothing about
    the evaluation protocol changes; only the model does.

CHECKPOINTS EXPECTED  (same paths eval_baselines.py --compare uses)
    /mnt/e/DBT_Stage2_Baseline_1/   ResNet-18 classifier only
    /mnt/e/DBT_Stage2_Baseline_2/   ResNet-18 + CenterNet (slice-independent)
    /mnt/e/DBT_Stage2_Baseline_3/   ResNet-18 + BiGRU + CenterNet

    It prefers best_patient_auc.pt, falls back to best_model.pt, then
    best_model.pth. That is the same precedence eval_ablations.py uses.

USAGE
    conda activate tomomamba
    cd ~/DBT/MambaCenterNet_v5.2

    python dump_baselines.py --split validation \
        --dump_dir dumps_val_all_locked_20260708 --score_thresh 0.16

    python dump_baselines.py --split test \
        --dump_dir dumps_test_all_locked_20260708 --score_thresh 0.16

    python stats_package.py --dump_dir dumps_val_all_locked_20260708

Baseline 1 has no detection heads. It gets a classification-only record with
n_gt filled in and n_matched / n_fp set to 0, so stats_package.py reports its
AUC and calibration but shows a detection rate of zero, which is correct.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_ablations import (load_dataset_cases, eval_case, aggregate_results,
                            load_boxes, DATA_ROOT, DEVICE)
from centernet_baselines import create_baseline


BASELINES = {
    1: ('resnet18_cls',       Path('/mnt/e/DBT_Stage2_Baseline_1')),
    2: ('slice_independent',  Path('/mnt/e/DBT_Stage2_Baseline_2')),
    3: ('bigru',              Path('/mnt/e/DBT_Stage2_Baseline_3')),
    4: ('transformer',      Path('/mnt/e/DBT_Stage2_Baseline_4')),
}


def find_ckpt(model_dir):
    for fname in ('best_patient_auc.pt', 'best_model.pt', 'best_model.pth'):
        p = model_dir / fname
        if p.exists():
            return p
    return None


def load_baseline(bid, ckpt_path, spatial_size):
    model = create_baseline(bid, num_classes=2, dropout=0.7,
                            spatial_size=spatial_size).to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ck['model_state_dict'] if 'model_state_dict' in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warn] {len(missing)} missing / {len(unexpected)} unexpected "
              f"tensors. Check this before trusting the numbers.")
    else:
        print(f"  checkpoint loaded clean (epoch {ck.get('epoch', '?')})")
    model.eval()
    return model


def eval_case_clsonly(model, case, spatial_size):
    """Baseline 1 path: no detector, so classify the whole volume."""
    import cv2
    volume = np.load(case['npy_path']).astype(np.float32)
    S, H, W = volume.shape
    if H != spatial_size or W != spatial_size:
        rz = np.zeros((S, spatial_size, spatial_size), dtype=np.float32)
        for s in range(S):
            rz[s] = cv2.resize(volume[s], (spatial_size, spatial_size))
        volume = rz
    lo, hi = np.percentile(volume, 1), np.percentile(volume, 99)
    if hi - lo > 1e-6:
        volume = (volume - lo) / (hi - lo)
    volume = np.clip(volume, 0, 1)

    gt_boxes = load_boxes(case['meta_path'], S) if case['meta_path'] else []
    vt = torch.from_numpy(volume).to(DEVICE)

    with torch.no_grad(), autocast(device_type='cuda'):
        out = model({'volume': vt.unsqueeze(0)})
    prob = out['vol_probs'][0, 1].item()

    gt_name = case['class_name']
    pred_name = 'Cancer' if prob > 0.5 else 'Benign'
    return {'case_id': case['case_id'], 'gt_label': gt_name,
            'pred_label': pred_name, 'cancer_prob': prob,
            'correct': pred_name == gt_name,
            'n_gt': len(gt_boxes), 'n_matched': 0, 'n_fp': 0,
            'n_missed': len(gt_boxes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='validation',
                    choices=['validation', 'test'])
    ap.add_argument('--dump_dir', required=True)
    ap.add_argument('--score_thresh', type=float, default=0.16)
    ap.add_argument('--spatial_size', type=int, default=384)
    ap.add_argument('--baselines', nargs='+', type=int, default=[1, 2, 3],
                    choices=[1, 2, 3, 4])
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'  Baseline per-case dumps | split={args.split} | '
          f'score_thresh={args.score_thresh}')
    print(f'  dump_dir = {dump_dir}')
    print('=' * 70)

    cases = load_dataset_cases(DATA_ROOT, split=args.split,
                               spatial_size=args.spatial_size)
    print(f'  {len(cases)} {args.split} cases loaded\n')

    summary = {}
    for bid in args.baselines:
        name, model_dir = BASELINES[bid]
        ckpt = find_ckpt(model_dir)
        if ckpt is None:
            print(f'  SKIP baseline {bid} ({name}): no checkpoint in {model_dir}')
            continue

        print('-' * 60)
        print(f'  Baseline {bid}: {name}')
        print(f'  Checkpoint: {ckpt}')
        model = load_baseline(bid, ckpt, args.spatial_size)

        out_path = dump_dir / f'per_case_{name}.jsonl'
        if out_path.exists():
            out_path.unlink()          # never append to a stale dump

        results = []
        with open(out_path, 'w') as fh:
            for i, case in enumerate(cases):
                if bid == 1:
                    r = eval_case_clsonly(model, case, args.spatial_size)
                else:
                    r = eval_case(model, case, args.spatial_size, DEVICE,
                                  args.score_thresh)
                results.append(r)
                fh.write(json.dumps({
                    'case_id': r['case_id'], 'gt_label': r['gt_label'],
                    'pred_label': r['pred_label'],
                    'cancer_prob': r['cancer_prob'], 'correct': r['correct'],
                    'n_gt': r['n_gt'], 'n_matched': r['n_matched'],
                    'n_fp': r['n_fp'], 'n_missed': r['n_missed']}) + '\n')
                if (i + 1) % 25 == 0:
                    print(f'    {i + 1}/{len(cases)}')

        m = aggregate_results(results)
        summary[name] = m
        print(f'  PatAUC={m["PatAUC"]}  Sens={m["Sens"]}  Spec={m["Spec"]}  '
              f'J={m["Youden_J"]}  DetRecall={m["Det_Recall"]}')
        print(f'  written: {out_path}\n')

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('=' * 70)
    print('  Now run:')
    print(f'    python stats_package.py --dump_dir {dump_dir}')
    print('=' * 70)


if __name__ == '__main__':
    main()
