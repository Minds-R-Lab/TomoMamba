#!/usr/bin/env python3
"""
eval_variant.py  ->  evaluate any checkpoint against any dataset root at any
detection threshold, and write per-case predictions.

WHY IT EXISTS
    eval_ablations.py, dump_baselines.py and seed_sweep.py all resolve the
    dataset through the module-level DATA_ROOT in eval_ablations.py, which
    points at the production dataset. None of them can be pointed at a
    slice-ablation variant. This one takes --data_root.

    It reuses eval_ablations.eval_case unchanged, so the matcher, autocast
    scope, box expansion and ROI classification path are byte-for-byte the
    ones that produced the published numbers. Only the data and the weights
    change.

WHY THIS MATTERS FOR THE ABLATION
    The per-epoch numbers in training_history.json come from eval_epoch, which
    hardcodes score_thresh=0.20. The reported value is tau_det = 0.16. Comparing
    gradient against random against uniform at 0.20 is internally consistent,
    but it is a different threshold from every other reported number,
    and is specifically about threshold consistency. This lets the
    ablation table be reported at 0.16 like everything else.

USAGE
    # the three slice-selection variants, all at the operating threshold
    python eval_variant.py --tag gradient \
        --checkpoint /mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_patient_auc.pt \
        --data_root /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15

    python eval_variant.py --tag random \
        --checkpoint /mnt/e/DBT_Stage2_ablation_sliceRandom/best_patient_auc.pt \
        --data_root /mnt/e/DBT_SliceAblation_random_1024_15

    python eval_variant.py --tag uniform \
        --checkpoint /mnt/e/DBT_Stage2_ablation_sliceUniform/best_patient_auc.pt \
        --data_root /mnt/e/DBT_SliceAblation_uniform_1024_15

    # then the full statistics on all three together
    python stats_package.py --dump_dir dumps_slice_ablation --ref gradient

NOTE ON THE COMPARISON
    Each variant model is evaluated on ITS OWN validation split, i.e. the one
    built with the same slice-selection rule it was trained under. That is the
    correct matched comparison: it asks which selection strategy produces the
    better model, not how a model copes with a selection rule it never saw.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--tag', required=True,
                    help='name for the dump file, e.g. gradient / random')
    ap.add_argument('--split', default='validation',
                    choices=['train', 'validation', 'test'])
    ap.add_argument('--dump_dir', default='dumps_slice_ablation')
    ap.add_argument('--score_thresh', type=float, default=0.16)
    ap.add_argument('--spatial_size', type=int, default=384)
    ap.add_argument('--dropout', type=float, default=0.7)
    ap.add_argument('--backbone', default='resnet18')
    ap.add_argument('--no_mamba', action='store_true')
    args = ap.parse_args()

    import torch
    from eval_ablations import (load_dataset_cases, eval_case,
                                aggregate_results, DEVICE)
    from centernet_models import MambaCenterNet

    root = Path(args.data_root)
    if not root.exists():
        raise SystemExit(f'data_root not found: {root}')
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f'checkpoint not found: {ckpt}')

    print('=' * 74)
    print(f'  EVAL VARIANT  |  tag = {args.tag}  |  split = {args.split}')
    print(f'  data       : {root}')
    print(f'  checkpoint : {ckpt}')
    print(f'  tau_det    : {args.score_thresh}')
    print('=' * 74)

    # first positional of load_dataset_cases is the root, so any dataset works
    cases = load_dataset_cases(str(root), split=args.split,
                               spatial_size=args.spatial_size)
    if not cases:
        raise SystemExit(f'no {args.split} cases under {root}')
    print(f'  {len(cases)} cases loaded')

    model = MambaCenterNet(num_classes=2, dropout=args.dropout,
                           use_mamba=not args.no_mamba,
                           spatial_size=args.spatial_size,
                           backbone=args.backbone).to(DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(
        ck.get('model_state_dict', ck), strict=False)
    if missing or unexpected:
        print(f'  [warn] {len(missing)} missing / {len(unexpected)} unexpected '
              f'tensors -- check this before trusting the numbers')
    else:
        print(f'  checkpoint loaded clean (epoch {ck.get("epoch", "?")})')
    model.eval()

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / f'per_case_{args.tag}.jsonl'
    if out.exists():
        out.unlink()                      # never append to a stale dump

    results = []
    with open(out, 'w') as fh:
        for i, case in enumerate(cases):
            r = eval_case(model, case, args.spatial_size, DEVICE,
                          args.score_thresh)
            results.append(r)
            fh.write(json.dumps({k: r[k] for k in
                                 ('case_id', 'gt_label', 'pred_label',
                                  'cancer_prob', 'correct', 'n_gt',
                                  'n_matched', 'n_fp', 'n_missed')}) + '\n')
            if (i + 1) % 25 == 0:
                print(f'    {i + 1}/{len(cases)}')

    m = aggregate_results(results)
    print(f'\n  PatAUC={m["PatAUC"]}  Sens={m["Sens"]}  Spec={m["Spec"]}  '
          f'J={m["Youden_J"]}  DetRecall={m["Det_Recall"]}')
    print(f'  written: {out}')
    print(f'\n  when all variants are dumped:')
    print(f'    python stats_package.py --dump_dir {dump_dir} --ref gradient')


if __name__ == '__main__':
    main()
