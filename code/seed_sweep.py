#!/usr/bin/env python3
"""
seed_sweep.py  ->  train TomoMamba with several random seeds on the FIXED
published split, evaluate each at the operating tau_det, and report
mean +/- SD with a statistical analysis. This is Yalda's request for
and .

WHAT IT DOES, IN THREE PHASES
    train   for each seed, run train_stage2.py into its own save_dir.
            A seed whose training_history.json already exists is SKIPPED,
            so an interrupted sweep resumes by rerunning the same command.
    dump    for each finished seed, write per-case predictions at
            --score_thresh (default 0.16, the operating tau_det) into one
            shared directory as per_case_seed{S}.jsonl.
    report  mean +/- SD across seeds for AUC, sensitivity, specificity,
            Youden J and detection rate, plus a paired bootstrap of each
            seed against the reference seed.

    Run all three with no flags. Run one with --phase.

SEED 42 IS FREE
    The published checkpoint at /mnt/e/DBT_Stage2_MambaCenterNet_v5.2 was
    trained with the default seed of 42, so it already counts as one of the
    five. Point --seed42_dir at it and only four new runs are needed.

TIME BUDGET, BE REALISTIC
    At the epoch times seen in the July runs (roughly 20-30 min per epoch),
    a run with --epochs 25 --patience 8 takes about 8 to 12 hours. Four new
    seeds is therefore 32 to 48 hours of uninterrupted GPU. Historical best
    epochs were 6 to 18, so a cap of 25 with patience 8 cannot truncate a run
    that would have stopped on its own.

    Launch it detached so a closed terminal cannot kill it:

    setsid python -u seed_sweep.py --epochs 25 --patience 8 \
        > /mnt/e/seed_sweep.log 2>&1 < /dev/null &

REPORTING NOTE
    Report every seed that runs. If one comes out poorly, that variance IS
    the finding and belongs in the table; the spread is what asked to
    see. Dropping a seed after seeing its result makes the reported SD
    smaller than the true SD.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------- phase: train

def train_one(seed, out_dir, args):
    hist = out_dir / 'training_history.json'
    if hist.exists():
        print(f'  seed {seed}: already trained ({out_dir}), skipping')
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-u', str(HERE / 'train_stage2.py'),
           '--data_root', args.data_root,
           '--save_dir', str(out_dir),
           '--seed', str(seed),
           '--epochs', str(args.epochs),
           '--patience', str(args.patience),
           '--spatial_size', str(args.spatial_size),
           '--batch_size', str(args.batch_size)]
    print(f'\n  seed {seed}: training -> {out_dir}')
    print('    ' + ' '.join(cmd))
    t0 = time.time()
    log = out_dir / 'train.log'
    with open(log, 'w') as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    mins = (time.time() - t0) / 60
    if r.returncode != 0:
        print(f'    FAILED after {mins:.1f} min, exit {r.returncode}. '
              f'See {log}')
        return False
    print(f'    done in {mins:.1f} min ({mins/60:.1f} h)')
    return True


# ----------------------------------------------------------------- phase: dump

def dump_one(seed, model_dir, dump_dir, args):
    import torch
    from eval_ablations import load_dataset_cases, eval_case, DATA_ROOT, DEVICE
    from centernet_models import MambaCenterNet

    out = dump_dir / f'per_case_seed{seed}.jsonl'
    if out.exists() and not args.force:
        print(f'  seed {seed}: dump exists, skipping ({out.name})')
        return True

    ckpt = None
    for name in ('best_patient_auc.pt', 'best_model.pt'):
        if (model_dir / name).exists():
            ckpt = model_dir / name
            break
    if ckpt is None:
        print(f'  seed {seed}: no checkpoint in {model_dir}, skipping')
        return False

    cases = load_dataset_cases(DATA_ROOT, split=args.split,
                               spatial_size=args.spatial_size)
    model = MambaCenterNet(num_classes=2, dropout=0.7, use_mamba=True,
                           spatial_size=args.spatial_size,
                           backbone='resnet18').to(DEVICE)
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    missing, unexpected = model.load_state_dict(
        ck.get('model_state_dict', ck), strict=False)
    if missing or unexpected:
        print(f'  seed {seed}: [warn] {len(missing)} missing / '
              f'{len(unexpected)} unexpected tensors')
    model.eval()

    dump_dir.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as fh:
        for i, case in enumerate(cases):
            r = eval_case(model, case, args.spatial_size, DEVICE,
                          args.score_thresh)
            fh.write(json.dumps({k: r[k] for k in
                                 ('case_id', 'gt_label', 'pred_label',
                                  'cancer_prob', 'correct', 'n_gt',
                                  'n_matched', 'n_fp', 'n_missed')}) + '\n')
            if (i + 1) % 25 == 0:
                print(f'    {i + 1}/{len(cases)}')
    print(f'  seed {seed}: {ckpt.name} (epoch {ck.get("epoch", "?")}) '
          f'-> {out.name}')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


# --------------------------------------------------------------- phase: report

def patient_level(path, tau=0.50):
    from sklearn.metrics import roc_auc_score
    by, lab = defaultdict(list), {}
    n_gt = n_matched = n_fp = n_vol = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pid = r['case_id'].split('_')[0]
        by[pid].append(float(r['cancer_prob']))
        lab[pid] = 1 if r['gt_label'] == 'Cancer' else 0
        n_gt += r.get('n_gt', 0); n_matched += r.get('n_matched', 0)
        n_fp += r.get('n_fp', 0); n_vol += 1
    pids = sorted(by)
    y = np.array([lab[p] for p in pids])
    p = np.array([max(by[q]) for q in pids])
    pred = (p > tau).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {'pids': pids, 'y': y, 'p': p,
            'auc': roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            'sens': sens, 'spec': spec, 'J': sens + spec - 1,
            'det_rate': n_matched / n_gt if n_gt else np.nan,
            'fp_per_vol': n_fp / n_vol if n_vol else np.nan}


def paired_boot_auc(y, pa, pb, n_boot=4000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n, d = len(y), []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    d = np.array(d)
    if len(d) == 0:
        return None
    lo, hi = np.percentile(d, [2.5, 97.5])
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(np.mean(d)), float(lo), float(hi), float(min(pv, 1.0))


def report(dump_dir, seeds, ref_seed, args):
    rows = {}
    for s in seeds:
        f = dump_dir / f'per_case_seed{s}.jsonl'
        if f.exists():
            rows[s] = patient_level(f, args.tau_cls)
    if not rows:
        print('\n  nothing to report; no dumps found.')
        return

    print('\n' + '=' * 78)
    print(f'  SEED SWEEP  |  split={args.split}  |  tau_det={args.score_thresh}'
          f'  |  tau_cls={args.tau_cls}')
    print('=' * 78)
    print(f'  {"seed":<8}{"AUC":>9}{"Sens":>8}{"Spec":>8}{"J":>9}'
          f'{"DetRate":>10}{"FP/vol":>9}')
    for s, m in rows.items():
        print(f'  {s:<8}{m["auc"]:>9.4f}{m["sens"]:>8.3f}{m["spec"]:>8.3f}'
              f'{m["J"]:>+9.3f}{m["det_rate"]:>10.3f}{m["fp_per_vol"]:>9.2f}')

    print(f'\n  Across {len(rows)} seeds, mean +/- SD:')
    for key, label in (('auc', 'Patient AUC'), ('sens', 'Sensitivity'),
                       ('spec', 'Specificity'), ('J', "Youden's J"),
                       ('det_rate', 'Detection rate')):
        v = np.array([m[key] for m in rows.values()], dtype=float)
        sd = np.std(v, ddof=1) if len(v) > 1 else 0.0
        print(f'    {label:<16}{np.mean(v):.4f} +/- {sd:.4f}'
              f'   (min {v.min():.4f}, max {v.max():.4f})')

    if ref_seed in rows and len(rows) > 1:
        print(f'\n  Paired bootstrap of AUC against seed {ref_seed}:')
        print(f'  {"seed":<8}{"dAUC":>9}{"95% CI":>22}{"p":>9}')
        r = rows[ref_seed]
        for s, m in rows.items():
            if s == ref_seed:
                continue
            b = paired_boot_auc(r['y'], r['p'], m['p'], args.n_boot)
            if b is None:
                continue
            mn, lo, hi, pv = b
            print(f'  {s:<8}{mn:>+9.4f}'
                  f'{f"[{lo:+.3f}, {hi:+.3f}]":>22}{pv:>9.3f}')

    out = dump_dir / f'seed_sweep_{args.split}.json'
    json.dump({s: {k: (float(v) if isinstance(v, (int, float, np.floating))
                       else None)
                   for k, v in m.items() if k not in ('pids', 'y', 'p')}
               for s, m in rows.items()}, open(out, 'w'), indent=2)
    print(f'\n  written: {out}')


# --------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int,
                    default=[42, 1337, 2024, 7, 99])
    ap.add_argument('--ref_seed', type=int, default=42)
    ap.add_argument('--seed42_dir',
                    default='/mnt/e/DBT_Stage2_MambaCenterNet_v5.2',
                    help='reuse the published run instead of retraining seed 42')
    ap.add_argument('--root', default='/mnt/e/DBT_Stage2_seeds')
    ap.add_argument('--dump_dir', default='dumps_seeds')
    ap.add_argument('--data_root',
                    default='/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15')
    ap.add_argument('--phase', default='all',
                    choices=['all', 'train', 'dump', 'report'])
    ap.add_argument('--split', default='validation',
                    choices=['train', 'validation', 'test'])
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--spatial_size', type=int, default=384)
    ap.add_argument('--score_thresh', type=float, default=0.16)
    ap.add_argument('--tau_cls', type=float, default=0.50)
    ap.add_argument('--n_boot', type=int, default=4000)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    root = Path(args.root)
    dump_dir = Path(args.dump_dir)

    def dir_for(seed):
        if seed == 42 and args.seed42_dir and Path(args.seed42_dir).exists():
            return Path(args.seed42_dir)
        return root / f'seed_{seed}'

    print('=' * 78)
    print(f'  SEED SWEEP  |  seeds {args.seeds}  |  phase {args.phase}')
    print(f'  epochs {args.epochs}, patience {args.patience}')
    print('=' * 78)

    if args.phase in ('all', 'train'):
        for s in args.seeds:
            d = dir_for(s)
            if s == 42 and d == Path(args.seed42_dir):
                print(f'  seed 42: reusing published run at {d}')
                continue
            train_one(s, d, args)

    if args.phase in ('all', 'dump'):
        print('\n  --- dumping per-case predictions ---')
        for s in args.seeds:
            dump_one(s, dir_for(s), dump_dir, args)

    if args.phase in ('all', 'report'):
        report(dump_dir, args.seeds, args.ref_seed, args)


if __name__ == '__main__':
    main()
