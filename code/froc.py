#!/usr/bin/env python3
"""
froc.py  ->  FROC analysis: sensitivity versus false positives per volume.

asks for FROC curves, sensitivity at fixed false positives per volume,
total false positives, lesion-wise sensitivity, performance at several IoU
thresholds, and an explanation of whether unmatched detections contribute to
the reported metrics.

TWO MODES

  --generate   sweeps the detection threshold and records, per volume and per
               threshold: how many ground-truth lesions were matched, how many
               predictions matched nothing, and the IoU of every matched pair.
               Needs the GPU, but only detection, so it is FAST: roughly 90
               seconds per model for 75 volumes across 8 thresholds. Writes
               froc_data_<tag>.json.

  --plot       reads one or more froc_data files and produces the curve, the
               sensitivity-at-fixed-FP table, and the multi-IoU table.
               CPU only, seconds.

WHY IT IS SEPARATE FROM THE EXISTING SWEEP
    analyze_checkpoint.py already gives four points on the curve. That is
    enough to quote numbers but not enough to draw a defensible FROC, and it
    records no IoUs, so "performance at several IoU thresholds" cannot be
    answered from it.

ON UNMATCHED DETECTIONS
    The reported detection rate counts matched ground-truth lesions only,
    so unmatched predictions do not enter it at all. That is exactly the gap
    identifies: a detector can score well on detection rate while
    producing many false positives. This script reports both halves, so the
    trade is visible.

USAGE
    # after the seed sweep frees the GPU, or alongside it (needs ~4 GB)
    python froc.py --generate \
        --checkpoint /mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_patient_auc.pt \
        --tag tomomamba_val --split validation

    python froc.py --plot --data froc_data_tomomamba_val.json \
        --out froc_val.pdf
"""

import json
import argparse
from pathlib import Path

import numpy as np


DEFAULT_TAUS = [0.03, 0.05, 0.08, 0.10, 0.14, 0.16, 0.20, 0.25, 0.32, 0.40]
STANDARD_FP = [0.25, 0.5, 1.0, 2.0, 4.0]
IOU_LEVELS = [0.1, 0.25, 0.5, 0.75]


# ------------------------------------------------------------------ geometry

def iou(a, b):
    """Boxes as (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(gt, preds, scores, pred_slices, gt_slices, spatial_size,
                 dist_floor=37.5, slice_frac=0.25, n_slices=15):
    """Matching criteria: centre distance within
    max(GT half-diagonal, dist_floor), and slice within +/- slice_frac of depth.
    Predictions are assigned to ground truth in descending score order.
    Returns (matched_flags_per_gt, iou_per_matched_gt, n_unmatched_preds)."""
    order = np.argsort(-np.asarray(scores))
    taken = set()
    matched = [False] * len(gt)
    ious = [0.0] * len(gt)
    slice_tol = max(1, int(round(slice_frac * n_slices)))
    for pi in order:
        pb = preds[pi]
        pcx, pcy = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
        best, best_j = -1.0, None
        for gj, gb in enumerate(gt):
            if matched[gj]:
                continue
            gcx, gcy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
            half_diag = 0.5 * np.hypot(gb[2] - gb[0], gb[3] - gb[1])
            radius = max(half_diag, dist_floor)
            if np.hypot(pcx - gcx, pcy - gcy) > radius:
                continue
            if abs(int(pred_slices[pi]) - int(gt_slices[gj])) > slice_tol:
                continue
            v = iou(pb, gb)
            if v > best:
                best, best_j = v, gj
        if best_j is not None:
            matched[best_j] = True
            ious[best_j] = max(best, 0.0)
            taken.add(int(pi))
    return matched, ious, len(preds) - len(taken)


# ------------------------------------------------------------------ generate

def generate(args):
    import sys
    import torch
    from torch.amp import autocast
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_ablations import load_dataset_cases, load_boxes, DATA_ROOT, DEVICE
    from centernet_models import MambaCenterNet

    cases = load_dataset_cases(DATA_ROOT, split=args.split,
                               spatial_size=args.spatial_size)
    model = MambaCenterNet(num_classes=2, dropout=0.7, use_mamba=not args.no_mamba,
                           spatial_size=args.spatial_size,
                           backbone='resnet18').to(DEVICE)
    ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    miss, unexp = model.load_state_dict(ck.get('model_state_dict', ck),
                                       strict=False)
    if miss or unexp:
        print(f'  [warn] {len(miss)} missing / {len(unexp)} unexpected tensors')
    model.eval()
    print(f'  checkpoint epoch {ck.get("epoch", "?")} | {len(cases)} volumes | '
          f'{len(args.taus)} thresholds')

    per_tau = {t: {'n_gt': 0, 'n_matched': 0, 'n_fp': 0, 'n_vol': 0,
                   'ious': []} for t in args.taus}

    for i, case in enumerate(cases):
        vol = np.load(case['npy_path']).astype(np.float32)
        S = vol.shape[0]
        if vol.shape[1] != args.spatial_size:
            rz = np.zeros((S, args.spatial_size, args.spatial_size), np.float32)
            for s in range(S):
                rz[s] = cv2.resize(vol[s], (args.spatial_size,) * 2)
            vol = rz
        lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
        if hi - lo > 1e-6:
            vol = (vol - lo) / (hi - lo)
        vol = np.clip(vol, 0, 1)

        boxes = load_boxes(case['meta_path'], S) if case['meta_path'] else []
        if not boxes:
            continue
        # load_boxes returns [s_idx, x, y, w, h] with x,y,w,h normalised 0-1
        # and s_idx already mapped into the 0..S-1 stack index.
        sp = args.spatial_size
        gt = [(b[1] * sp, b[2] * sp, (b[1] + b[3]) * sp, (b[2] + b[4]) * sp)
              for b in boxes]
        gt_slices = [int(b[0]) for b in boxes]

        vt = torch.from_numpy(vol).to(DEVICE)
        for t in args.taus:
            with torch.no_grad(), autocast(device_type='cuda'):
                dets = model.detect(vt.unsqueeze(0), score_thresh=t)[0]
            pb = dets['boxes'].detach().cpu().numpy()
            ps = dets['scores'].detach().cpu().numpy()
            pl = dets['slice_indices'].detach().cpu().numpy()
            matched, ious, n_fp = greedy_match(
                gt, pb, ps, pl, gt_slices, args.spatial_size,
                n_slices=S)
            d = per_tau[t]
            d['n_gt'] += len(gt)
            d['n_matched'] += sum(matched)
            d['n_fp'] += n_fp
            d['n_vol'] += 1
            d['ious'].extend([float(v) for v, m in zip(ious, matched) if m])
        if (i + 1) % 20 == 0:
            print(f'    {i + 1}/{len(cases)}')

    out = Path(f'froc_data_{args.tag}.json')
    json.dump({'tag': args.tag, 'split': args.split,
               'checkpoint': str(args.checkpoint),
               'points': {str(t): per_tau[t] for t in args.taus}},
              open(out, 'w'), indent=2, default=float)
    print(f'  written: {out}')
    return out


# ---------------------------------------------------------------------- plot

def curve(data):
    pts = []
    for t, d in data['points'].items():
        if d['n_gt'] == 0 or d['n_vol'] == 0:
            continue
        pts.append({'tau': float(t),
                    'sens': d['n_matched'] / d['n_gt'],
                    'fp_per_vol': d['n_fp'] / d['n_vol'],
                    'n_fp': d['n_fp'], 'n_matched': d['n_matched'],
                    'n_gt': d['n_gt'], 'ious': d['ious']})
    return sorted(pts, key=lambda p: p['fp_per_vol'])


def sens_at_fp(pts, target):
    """Linear interpolation in FP/volume. None if the curve never reaches it."""
    xs = [p['fp_per_vol'] for p in pts]
    ys = [p['sens'] for p in pts]
    if not xs or target < min(xs):
        return None
    if target >= max(xs):
        return ys[int(np.argmax(xs))]
    return float(np.interp(target, xs, ys))


def plot(args):
    datasets = [json.load(open(p)) for p in args.data]

    print('=' * 78)
    print('  FROC ANALYSIS')
    print('=' * 78)

    for data in datasets:
        pts = curve(data)
        print(f'\n  {data["tag"]}  ({data["split"]})')
        print(f'    {"tau":>6}{"sens":>9}{"FP/vol":>10}{"total FP":>10}'
              f'{"matched":>12}')
        for p in pts:
            frac = f'{p["n_matched"]}/{p["n_gt"]}'
            print(f'    {p["tau"]:>6.2f}{p["sens"]:>9.3f}'
                  f'{p["fp_per_vol"]:>10.2f}{p["n_fp"]:>10d}{frac:>12}')

        print(f'\n    Lesion-wise sensitivity at fixed false positives '
              f'per volume:')
        for f in STANDARD_FP:
            s = sens_at_fp(pts, f)
            print(f'      {f:>5.2f} FP/vol   '
                  + ('not reached' if s is None else f'{s:.3f}'))

        # multi-IoU at the operating point
        ref = min(pts, key=lambda p: abs(p['tau'] - args.tau_ref))
        io = np.array(ref['ious'])
        print(f'\n    At tau = {ref["tau"]:.2f} (detection rate '
              f'{ref["sens"]:.3f}), recall at several IoU thresholds:')
        for lv in IOU_LEVELS:
            n = int((io >= lv).sum())
            print(f'      IoU >= {lv:<5}  {n}/{ref["n_gt"]} = '
                  f'{n / ref["n_gt"]:.3f}')
        if len(io):
            print(f'      mean IoU over matched pairs  {io.mean():.3f}  '
                  f'(median {np.median(io):.3f})')

    if args.out:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=200)
            for data in datasets:
                pts = curve(data)
                ax.plot([p['fp_per_vol'] for p in pts],
                        [p['sens'] for p in pts], marker='o', ms=4,
                        lw=1.6, label=data['tag'])
                for p in pts:
                    if abs(p['tau'] - args.tau_ref) < 1e-9:
                        ax.plot(p['fp_per_vol'], p['sens'], marker='*',
                                ms=15, color='crimson', zorder=5)
            ax.set_xscale('log')
            ax.set_xlabel('False positives per volume')
            ax.set_ylabel('Lesion-wise sensitivity')
            ax.set_ylim(0, 1.0)
            ax.grid(alpha=0.3, which='both')
            ax.legend(fontsize=8, loc='lower right')
            ax.set_title('FROC, Stage 2 lesion detection', fontsize=10)
            fig.tight_layout()
            fig.savefig(args.out, bbox_inches='tight')
            fig.savefig(str(Path(args.out).with_suffix('.png')),
                        bbox_inches='tight')
            print(f'\n  figure: {args.out}  (star = operating point '
                  f'tau = {args.tau_ref})')
        except Exception as e:
            print(f'\n  [warn] plotting failed ({e}); the tables above are '
                  f'still valid')

    print(f'\n  Note: the detection rate counts matched')
    print(f'  ground-truth lesions only, so unmatched predictions do not')
    print(f'  enter it. The FP/volume column is the other half of the trade')
    print(f'  and should be reported alongside it. This is what asks.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generate', action='store_true')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--checkpoint',
                    default='/mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_patient_auc.pt')
    ap.add_argument('--tag', default='tomomamba_val')
    ap.add_argument('--split', default='validation',
                    choices=['validation', 'test'])
    ap.add_argument('--spatial_size', type=int, default=384)
    ap.add_argument('--no_mamba', action='store_true')
    ap.add_argument('--taus', nargs='+', type=float, default=DEFAULT_TAUS)
    ap.add_argument('--data', nargs='+', default=None)
    ap.add_argument('--tau_ref', type=float, default=0.16)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.generate:
        p = generate(args)
        if not args.data:
            args.data = [str(p)]
    if args.plot or (args.generate and args.data):
        if not args.data:
            raise SystemExit('--plot needs --data')
        plot(args)
    if not args.generate and not args.plot:
        raise SystemExit('pass --generate, --plot, or both')


if __name__ == '__main__':
    main()
