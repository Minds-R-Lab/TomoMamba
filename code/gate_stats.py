#!/usr/bin/env python3
"""
gate_stats.py  ->  gate-behaviour statistics at lesion and background positions.

Statements like "Mamba writes lesion slices more
aggressively into memory" and "the gate preserves lesion channels" are stronger
than the evidence supports, and asked specifically for:
    (a) how lesion and background positions were selected,
    (b) mean AND variance of gate values,
    (c) a statistical test of the difference,
    (d) whether the same selectivity appears in the BiGRU baseline.

This script produces all four. Inference only, no training.

WHAT IS MEASURED
----------------
MambaCrossSlicePropagation computes
    gate = Sigmoid(Linear([x, ssm_out]))        # (positions, slices, channels)
    enhanced = x + gate * out
so the gate is a per-position, per-slice, per-channel value in [0,1] controlling
how much propagated context is written into each feature. We hook it and compare
its value at LESION positions against BACKGROUND positions.

POSITION SELECTION
  lesion     : feature-map cells whose centre falls inside a GROUND-TRUTH box,
               on the annotated slice. Ground truth, not predictions, so the
               measurement is independent of detector quality.
  background : cells at least --margin cells away from every ground-truth box,
               on the same slices, subsampled to match the lesion count.

Reported per volume and pooled: mean, SD, n, Mann-Whitney U (non-parametric, no
normality assumption), and Cohen's d for effect size. A p-value on tens of
thousands of correlated feature cells is easy to make small, so the effect size
is the honest headline, not the p-value.

USAGE
    python gate_stats.py --checkpoint /mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_model.pt
    # optional BiGRU control, if a baseline-3 checkpoint exists:
    python gate_stats.py --checkpoint ... --bigru_ckpt /mnt/e/DBT_Stage2_Baseline_3/best_model.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from torch.amp import autocast
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_ROOT = '/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15'
SPATIAL = 384
STRIDE = 4                      # feature map is 96x96 for a 384 input
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
KEEP_T, BLOCK_T = 0.7, 0.3      # same thresholds mamba_activations.py used


def load_cases(data_root, split):
    out = []
    for cls in ['Benign', 'Cancer']:
        cdir = Path(data_root) / split / cls
        mdir = Path(data_root) / 'metadata' / split / cls
        if not cdir.exists():
            continue
        for npy in sorted(cdir.glob('*.npy')):
            meta = mdir / f'{npy.stem}.json'
            if not meta.exists():
                meta = mdir / f'{npy.stem.lower()}.json'
            out.append({'npy': str(npy), 'meta': str(meta) if meta.exists() else None,
                        'case_id': npy.stem, 'cls': cls})
    return out


def load_boxes(meta_path, n_slices):
    """Normalised boxes, same parsing as eval_official.load_gt_boxes."""
    boxes = []
    if not meta_path:
        return boxes
    with open(meta_path) as f:
        meta = json.load(f)
    for b in meta.get('boxes', []):
        s = b.get('slice_idx', b.get('mapped_slice', 0))
        s = max(0, min(int(s), n_slices - 1))
        x, y, w, h = float(b['x']), float(b['y']), float(b['width']), float(b['height'])
        if x > 1.5:                       # absolute pixels -> normalise
            ow = b.get('orig_width', b.get('image_width', 1024))
            oh = b.get('orig_height', b.get('image_height', 1024))
            x, w = x / ow, w / ow
            y, h = y / oh, h / oh
        boxes.append({'slice': s, 'x': x, 'y': y, 'w': w, 'h': h})
    return boxes


def prep_volume(path):
    vol = np.load(path).astype(np.float32)
    S = vol.shape[0]
    if vol.shape[1] != SPATIAL:
        r = np.zeros((S, SPATIAL, SPATIAL), dtype=np.float32)
        for s in range(S):
            r[s] = cv2.resize(vol[s], (SPATIAL, SPATIAL))
        vol = r
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi - lo > 1e-6:
        vol = (vol - lo) / (hi - lo)
    return np.clip(vol, 0, 1)


def find_gate_module(model):
    """Return the module holding a .gate submodule, or None."""
    for name, m in model.named_modules():
        if hasattr(m, 'gate') and isinstance(getattr(m, 'gate'), torch.nn.Module):
            return name, m
    return None, None


def collect(model, cases, margin, max_cases, seed=0, original=False):
    """Hook the gate and split its values into lesion vs background."""
    name, mod = find_gate_module(model)
    if mod is None:
        return None, None, None
    captured = {}

    def hook(module, inp, out):
        # out: (B*H*W, S, C) in [0,1]
        captured['gate'] = out.detach().float().cpu()

    h = mod.gate.register_forward_hook(hook)
    rng = np.random.default_rng(seed)
    les_all, bg_all, per_case = [], [], []
    keep_all, keepbg_all = [], []
    std_all, stdbg_all = [], []
    blk_all, blkbg_all = [], []
    act_all, actbg_all = [], []
    rng_all, rngbg_all = [], []

    with torch.no_grad():
        for i, c in enumerate(cases[:max_cases]):
            vol = prep_volume(c['npy'])
            S = vol.shape[0]
            boxes = load_boxes(c['meta'], S)
            if not boxes:
                continue
            vt = torch.from_numpy(vol).unsqueeze(0).to(DEV)
            captured.clear()
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                _ = model({'volume': vt})
            if 'gate' not in captured:
                continue
            g = captured['gate']                       # (H*W, S, C) for B=1
            HW, Sg, C = g.shape
            H = W = int(round(HW ** 0.5))
            if H * W != HW:
                continue
            gfull = g.reshape(H, W, Sg, C).numpy()        # (H,W,S,C)
            g = gfull.mean(-1)                            # mean over channels -> (H,W,S)
            g_keep = (gfull > KEEP_T).sum(-1).astype(np.float32)   # channels "kept"
            g_block = (gfull < BLOCK_T).sum(-1).astype(np.float32) # channels "blocked"
            # THE selectivity measure: spread ACROSS the 128 channels at each cell.
            # mamba_activations.py plots exactly this as "Gate selectivity (ch. std)".
            # Averaging over channels (g, above) destroys this signal by construction.
            g_std = gfull.std(-1)
            # Active channels = either tail.
            # Table 2 reports 68/128 at lesions vs 21/128 at background = 3.2x.
            g_active = g_keep + g_block
            # Gate value range across channels (Table 2 reports 0.15-0.85 vs 0.48-0.52)
            g_rng = gfull.max(-1) - gfull.min(-1)

            les_mask = np.zeros((H, W, Sg), dtype=bool)
            near = np.zeros((H, W), dtype=bool)
            if original:
                # mamba_activations.py: ONE cell at the box centre, and background
                # taken from the single cell (5,5) on the same slice.
                b0 = boxes[0]
                cx = min(W - 1, max(0, int((b0['x'] + b0['w'] / 2) * W)))
                cy = min(H - 1, max(0, int((b0['y'] + b0['h'] / 2) * H)))
                s0 = min(b0['slice'], Sg - 1)
                les_mask[cy, cx, s0] = True
                near[:] = True          # exclude everything...
                near[5, 5] = False      # ...except the original background cell
            for b in boxes:
                x0 = int(np.floor(b['x'] * W)); x1 = int(np.ceil((b['x'] + b['w']) * W))
                y0 = int(np.floor(b['y'] * H)); y1 = int(np.ceil((b['y'] + b['h']) * H))
                x0, x1 = max(0, x0), min(W, max(x0 + 1, x1))
                y0, y1 = max(0, y0), min(H, max(y0 + 1, y1))
                s = min(b['slice'], Sg - 1)
                les_mask[y0:y1, x0:x1, s] = True
                near[max(0, y0 - margin):min(H, y1 + margin),
                     max(0, x0 - margin):min(W, x1 + margin)] = True

            slices_used = sorted({min(b['slice'], Sg - 1) for b in boxes})
            les = g[les_mask]
            les_k = g_keep[les_mask]
            les_s = g_std[les_mask]
            les_b = g_block[les_mask]
            les_a = g_active[les_mask]
            les_r = g_rng[les_mask]
            bg_pool, bgk_pool, bgs_pool, bgb_pool = [], [], [], []
            bga_pool, bgr_pool = [], []
            for s in slices_used:
                bg_pool.append(g[:, :, s][~near])
                bgk_pool.append(g_keep[:, :, s][~near])
                bgs_pool.append(g_std[:, :, s][~near])
                bgb_pool.append(g_block[:, :, s][~near])
                bga_pool.append(g_active[:, :, s][~near])
                bgr_pool.append(g_rng[:, :, s][~near])
            if not bg_pool:
                continue
            bg_pool = np.concatenate(bg_pool)
            bgk_pool = np.concatenate(bgk_pool)
            bgs_pool = np.concatenate(bgs_pool)
            bgb_pool = np.concatenate(bgb_pool)
            bga_pool = np.concatenate(bga_pool)
            bgr_pool = np.concatenate(bgr_pool)
            if len(les) == 0 or len(bg_pool) == 0:
                continue
            k = min(len(les), len(bg_pool))
            sel = rng.choice(len(bg_pool), k, replace=False)
            bg, bg_k = bg_pool[sel], bgk_pool[sel]
            bg_s, bg_b = bgs_pool[sel], bgb_pool[sel]
            bg_a, bg_r = bga_pool[sel], bgr_pool[sel]

            keep_all.append(les_k); keepbg_all.append(bg_k)
            std_all.append(les_s); stdbg_all.append(bg_s)
            blk_all.append(les_b); blkbg_all.append(bg_b)
            act_all.append(les_a); actbg_all.append(bg_a)
            rng_all.append(les_r); rngbg_all.append(bg_r)
            les_all.append(les); bg_all.append(bg)
            per_case.append({'case_id': c['case_id'], 'cls': c['cls'],
                             'lesion_mean': float(les.mean()),
                             'bg_mean': float(bg.mean()),
                             'n_lesion': int(len(les))})
            if (i + 1) % 20 == 0:
                print(f"      {i+1} volumes")
    h.remove()
    if not les_all:
        return None
    return {'mean_les': np.concatenate(les_all), 'mean_bg': np.concatenate(bg_all),
            'keep_les': np.concatenate(keep_all), 'keep_bg': np.concatenate(keepbg_all),
            'std_les': np.concatenate(std_all), 'std_bg': np.concatenate(stdbg_all),
            'blk_les': np.concatenate(blk_all), 'blk_bg': np.concatenate(blkbg_all),
            'act_les': np.concatenate(act_all), 'act_bg': np.concatenate(actbg_all),
            'rng_les': np.concatenate(rng_all), 'rng_bg': np.concatenate(rngbg_all),
            'per_case': per_case}


def stat_block(label, les, bg, unit=""):
    """Report one statistic with n, SD, ratio, effect size and a rank test."""
    ratio = les.mean() / max(abs(bg.mean()), 1e-9)
    pooled = np.sqrt((les.var() + bg.var()) / 2)
    d = (les.mean() - bg.mean()) / max(pooled, 1e-9)
    try:
        _, p = mannwhitneyu(les, bg, alternative='two-sided')
        pstr = f"{p:.2e}"
    except Exception:
        pstr = "n/a"
    print(f"    {label}")
    print(f"      lesion {les.mean():8.4f}{unit}  SD {les.std():.4f}   "
          f"background {bg.mean():8.4f}{unit}  SD {bg.std():.4f}")
    print(f"      ratio {ratio:6.3f}x    Cohen's d {d:6.3f}    p {pstr}")
    return {'lesion_mean': float(les.mean()), 'lesion_sd': float(les.std()),
            'bg_mean': float(bg.mean()), 'bg_sd': float(bg.std()),
            'ratio': float(ratio), 'cohens_d': float(d), 'p': pstr}


def report_all(tag, r):
    print(f"\n  {tag}   (n = {len(r['mean_les']):,} lesion cells, "
          f"{len(r['per_case'])} volumes)")
    out = {}
    out['active_channels'] = stat_block(
        "ACTIVE channels (g>0.7 or g<0.3) of 128   <- Table 2: 68 vs 21 = 3.2x",
        r['act_les'], r['act_bg'])
    out['gate_range'] = stat_block(
        "gate range across channels (max-min)      <- Table 2: 0.15-0.85 vs 0.48-0.52",
        r['rng_les'], r['rng_bg'])
    out['channels_above_0.7'] = stat_block(
        "  of which amplified (g > 0.7)            <- Table 2: ~32 vs ~8",
        r['keep_les'], r['keep_bg'])
    out['channel_std'] = stat_block(
        "channel-wise SD                           <- 'selectivity' map in fig. 4",
        r['std_les'], r['std_bg'])
    out['channels_below_0.3'] = stat_block(
        "  of which suppressed (g < 0.3)           <- Table 2: ~36 vs ~13",
        r['blk_les'], r['blk_bg'])
    out['mean_gate'] = stat_block(
        "mean gate value  (NOT a selectivity measure; a selective and an inert\n"
        "                      gate have the same mean. Shown only for completeness.)",
        r['mean_les'], r['mean_bg'])
    higher = sum(1 for c in r['per_case'] if c['lesion_mean'] > c['bg_mean'])
    print(f"      volumes with higher lesion gate mean: {higher}/{len(r['per_case'])}")
    out['volumes_lesion_higher'] = higher
    out['n_volumes'] = len(r['per_case'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='/mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_model.pt')
    ap.add_argument('--bigru_ckpt', default=None,
                    help='Optional baseline-3 (BiGRU) checkpoint for the control.')
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--margin', type=int, default=4,
                    help='Feature cells of clearance required for a background cell.')
    ap.add_argument('--max_cases', type=int, default=200)
    ap.add_argument('--original_protocol', action='store_true',
                    help='Reproduce mamba_activations.py exactly: lesion = the single '
                         'box-centre cell, background = the single cell at (5,5). Use '
                         'this to check the script against the published Table 2.')
    args = ap.parse_args()

    print("=" * 74)
    print("GATE BEHAVIOUR ANALYSIS")
    print("=" * 74)
    if args.original_protocol:
        print("  PROTOCOL: original (mamba_activations.py) - lesion = box-centre cell,")
        print("            background = the single cell at (5,5)")
    else:
        print(f"  positions: lesion = inside GT box on the annotated slice")
        print(f"             background = >= {args.margin} feature cells from any box")
    print(f"  ground-truth boxes are used, so this does not depend on detector quality")

    cases = load_cases(args.data_root, args.split)
    print(f"  volumes: {len(cases)}")

    from centernet_models import MambaCenterNet
    model = MambaCenterNet(num_classes=2, use_mamba=True, spatial_size=SPATIAL).to(DEV)
    ck = torch.load(args.checkpoint, map_location=DEV, weights_only=False)
    st = ck.get('ema_state_dict', ck.get('model_state_dict', ck))
    model.load_state_dict(st, strict=False)
    model.eval()
    gname, _ = find_gate_module(model)
    print(f"  gate module: {gname}")

    print("\n[1/2] TomoMamba...")
    r = collect(model, cases, args.margin, args.max_cases,
                original=args.original_protocol)
    if r is None:
        print("  [fatal] No gate values captured. Check that the model has a gate.")
        sys.exit(1)
    out = {'tomomamba': report_all("TomoMamba (Mamba + gated residual)", r)}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.bigru_ckpt and Path(args.bigru_ckpt).exists():
        print("\n[2/2] BiGRU control...")
        from centernet_baselines import create_baseline
        b = create_baseline(3, num_classes=2, dropout=0.7, spatial_size=SPATIAL).to(DEV)
        ckb = torch.load(args.bigru_ckpt, map_location=DEV, weights_only=False)
        stb = ckb.get('ema_state_dict', ckb.get('model_state_dict', ckb))
        mb, ub = b.load_state_dict(stb, strict=False)
        n_par = sum(1 for _ in b.state_dict())
        print(f"      loaded (epoch {ckb.get('epoch','?')}); "
              f"{len(mb)} missing / {len(ub)} unexpected of {n_par} tensors")
        if len(mb) > 0.5 * n_par:
            print("      [FATAL] Most weights did not load. This model is effectively")
            print("              RANDOM and any gate numbers from it are meaningless.")
            print("              Check that the checkpoint matches baseline 3.")
            sys.exit(1)
        gate_w = b.cross_slice.gate[0].weight
        print(f"      cross_slice.gate weight: mean {gate_w.mean():.4f} "
              f"std {gate_w.std():.4f}  (random init would be ~0.0 / ~0.036)")
        b.eval()
        n, _ = find_gate_module(b)
        if n is None:
            print("  The BiGRU baseline has no comparable gate, so the control cannot")
            print("  be run as a like-for-like comparison.")
            print("  rather than implying the comparison was made.")
        else:
            rb = collect(b, cases, args.margin, args.max_cases,
                         original=args.original_protocol)
            if rb is not None:
                out['bigru'] = report_all("BiGRU control (same gated residual)", rb)
    else:
        print("\n[2/2] BiGRU control SKIPPED (no --bigru_ckpt given).")
        print("      The same selectivity measure is applied to the control")
        print("      appears in the BiGRU baseline. Without it, the response must")
        print("      say the control was not run.")

    p = Path(args.checkpoint).parent / 'gate_stats.json'
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved: {p}")
    print("\n  Note: with tens of thousands of spatially correlated feature cells,")
    print("  a small p-value is nearly guaranteed and means little on its own.")
    print("  Report the effect size and the per-volume consistency alongside it.")
    print("=" * 74)


if __name__ == '__main__':
    main()
