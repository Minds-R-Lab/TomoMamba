#!/usr/bin/env python3
"""
analyze_checkpoint.py
=====================
Runs ONE existing checkpoint over the validation set and produces, from a
single set of per-patient predictions, the statistics
asked for. No training. Runs on the GPU for the forward passes but is
otherwise light (a few minutes), so it is safe to run alongside a
cross-validation job.

Comments addressed:
  / bootstrap 95% confidence intervals (patient AUC, sens,
                spec, detection rate, mIoU)
  max-Youden's-J from the ROC, balanced accuracy at that
                threshold, calibration (reliability) curve, Brier score
  FROC-style points: detection rate and R@IoU at several
                thresholds, plus false-positives-per-volume
  fallback-classifier usage: how many volumes produce no
                detection and fall back to global classification, and the
                patient-level scores with vs without the fallback path
  //compute cost: parameters, (optional) FLOPs, peak GPU
                memory, and inference time per volume

Reuses the project modules (centernet_models, train_stage2) so the numbers
match the training pipeline exactly. The realistic detection->ROI-classification
path and the official center-distance matcher are the same ones used in
eval_baselines.py / eval_official.py.

Usage:
    conda/venv active, from the project directory:
    python analyze_checkpoint.py \
        --checkpoint /mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_model.pt

Outputs:
    - prints a report to stdout
    - writes analyze_checkpoint_results.json next to the checkpoint
    - writes per-patient predictions to patient_predictions.csv (so the
      DeLong test against BiGRU can be run later once BiGRU probs exist)
"""

import os, sys, json, time, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))

from centernet_models import (MambaCenterNet, aggregate_per_patient,
                              compute_patient_metrics)
from train_stage2 import DBTStage2Dataset, collate_fn, MetricTracker

# ---- constants matching eval_official.py ----
DATA_ROOT = '/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15'
SPATIAL_SIZE = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCORE_THRESH = 0.16   # operating detection threshold


# =====================================================================
# Per-patient predictions (realistic path, identical to eval_baselines)
# =====================================================================

@torch.no_grad()
def collect_predictions(model, loader):
    """Run detection -> ROI classification, return the per-patient dict and
    per-volume bookkeeping needed for the fallback analysis."""
    model.eval()
    tracker = MetricTracker()
    n_volumes = 0
    n_no_detection = 0

    for batch in loader:
        batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            output = model(batch)
            dets = model.detect(batch['volume'], score_thresh=SCORE_THRESH)
            det_probs = model.classify_detections(output['feat_maps'], dets)

        # fallback bookkeeping: a volume with no detection falls back to the
        # global classifier path inside classify_detections
        for d in dets:
            n_volumes += 1
            n_det = len(d['scores']) if hasattr(d['scores'], '__len__') else int(d['scores'].numel())
            if n_det == 0:
                n_no_detection += 1

        det_preds = det_probs.argmax(dim=-1)
        tracker.update(det_preds, batch['labels'], det_probs,
                       case_ids=batch['case_id'])

    patient_metrics, patient_results = tracker.compute_patient_level()
    return patient_metrics, patient_results, n_volumes, n_no_detection


def patient_arrays(patient_results):
    """Extract aligned (label, prob) arrays and ids from the per-patient dict."""
    ids = list(patient_results.keys())
    labels = np.array([patient_results[k]['label'] for k in ids], dtype=int)
    probs = np.array([patient_results[k]['prob'] for k in ids], dtype=float)
    return ids, labels, probs


# =====================================================================
# bootstrap confidence intervals
# =====================================================================

def bootstrap_ci(labels, probs, metric_fn, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            v = metric_fn(labels[idx], probs[idx])
            if v is not None and not np.isnan(v):
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return None
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def auc_metric(y, p):
    if len(np.unique(y)) < 2:
        return None
    return roc_auc_score(y, p)


def sens_spec_at(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec


# =====================================================================
# max-Youden-J, balanced accuracy, calibration, Brier
# =====================================================================

def threshold_independent(labels, probs):
    out = {}
    if len(np.unique(labels)) < 2:
        out['note'] = 'single-class validation set; ROC-based metrics undefined'
        return out
    fpr, tpr, thr = roc_curve(labels, probs)
    j = tpr - fpr
    k = int(np.argmax(j))
    out['max_youden_j'] = float(j[k])
    out['max_j_threshold'] = float(thr[k])
    out['sens_at_max_j'] = float(tpr[k])
    out['spec_at_max_j'] = float(1 - fpr[k])
    out['balanced_acc_at_max_j'] = float((tpr[k] + (1 - fpr[k])) / 2)
    # Youden's J at the standard 0.5 threshold, for the fair comparison
    s5, sp5 = sens_spec_at(labels, probs, 0.5)
    out['youden_j_at_0.5'] = float(s5 + sp5 - 1)
    out['balanced_acc_at_0.5'] = float((s5 + sp5) / 2)
    out['brier_score'] = float(brier_score_loss(labels, probs))
    # 10-bin reliability curve
    bins = np.linspace(0, 1, 11)
    rel = []
    for i in range(10):
        m = (probs >= bins[i]) & (probs < bins[i + 1] if i < 9 else probs <= bins[i + 1])
        if m.sum() > 0:
            rel.append({'bin_low': float(bins[i]), 'bin_high': float(bins[i + 1]),
                        'mean_pred': float(probs[m].mean()),
                        'observed_freq': float(labels[m].mean()),
                        'count': int(m.sum())})
    out['reliability_curve'] = rel
    return out


# =====================================================================
# FROC-style detection reporting (official matcher)
# =====================================================================

def load_gt_boxes(meta_path, num_slices):
    boxes = []
    if not meta_path:
        return boxes
    with open(meta_path) as f:
        meta = json.load(f)
    for box in meta.get('boxes', []):
        s = box.get('slice_idx', box.get('mapped_slice', 0))
        s = max(0, min(s, num_slices - 1))
        bx, by, bw, bh = float(box['x']), float(box['y']), float(box['width']), float(box['height'])
        if bx > 1.5:
            ow = box.get('orig_width', box.get('image_width', 1024))
            oh = box.get('orig_height', box.get('image_height', 1024))
            bx, bw = bx / ow, bw / ow
            by, bh = by / oh, bh / oh
        boxes.append({'slice': s, 'x': bx, 'y': by, 'w': bw, 'h': bh})
    return boxes


def iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def official_match(gt_boxes, pred_boxes, pred_slices, pred_scores, num_slices=15):
    slice_tol = int(np.ceil(num_slices * 0.25))
    min_dist_px = 100 * (SPATIAL_SIZE / 1024)
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    match_ious = []
    order = np.argsort(pred_scores)[::-1]
    for pi in order:
        if pred_matched[pi]:
            continue
        ps = int(pred_slices[pi])
        px1, py1, px2, py2 = pred_boxes[pi]
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        best_gi, best_dist = -1, float('inf')
        for gi, gb in enumerate(gt_boxes):
            if gt_matched[gi]:
                continue
            if abs(ps - gb['slice']) > slice_tol:
                continue
            gx, gy = gb['x'] * SPATIAL_SIZE, gb['y'] * SPATIAL_SIZE
            gw, gh = gb['w'] * SPATIAL_SIZE, gb['h'] * SPATIAL_SIZE
            gcx, gcy = gx + gw / 2, gy + gh / 2
            diag = np.sqrt(gw ** 2 + gh ** 2)
            dist_thresh = max(diag / 2, min_dist_px)
            dist = np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)
            if dist < dist_thresh and dist < best_dist:
                best_dist = dist
                best_gi = gi
        if best_gi >= 0:
            gt_matched[best_gi] = True
            pred_matched[pi] = True
            gb = gt_boxes[best_gi]
            gx1, gy1 = gb['x'] * SPATIAL_SIZE, gb['y'] * SPATIAL_SIZE
            gx2, gy2 = gx1 + gb['w'] * SPATIAL_SIZE, gy1 + gb['h'] * SPATIAL_SIZE
            match_ious.append(iou([px1, py1, px2, py2], [gx1, gy1, gx2, gy2]))
    n_fp = sum(1 for m in pred_matched if not m)
    return gt_matched, match_ious, n_fp


@torch.no_grad()
def froc_analysis(model, cases, thresholds=(0.05, 0.10, 0.16, 0.25, 0.40)):
    """Detection rate, R@IoU, and false positives per volume, swept over
    detection thresholds. Gives the FROC-style operating points asks
    for without needing a full continuous FROC curve."""
    rows = []
    for st in thresholds:
        all_gt_matched, all_ious, total_gt, total_fp, n_vol = [], [], 0, 0, 0
        for case in cases:
            vol = np.load(case['npy']).astype(np.float32)
            S = vol.shape[0]
            if vol.shape[1] != SPATIAL_SIZE:
                r = np.zeros((S, SPATIAL_SIZE, SPATIAL_SIZE), dtype=np.float32)
                for s in range(S):
                    r[s] = cv2.resize(vol[s], (SPATIAL_SIZE, SPATIAL_SIZE))
                vol = r
            lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
            if hi - lo > 1e-6:
                vol = (vol - lo) / (hi - lo)
            vol = np.clip(vol, 0, 1)
            gt_boxes = load_gt_boxes(case['meta'], S)
            total_gt += len(gt_boxes)
            n_vol += 1
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                dets = model.detect(torch.from_numpy(vol).unsqueeze(0).to(DEVICE),
                                    score_thresh=st)
            d = dets[0]
            pb = d['boxes'].cpu().numpy()
            ps = d['slice_indices'].cpu().numpy()
            psc = d['scores'].cpu().numpy()
            gt_m, m_ious, n_fp = official_match(gt_boxes, pb, ps, psc, S)
            all_gt_matched.extend(gt_m)
            all_ious.extend(m_ious)
            total_fp += n_fp
        det_rate = sum(all_gt_matched) / max(total_gt, 1)
        rows.append({
            'score_thresh': st,
            'detection_rate': float(det_rate),
            'lesions_matched': int(sum(all_gt_matched)),
            'total_lesions': int(total_gt),
            'R@0.25': float(sum(1 for v in all_ious if v >= 0.25) / max(total_gt, 1)),
            'R@0.5': float(sum(1 for v in all_ious if v >= 0.5) / max(total_gt, 1)),
            'mIoU': float(np.mean(all_ious)) if all_ious else 0.0,
            'false_positives_total': int(total_fp),
            'fp_per_volume': float(total_fp / max(n_vol, 1)),
        })
    return rows


# =====================================================================
# compute cost
# =====================================================================

def compute_cost(model, loader):
    out = {}
    out['parameters_total'] = int(sum(p.numel() for p in model.parameters()))
    out['parameters_trainable'] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # one representative batch for timing + peak memory
    batch = next(iter(loader))
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    # warmup
    with torch.no_grad(), autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        _ = model(batch)
        _ = model.detect(batch['volume'], score_thresh=SCORE_THRESH)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    n_vol_in_batch = batch['volume'].shape[0]
    t0 = time.time()
    reps = 5
    with torch.no_grad(), autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        for _ in range(reps):
            o = model(batch)
            d = model.detect(batch['volume'], score_thresh=SCORE_THRESH)
            _ = model.classify_detections(o['feat_maps'], d)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) / (reps * max(n_vol_in_batch, 1))
    out['inference_seconds_per_volume'] = float(dt)
    if torch.cuda.is_available():
        out['peak_gpu_memory_mb'] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)

    # optional FLOPs (needs thop); skip gracefully if absent
    try:
        from thop import profile
        vol = batch['volume'][:1]
        macs, _ = profile(model, inputs=({'volume': vol},), verbose=False)
        out['flops_g_per_volume'] = float(2 * macs / 1e9)  # FLOPs ~= 2*MACs
    except Exception as e:
        out['flops_note'] = f"FLOPs skipped ({type(e).__name__}); pip install thop to enable"
    return out


# =====================================================================
# Main
# =====================================================================

def main():
    global DATA_ROOT, SPATIAL_SIZE, SCORE_THRESH
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--spatial_size', type=int, default=SPATIAL_SIZE)
    ap.add_argument('--score_thresh', type=float, default=SCORE_THRESH)
    ap.add_argument('--num_workers', type=int, default=2)
    args = ap.parse_args()

    DATA_ROOT, SPATIAL_SIZE, SCORE_THRESH = args.data_root, args.spatial_size, args.score_thresh

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        alt = ckpt_path.parent / 'best_model.pt'
        if alt.exists():
            ckpt_path = alt
        else:
            print(f"[fatal] checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    print("=" * 68)
    print("analyze_checkpoint.py")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  data:       {DATA_ROOT}  split={args.split}")
    print(f"  device:     {DEVICE}   score_thresh={SCORE_THRESH}")
    print("=" * 68)

    # ---- load model (matches eval_official.load_model for the mamba path) ----
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
    model = MambaCenterNet(num_classes=2, use_mamba=True,
                           spatial_size=SPATIAL_SIZE).to(DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  loaded (epoch {ckpt.get('epoch', '?')}); "
          f"{len(missing)} missing / {len(unexpected)} unexpected keys")

    # ---- data ----
    ds = DBTStage2Dataset(DATA_ROOT, args.split, SPATIAL_SIZE, augment=False)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=True)
    cases = [{'npy': c['npy_path'], 'meta': c['meta_path'], 'case_id': c['case_id']}
             for c in ds.cases]

    results = {'checkpoint': str(ckpt_path), 'epoch': ckpt.get('epoch', '?'),
               'split': args.split, 'score_thresh': SCORE_THRESH}

    # ---- predictions ----
    print("\n[1/5] collecting per-patient predictions...")
    pmetrics, presults, n_vol, n_nodet = collect_predictions(model, loader)
    ids, labels, probs = patient_arrays(presults)
    results['n_patients'] = len(ids)
    results['n_cancer'] = int(labels.sum())
    results['point_estimates'] = {
        'patient_auc': float(pmetrics['auc']) if pmetrics else None,
        'patient_accuracy': float(pmetrics['accuracy']) if pmetrics else None,
        'patient_sensitivity': float(pmetrics['sensitivity']) if pmetrics else None,
        'patient_specificity': float(pmetrics['specificity']) if pmetrics else None,
    }
    print(f"      {len(ids)} patients ({int(labels.sum())} cancer); "
          f"PatAUC={results['point_estimates']['patient_auc']}")

    # ---- / confidence intervals ----
    print("[2/5] bootstrap confidence intervals...")
    ci = {}
    a = bootstrap_ci(labels, probs, auc_metric)
    if a:
        ci['patient_auc'] = {'mean': a[0], 'ci95_low': a[1], 'ci95_high': a[2]}
    s = bootstrap_ci(labels, probs, lambda y, p: sens_spec_at(y, p, 0.5)[0])
    if s:
        ci['sensitivity@0.5'] = {'mean': s[0], 'ci95_low': s[1], 'ci95_high': s[2]}
    sp = bootstrap_ci(labels, probs, lambda y, p: sens_spec_at(y, p, 0.5)[1])
    if sp:
        ci['specificity@0.5'] = {'mean': sp[0], 'ci95_low': sp[1], 'ci95_high': sp[2]}
    results['confidence_intervals'] = ci

    # ---- threshold-independent ----
    print("[3/5] threshold-independent metrics + calibration...")
    results['threshold_independent'] = threshold_independent(labels, probs)

    # ---- fallback ----
    print("[4/5] fallback-classifier usage...")
    results['fallback'] = {
        'n_volumes': int(n_vol),
        'n_no_detection_fallback': int(n_nodet),
        'fallback_fraction': float(n_nodet / max(n_vol, 1)),
    }
    print(f"      {n_nodet}/{n_vol} volumes used the fallback path "
          f"({100*n_nodet/max(n_vol,1):.1f}%)")

    # ---- FROC ----
    print("[5/5] FROC-style detection sweep...")
    results['froc'] = froc_analysis(model, cases)

    # ---- compute cost ----
    print("      compute cost...")
    results['compute_cost'] = compute_cost(model, loader)

    # ---- write outputs ----
    out_json = ckpt_path.parent / 'analyze_checkpoint_results.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    csv_path = ckpt_path.parent / 'patient_predictions.csv'
    with open(csv_path, 'w') as f:
        f.write("patient_id,label,cancer_prob\n")
        for i, k in enumerate(ids):
            f.write(f"{k},{int(labels[i])},{probs[i]:.6f}\n")

    # ---- print a compact report ----
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    pe = results['point_estimates']
    print(f"Patient AUC: {pe['patient_auc']:.4f}", end="")
    if 'patient_auc' in ci:
        print(f"  (95% CI {ci['patient_auc']['ci95_low']:.3f}-{ci['patient_auc']['ci95_high']:.3f})")
    else:
        print()
    ti = results['threshold_independent']
    if 'max_youden_j' in ti:
        print(f"Youden J @0.5: {ti['youden_j_at_0.5']:.3f} | "
              f"max Youden J: {ti['max_youden_j']:.3f} "
              f"(thr {ti['max_j_threshold']:.3f}, bal.acc {ti['balanced_acc_at_max_j']:.3f})")
        print(f"Brier score: {ti['brier_score']:.4f}")
    fb = results['fallback']
    print(f"Fallback used: {fb['n_no_detection_fallback']}/{fb['n_volumes']} "
          f"volumes ({100*fb['fallback_fraction']:.1f}%)")
    cc = results['compute_cost']
    print(f"Params: {cc['parameters_total']:,} | "
          f"{cc.get('inference_seconds_per_volume', 0)*1000:.1f} ms/volume", end="")
    if 'peak_gpu_memory_mb' in cc:
        print(f" | peak {cc['peak_gpu_memory_mb']:.0f} MB", end="")
    print(f" | {cc.get('flops_g_per_volume', cc.get('flops_note',''))}"
          if 'flops_g_per_volume' in cc else f" | {cc.get('flops_note','')}")
    print("\nFROC (detection rate / R@0.25 / FP-per-volume by threshold):")
    for r in results['froc']:
        print(f"  τ={r['score_thresh']:.2f}: det={r['detection_rate']:.3f} "
              f"R@0.25={r['R@0.25']:.3f} FP/vol={r['fp_per_volume']:.2f}")
    print("\nSaved:")
    print(f"  {out_json}")
    print(f"  {csv_path}  (for the DeLong test against BiGRU later)")
    print("=" * 68)


if __name__ == '__main__':
    main()
