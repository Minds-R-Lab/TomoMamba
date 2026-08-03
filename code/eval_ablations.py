#!/usr/bin/env python3
"""
Validation-Only Runner for MambaCenterNet v5.2 Ablations
=========================================================
Uses EXACT same pipeline as visualize.py (same data loading,
same autocast scope, same patient grouping) — just no images.

USAGE:
    python eval_ablations.py
    python eval_ablations.py --config baseline
    python eval_ablations.py --config no_cls_upweight
    python eval_ablations.py --score_thresh 0.20
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
import cv2

PROJECT_DIR = Path(os.path.expanduser("~/DBT/MambaCenterNet_v5.2"))
sys.path.insert(0, str(PROJECT_DIR))

from centernet_models import MambaCenterNet

ABLATION_ROOT = Path("/mnt/e/DBT_Stage2_ablations")
BASELINE_DIR  = Path("/mnt/e/DBT_Stage2_MambaCenterNet_v5.2")
DATA_ROOT     = Path("/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15")

ALL_CONFIGS = [
    # "no_cls_upweight", "equal_weights", "no_peak_reg", "high_cls_weight",
    # "no_copypaste", "roi_5x5", "roi_9x9",
    "no_gate", "no_fpn"
]

CATEGORIES = {
    # "Loss ablations":   ["no_cls_upweight", "equal_weights", "no_peak_reg", "high_cls_weight"],
    # "Design ablations": ["no_copypaste", "roi_5x5", "roi_9x9"],
    "Architecture ablations": ["no_gate", "no_fpn"]
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
#  Per-case dump (added by patch_eval_ablations.py)
# =====================================================================
_DUMP_DIR = None  # set by main() when --dump_dir is used

def _dump_case(dump_dir, config_name, r):
    import json as _json
    from pathlib import Path as _Path
    safe = config_name.replace(' ', '_').replace('/', '_')
    out = _Path(dump_dir) / f'per_case_{safe}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        'case_id':      r.get('case_id'),
        'gt_label':     r.get('gt_label'),
        'pred_label':   r.get('pred_label'),
        'cancer_prob':  r.get('cancer_prob'),
        'correct':      r.get('correct'),
        'n_gt':         r.get('n_gt'),
        'n_matched':    r.get('n_matched'),
        'n_fp':         r.get('n_fp'),
        'n_missed':     r.get('n_missed'),
    }
    with open(out, 'a') as f:
        f.write(_json.dumps(row) + '\n')


# =====================================================================
#  Data loading — EXACT copy from visualize.py
# =====================================================================

def load_dataset_cases(data_root, split='validation', spatial_size=384):
    data_root = Path(data_root)
    split_dir = data_root / split
    meta_dir = data_root / 'metadata' / split
    cases = []
    CLASS_MAP = {'Benign': 0, 'Cancer': 1}
    for class_name in ['Benign', 'Cancer']:
        class_dir = split_dir / class_name
        meta_class_dir = meta_dir / class_name
        if not class_dir.exists():
            continue
        for npy_path in sorted(class_dir.glob("*.npy")):
            meta_path = meta_class_dir / f"{npy_path.stem}.json"
            if not meta_path.exists():
                meta_path = meta_class_dir / f"{npy_path.stem.lower()}.json"
            if not meta_path.exists():
                meta_path = None
            case_id = npy_path.stem
            study_id = '_'.join(case_id.split('_')[:2])
            cases.append({
                'npy_path': str(npy_path),
                'meta_path': str(meta_path) if meta_path else None,
                'class_name': class_name,
                'class_idx': CLASS_MAP[class_name],
                'case_id': case_id,
                'study_id': study_id,
            })
    return cases


def load_boxes(meta_path, num_slices):
    boxes = []
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for box in meta.get('boxes', []):
            orig_slice = box.get('slice', box.get('original_slice',
                                 box.get('slice_idx', -1)))
            if 'mapped_slice' in box:
                s_idx = box['mapped_slice']
            elif 'slice_idx' in box:
                s_idx = box['slice_idx']
            else:
                s_idx = int(round(orig_slice / max(1, box.get('total_slices', num_slices)) * (num_slices - 1)))
            s_idx = max(0, min(s_idx, num_slices - 1))
            bx = float(box['x'])
            by = float(box['y'])
            bw = float(box['width'])
            bh = float(box['height'])
            if bx > 1.5:
                orig_w = box.get('orig_width', box.get('image_width', 1024))
                orig_h = box.get('orig_height', box.get('image_height', 1024))
                bx, bw = bx / orig_w, bw / orig_w
                by, bh = by / orig_h, bh / orig_h
            boxes.append([s_idx, bx, by, bw, bh])
    return boxes


def match_predictions_to_gt(gt_boxes, pred_boxes, pred_slices, spatial_size,
                             num_slices=15):
    """BCS-DBT official spatial matching (paper Section 4).

    - Slice tolerance: +/- ceil(num_slices * 0.25)  (== 4 for 15 slices)
    - Distance floor:  spatial_size / 1024 * 100    (== 37.5 for spatial=384)
      i.e. d_match = 100 px at the 1024x1024 intermediate resolution,
      scaled to the model's operating resolution.
    - Predicted center must fall within max(GT_diag/2, dist_floor) of the
      GT center. Greedy assignment: nearest surviving prediction wins.
    """
    import math
    slice_tol = int(math.ceil(num_slices * 0.25))
    dist_floor = spatial_size * 100.0 / 1024.0

    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    for gi, gb in enumerate(gt_boxes):
        gt_s = int(gb[0])
        gx = gb[1] * spatial_size
        gy = gb[2] * spatial_size
        gw = gb[3] * spatial_size
        gh = gb[4] * spatial_size
        gt_cx, gt_cy = gx + gw / 2, gy + gh / 2
        gt_diag = np.sqrt(gw**2 + gh**2)
        dist_thresh = max(gt_diag / 2, dist_floor)
        best_dist = float('inf')
        best_pi = -1
        for pi in range(len(pred_boxes)):
            if pred_matched[pi]:
                continue
            ps = int(pred_slices[pi])
            if abs(ps - gt_s) > slice_tol:
                continue
            x1, y1, x2, y2 = pred_boxes[pi]
            pred_cx = (x1 + x2) / 2
            pred_cy = (y1 + y2) / 2
            dist = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
            if dist < dist_thresh and dist < best_dist:
                best_dist = dist
                best_pi = pi
        if best_pi >= 0:
            gt_matched[gi] = True
            pred_matched[best_pi] = True
    return gt_matched, pred_matched


# =====================================================================
#  Per-case evaluation — same as visualize_case but no plotting
# =====================================================================

def eval_case(model, case, spatial_size, device, score_thresh):
    """Exact same logic as visualize_case minus the figure."""
    volume = np.load(case['npy_path']).astype(np.float32)
    S, H_orig, W_orig = volume.shape

    if H_orig != spatial_size or W_orig != spatial_size:
        resized = np.zeros((S, spatial_size, spatial_size), dtype=np.float32)
        for s in range(S):
            resized[s] = cv2.resize(volume[s], (spatial_size, spatial_size))
        volume = resized

    lo, hi = np.percentile(volume, 1), np.percentile(volume, 99)
    if hi - lo > 1e-6:
        volume = (volume - lo) / (hi - lo)
    volume = np.clip(volume, 0, 1)

    gt_boxes = load_boxes(case['meta_path'], S) if case['meta_path'] else []

    vol_tensor = torch.from_numpy(volume).to(device)
    batch = {'volume': vol_tensor.unsqueeze(0)}

    model.eval()
    # SAME autocast scope as visualize.py — detect + classify INSIDE autocast
    with torch.no_grad(), autocast(device_type='cuda'):
        output = model(batch)
        dets = model.detect(vol_tensor.unsqueeze(0), score_thresh=score_thresh)
        det_probs = model.classify_detections(output['feat_maps'], dets)

    det = dets[0]
    pred_boxes = det['boxes'].cpu().numpy()
    pred_scores = det['scores'].cpu().numpy()
    pred_slices = det['slice_indices'].cpu().numpy()
    cancer_prob = det_probs[0, 1].item()

    gt_matched, pred_matched = match_predictions_to_gt(
        gt_boxes, pred_boxes, pred_slices, spatial_size)

    gt_label = case['class_name']
    pred_label = "Cancer" if cancer_prob > 0.5 else "Benign"
    n_missed = sum(1 for m in gt_matched if not m)
    n_fp = sum(1 for m in pred_matched if not m)

    return {
        'case_id': case['case_id'],
        'gt_label': gt_label,
        'pred_label': pred_label,
        'cancer_prob': cancer_prob,
        'n_gt': len(gt_boxes),
        'n_pred': len(pred_boxes),
        'n_matched': sum(gt_matched),
        'n_missed': n_missed,
        'n_fp': n_fp,
        'correct': pred_label == gt_label,
    }


# =====================================================================
#  Patient aggregation — EXACT copy from visualize.py
# =====================================================================

def aggregate_results(results):
    """Same patient grouping as visualize.py: '_'.join(case_id.split('_')[:2])"""
    from sklearn.metrics import roc_auc_score, roc_curve

    # Patient-level aggregation — SAME as visualize.py
    patient_map = defaultdict(list)
    for r in results:
        pid = '_'.join(r['case_id'].split('_')[:2])
        patient_map[pid].append(r)

    patient_results = {}
    for pid, views in patient_map.items():
        gt = views[0]['gt_label']
        max_prob = max(v['cancer_prob'] for v in views)
        pred = 'Cancer' if max_prob > 0.5 else 'Benign'
        patient_results[pid] = {
            'gt': gt, 'pred': pred, 'max_prob': max_prob,
            'correct': gt == pred,
            'n_views': len(views),
            'total_missed': sum(v['n_missed'] for v in views),
            'total_fp': sum(v['n_fp'] for v in views),
        }

    n_correct = sum(1 for p in patient_results.values() if p['correct'])
    n_total = len(patient_results)
    gts = [1 if p['gt'] == 'Cancer' else 0 for p in patient_results.values()]
    probs = [p['max_prob'] for p in patient_results.values()]

    try:
        auc = roc_auc_score(gts, probs)
    except:
        auc = 0.0

    tp = sum(1 for p in patient_results.values() if p['gt'] == 'Cancer' and p['pred'] == 'Cancer')
    fn = sum(1 for p in patient_results.values() if p['gt'] == 'Cancer' and p['pred'] == 'Benign')
    tn = sum(1 for p in patient_results.values() if p['gt'] == 'Benign' and p['pred'] == 'Benign')
    fp = sum(1 for p in patient_results.values() if p['gt'] == 'Benign' and p['pred'] == 'Cancer')
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)

    total_missed = sum(r['n_missed'] for r in results)
    total_gt = sum(r['n_gt'] for r in results)

    # Youden's J at the fixed classification threshold tau_cls=0.5.
    youden_j_fixed = sens + spec - 1.0
    # Also compute max-J over the ROC for internal comparison.
    try:
        fpr, tpr, thresholds = roc_curve(gts, probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        youden_j_max = float(j_scores[best_idx])
        opt_thresh = float(thresholds[best_idx])
    except:
        youden_j_max = 0.0
        opt_thresh = 0.5
    # Headline J = fixed-threshold J
    youden_j = youden_j_fixed

    # Prob spread
    probs_arr = np.array(probs)

    return {
        'PatAUC':     round(auc, 4),
        'Sens':       round(sens, 4),
        'Spec':       round(spec, 4),
        'Youden_J':     round(float(youden_j), 4),
        'Youden_J_max': round(float(youden_j_max), 4),
        'Opt_Thresh':   round(float(opt_thresh), 4),
        'Prob_min':   round(float(probs_arr.min()), 4),
        'Prob_max':   round(float(probs_arr.max()), 4),
        'Accuracy':   round(n_correct / max(n_total, 1), 4),
        'Det_Recall': round(1 - total_missed / max(total_gt, 1), 4),
        'N_Patients': n_total,
        'N_GT':       total_gt,
        'N_Missed':   total_missed,
        'View_Acc':   round(sum(1 for r in results if r['correct']) / max(len(results), 1), 4),
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
    }


# =====================================================================
#  Model loading — EXACT same as visualize.py
# =====================================================================

def load_model(checkpoint_path):
    model = MambaCenterNet(num_classes=2, use_mamba=True).to(DEVICE)
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    # SAME as visualize.py: ema_state_dict first, then model_state_dict
    state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    print(f"  ✓ Loaded checkpoint (epoch {epoch})")
    model.eval()
    return model


# =====================================================================
#  Run one ablation
# =====================================================================

def validate_one(name, model_dir, cases, score_thresh, spatial_size):
    ckpt = model_dir / "best_patient_auc.pt"
    if not ckpt.exists():
        ckpt = model_dir / "best_model.pt"
    if not ckpt.exists():
        ckpt = model_dir / "best_model.pth"
    if not ckpt.exists():
        print(f"  ✗ No checkpoint in {model_dir}")
        return None

    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  score_thresh = {score_thresh}")

    model = load_model(ckpt)

    results = []
    for i, case in enumerate(cases):
        r = eval_case(model, case, spatial_size, DEVICE, score_thresh)
        results.append(r)
        # -- per-case dump (used later for CI / DeLong / FROC) --
        if _DUMP_DIR is not None:
            _dump_case(_DUMP_DIR, name, r)
        status = "✓" if r['correct'] else "✗"
        missed_str = f" [MISSED {r['n_missed']}]" if r['n_missed'] > 0 else ""
        print(f"  [{i+1}/{len(cases)}] {status} {r['case_id']} | "
              f"GT:{r['gt_label']} Pred:{r['pred_label']} (p={r['cancer_prob']:.3f})"
              f" | Det:{r['n_matched']}/{r['n_gt']} FP:{r['n_fp']}{missed_str}")

    metrics = aggregate_results(results)

    print(f"\n  PatAUC={metrics['PatAUC']}  Sens={metrics['Sens']}  Spec={metrics['Spec']}  "
          f"J={metrics['Youden_J']}  DetRecall={metrics['Det_Recall']}")
    print(f"  Probs: [{metrics['Prob_min']}, {metrics['Prob_max']}]  "
          f"Acc={metrics['Accuracy']}  #Pat={metrics['N_Patients']}  "
          f"TP={metrics['TP']} FP={metrics['FP']} TN={metrics['TN']} FN={metrics['FN']}")

    del model
    torch.cuda.empty_cache()
    return metrics


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="all",
                        choices=["all", "baseline"] + ALL_CONFIGS)
    parser.add_argument("--score_thresh", type=float, default=0.16)
    parser.add_argument("--spatial_size", type=int, default=384)
    parser.add_argument("--split", default="validation",
                        choices=["validation", "test"],
                        help="which official partition to evaluate on")
    parser.add_argument("--dump_dir", default=None,
                        help="if set, write per-case JSONL for each config here")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  MambaCenterNet v5.2 — Ablation Eval (visualize.py pipeline)")
    print(f"  score_thresh = {args.score_thresh}")
    print(f"{'='*70}")

    global _DUMP_DIR
    if args.dump_dir:
        _DUMP_DIR = args.dump_dir
        print(f"  per-case dump -> {args.dump_dir}")

    cases = load_dataset_cases(DATA_ROOT, split=args.split, spatial_size=args.spatial_size)
    print(f"  split = {args.split}")
    print(f"  {len(cases)} {args.split} cases loaded")

    all_results = {}

    # Baseline
    if args.config in ("all", "baseline"):
        r = validate_one("BASELINE (v5.2)", BASELINE_DIR, cases,
                         args.score_thresh, args.spatial_size)
        if r:
            all_results["BASELINE (v5.2)"] = r

    # Ablations
    if args.config == "all":
        configs_to_run = ALL_CONFIGS
    elif args.config in ALL_CONFIGS:
        configs_to_run = [args.config]
    else:
        configs_to_run = []

    for name in configs_to_run:
        model_dir = ABLATION_ROOT / name
        if not model_dir.exists():
            print(f"\n  SKIP {name}: not found")
            all_results[name] = {"PatAUC": "NOT FOUND"}
            continue
        r = validate_one(name, model_dir, cases, args.score_thresh, args.spatial_size)
        if r:
            all_results[name] = r
        else:
            all_results[name] = {"PatAUC": "NO CKPT"}

    # Summary
    print(f"\n\n{'='*95}")
    print(f"  SUMMARY  (score_thresh={args.score_thresh})")
    print(f"{'='*95}")
    header = (f"{'Config':<25} {'PatAUC':<8} {'Sens':<7} {'Spec':<7} {'J':<7} "
              f"{'DetRec':<8} {'Acc':<7} {'Probs':<22} {'CM'}")
    print(header)
    print("─" * 95)

    prev_cat = None
    for name, r in all_results.items():
        for cat, members in CATEGORIES.items():
            if name in members and cat != prev_cat:
                print(f"\n  {cat}")
                prev_cat = cat

        if isinstance(r.get("PatAUC"), str):
            print(f"  {name:<23} {r['PatAUC']}")
            continue

        prob_str = f"[{r['Prob_min']}, {r['Prob_max']}]"
        cm_str = f"TP={r['TP']} FP={r['FP']} TN={r['TN']} FN={r['FN']}"
        print(f"  {name:<23} {r['PatAUC']:<8} {r['Sens']:<7} {r['Spec']:<7} "
              f"{r['Youden_J']:<7} {r['Det_Recall']:<8} {r['Accuracy']:<7} "
              f"{prob_str:<22} {cm_str}")

    # Save
    out_json = ABLATION_ROOT / f"ablation_eval_t{args.score_thresh}.json"
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()