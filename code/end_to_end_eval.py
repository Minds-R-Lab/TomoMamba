#!/usr/bin/env python3
"""
end_to_end_eval.py  ->  end-to-end evaluation of the complete two-stage cascade.

Evaluates the COMPLETE two-stage pipeline on the screening population, which
is the deployment-relevant task and has never been measured. Until now Stage 1
and Stage 2 were only evaluated independently, with Stage 2 receiving
ground-truth suspicious cases rather than cases Stage 1 actually forwarded.

WHAT THIS MEASURES
------------------
Population: every breast in the split, Normal + Benign + Cancer, paired
CC/MLO, exactly as Stage 1 was trained to see it.

    Stage 1  ->  P(abnormal) per breast
    filter   ->  breasts below the screening threshold are triaged out.
                 Any cancer among them is MISSED, permanently.
    Stage 2  ->  survivors get a cancer probability
    final    ->  scored as Cancer vs not-Cancer (Normal + Benign)

Note the task change. Stage 2's ~0.74 patient AUC is benign-vs-cancer among
patients who were ALL biopsied: the hardest slice of the population, where
imaging alone was already ambiguous enough to warrant a needle. The pipeline
task is finding cancer in a screening population, which is a different and
more clinically meaningful question.

TWO CASCADE VARIANTS, BOTH REPORTED
-----------------------------------
  hard  screened-out breasts score 0. This is literally what the deployed
        system does, but it creates a large tie group at 0 which depresses AUC.
  soft  every breast scores P(abnormal) x P(cancer | abnormal), the proper
        probability chain. Better behaved as a ranking metric.
Neither is "the" answer; report both and say which is which.

PREPROCESSING (they differ, and getting this wrong fails silently)
------------------------------------------------------------------
  Stage 1: volumes used AS SAVED (already z-scored by preprocessing),
           bilinear-resized to 512x512.        [datasets.py::_load_volume]
  Stage 2: 1st/99th percentile normalisation to [0,1], resized to 384x384.
                                               [eval_official.py::evaluate]

USAGE
    python end_to_end_eval.py \
        --stage1_ckpt ~/DBT/checkpoints/stage1/best_model.pt \
        --stage2_ckpt /mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_model.pt \
        --split validation
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
import cv2
from torch.amp import autocast
from sklearn.metrics import roc_auc_score

HOME = Path(os.path.expanduser("~"))
# v5.2 FIRST so its centernet_models wins over the older copy in ~/DBT
sys.path.insert(0, str(HOME / "DBT" / "MambaCenterNet_v5.2"))
sys.path.insert(1, str(HOME / "DBT"))

DEFAULT_ROOT = "/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15"


# =====================================================================
# Breast index  (mirrors DBTBreastDatasetStage1._build_breast_index)
# =====================================================================

def parse_filename(p: Path):
    """{PatientID}_{StudyUID}_{View}.npy  where View is e.g. LCC, RMLO."""
    stem = p.stem
    parts = stem.rsplit('_', 1)
    if len(parts) != 2:
        return None
    patient_study, view = parts
    view = view.upper().replace(' ', '')
    if view.startswith('L'):
        lat, vt = 'L', view[1:]
    elif view.startswith('R'):
        lat, vt = 'R', view[1:]
    else:
        return None
    vt = vt.rstrip('0123456789')
    if vt not in ('CC', 'MLO'):
        return None
    ps = patient_study.rsplit('_', 1)
    pid = ps[0] if len(ps) == 2 else patient_study
    study = ps[1] if len(ps) == 2 else ""
    return {'filepath': p, 'patient_id': pid, 'study_uid': study,
            'laterality': lat, 'view_type': vt, 'case_id': stem}


def build_breasts(data_root, split):
    """One entry per (patient, study, laterality), with CC and/or MLO."""
    root = Path(data_root) / split
    d = defaultdict(lambda: {'CC': None, 'MLO': None, 'classes': set()})
    for cls in ['Normal', 'Benign', 'Cancer']:
        cdir = root / cls
        if not cdir.exists():
            continue
        for f in sorted(cdir.glob("*.npy")):
            info = parse_filename(f)
            if info is None:
                continue
            key = (info['patient_id'], info['study_uid'], info['laterality'])
            d[key][info['view_type']] = info
            d[key]['classes'].add(cls)

    breasts = []
    for key, v in d.items():
        classes = v['classes']
        detailed = 'Cancer' if 'Cancer' in classes else (
                   'Benign' if 'Benign' in classes else 'Normal')
        breasts.append({
            'key': key, 'patient_id': key[0], 'laterality': key[2],
            'cc': v['CC'], 'mlo': v['MLO'],
            'detailed': detailed,
            'abnormal': int(detailed in ('Benign', 'Cancer')),  # Stage 1 target
            'cancer': int(detailed == 'Cancer'),                # final target
        })
    return sorted(breasts, key=lambda b: (b['patient_id'], b['laterality']))


# =====================================================================
# Preprocessing, one per stage
# =====================================================================

def load_stage1(info, size):
    """AS SAVED (already z-scored), bilinear resize. No normalisation."""
    if info is None:
        return np.zeros((15, size, size), dtype=np.float32), 0.0
    vol = np.load(info['filepath']).astype(np.float32)
    if vol.shape[1] != size or vol.shape[2] != size:
        t = torch.from_numpy(vol).unsqueeze(1)
        t = F.interpolate(t, size=(size, size), mode='bilinear', align_corners=False)
        vol = t.squeeze(1).numpy()
    return vol, 1.0


def load_stage2(info, size):
    """Percentile normalisation to [0,1] then resize. Matches eval_official."""
    vol = np.load(info['filepath']).astype(np.float32)
    S = vol.shape[0]
    if vol.shape[1] != size:
        r = np.zeros((S, size, size), dtype=np.float32)
        for s in range(S):
            r[s] = cv2.resize(vol[s], (size, size))
        vol = r
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi - lo > 1e-6:
        vol = (vol - lo) / (hi - lo)
    return np.clip(vol, 0, 1)


# =====================================================================
# Metrics
# =====================================================================

def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return float('nan')
    return roc_auc_score(y, p)


def sens_spec(y, p, thr):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return (tp / max(tp + fn, 1), tn / max(tn + fp, 1), tp, fn, tn, fp)


def boot_ci(y, p, n=5000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        try:
            vals.append(roc_auc_score(y[i], p[i]))
        except Exception:
            pass
    if not vals:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =====================================================================
# Main
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1_ckpt', default=str(HOME / 'DBT/checkpoints/stage1/best_model.pt'))
    ap.add_argument('--stage2_ckpt', default='/mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_model.pt')
    ap.add_argument('--data_root', default=DEFAULT_ROOT)
    ap.add_argument('--split', default='validation')
    ap.add_argument('--stage1_size', type=int, default=512)
    ap.add_argument('--stage2_size', type=int, default=384)
    ap.add_argument('--stage1_thresh', type=float, default=0.139,
                    help='Default tau_scr = 0.139 (>=90%% sensitivity).')
    ap.add_argument('--det_thresh', type=float, default=0.16)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = torch.cuda.is_available()

    print("=" * 76)
    print("END-TO-END TWO-STAGE EVALUATION")
    print("=" * 76)
    print(f"  data   : {args.data_root}  split={args.split}")
    print(f"  stage1 : {args.stage1_ckpt}  @ {args.stage1_size}px")
    print(f"  stage2 : {args.stage2_ckpt}  @ {args.stage2_size}px")
    print(f"  device : {dev}")

    # ---- index ----
    breasts = build_breasts(args.data_root, args.split)
    if not breasts:
        print("\n[fatal] No breasts indexed. Check --data_root / --split.")
        sys.exit(1)
    n_can = sum(b['cancer'] for b in breasts)
    n_abn = sum(b['abnormal'] for b in breasts)
    both = sum(1 for b in breasts if b['cc'] and b['mlo'])
    print(f"\n  breasts: {len(breasts)}  (cancer {n_can}, abnormal {n_abn}, "
          f"normal {len(breasts)-n_abn})")
    print(f"  with both views: {both}   single-view: {len(breasts)-both}")

    # ---- Stage 1 ----
    print("\n[1/3] Stage 1 screening...")
    from models.stage1_screening import DBTScreeningModel
    s1 = DBTScreeningModel(backbone='resnet18', pretrained=False,
                           feature_dim=512, fusion_dim=256,
                           pool_method='attention', num_classes=2).to(dev)
    ck1 = torch.load(args.stage1_ckpt, map_location=dev, weights_only=False)
    st1 = ck1.get('model_state_dict', ck1.get('state_dict', ck1))
    miss, unexp = s1.load_state_dict(st1, strict=False)
    if len(miss) > 20:
        print(f"      [warn] {len(miss)} missing keys. Architecture may not match "
              f"the checkpoint; results would be meaningless. Check config.stage1.")
    s1.eval()
    print(f"      loaded (epoch {ck1.get('epoch','?')}); "
          f"{len(miss)} missing / {len(unexp)} unexpected keys")

    p_abn = np.zeros(len(breasts))
    with torch.no_grad():
        for i, b in enumerate(breasts):
            cc, ccv = load_stage1(b['cc'], args.stage1_size)
            ml, mlv = load_stage1(b['mlo'], args.stage1_size)
            batch = {
                'cc_volume': torch.from_numpy(cc).unsqueeze(0).to(dev),
                'mlo_volume': torch.from_numpy(ml).unsqueeze(0).to(dev),
                'cc_valid': torch.tensor([ccv]).to(dev),
                'mlo_valid': torch.tensor([mlv]).to(dev),
            }
            with autocast(device_type='cuda', enabled=amp):
                p_abn[i] = float(s1(batch)['probs'][0, 1])
            if (i + 1) % 100 == 0:
                print(f"      {i+1}/{len(breasts)}")

    y_abn = np.array([b['abnormal'] for b in breasts])
    y_can = np.array([b['cancer'] for b in breasts])
    s1_auc = auc(y_abn, p_abn)
    se, sp, tp, fn, tn, fp = sens_spec(y_abn, p_abn, args.stage1_thresh)
    print(f"\n      Stage 1 alone (breast-level, Normal vs Abnormal)")
    print(f"        AUC {s1_auc:.4f}   sens {se:.3f}  spec {sp:.3f}  "
          f"@tau={args.stage1_thresh}")
    print(f"        standalone Stage 1 eval: AUC 0.9696, sens 0.90, spec 0.945 "
          f"at tau_scr = 0.139")

    passed = p_abn >= args.stage1_thresh
    cancers_lost = int(((~passed) & (y_can == 1)).sum())
    benign_lost = int(((~passed) & (y_abn == 1) & (y_can == 0)).sum())
    n_ben = int(((y_abn == 1) & (y_can == 0)).sum())
    print(f'        BENIGN LOST HERE: {benign_lost}/{n_ben}')
    print(f"        forwarded {int(passed.sum())}/{len(breasts)} breasts; "
          f"CANCERS LOST HERE: {cancers_lost}/{n_can}")

    del s1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Stage 2 on survivors ----
    print("\n[2/3] Stage 2 on forwarded breasts...")
    from centernet_models import MambaCenterNet
    s2 = MambaCenterNet(num_classes=2, use_mamba=True,
                        spatial_size=args.stage2_size).to(dev)
    ck2 = torch.load(args.stage2_ckpt, map_location=dev, weights_only=False)
    st2 = ck2.get('ema_state_dict', ck2.get('model_state_dict', ck2))
    m2, u2 = s2.load_state_dict(st2, strict=False)
    s2.eval()
    print(f"      loaded (epoch {ck2.get('epoch','?')}); "
          f"{len(m2)} missing / {len(u2)} unexpected keys")

    # Score every breast that survives the LOWEST threshold we will sweep, not
    # just the operating threshold. Otherwise breasts that pass at a lower tau
    # would score 0 simply because Stage 2 never saw them, which silently
    # flattens the sweep and makes the trade-off curve meaningless.
    SWEEP_TAUS = [0.05, 0.10, 0.139, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    scored = p_abn >= min(min(SWEEP_TAUS), args.stage1_thresh)
    p_can_given = np.zeros(len(breasts))
    idx = np.where(scored)[0]
    print(f"      scoring {len(idx)} breasts (all above tau={min(min(SWEEP_TAUS), args.stage1_thresh):.3f}) "
          f"so the threshold sweep is valid")
    with torch.no_grad():
        for n, i in enumerate(idx):
            b = breasts[i]
            probs = []
            for info in (b['cc'], b['mlo']):
                if info is None:
                    continue
                vol = load_stage2(info, args.stage2_size)
                vt = torch.from_numpy(vol).unsqueeze(0).to(dev)
                with autocast(device_type='cuda', enabled=amp):
                    out = s2({'volume': vt})
                    dets = s2.detect(vt, score_thresh=args.det_thresh)
                    dp = s2.classify_detections(out['feat_maps'], dets)
                probs.append(float(dp[0, 1]))
            p_can_given[i] = max(probs) if probs else 0.0   # max over views
            if (n + 1) % 50 == 0:
                print(f"      {n+1}/{len(idx)}")

    # ---- Cascade scores ----
    print("\n[3/3] Combining...")
    hard = np.where(passed, p_can_given, 0.0)
    soft = p_abn * p_can_given          # proper probability chain

    results = {'split': args.split, 'n_breasts': len(breasts),
               'n_cancer': int(n_can), 'n_abnormal': int(n_abn),
               'stage1': {'auc': s1_auc, 'sens': se, 'spec': sp,
                          'threshold': args.stage1_thresh,
                          'forwarded': int(passed.sum()),
                          'cancers_lost': cancers_lost, 'benign_lost': benign_lost, 'n_benign': n_ben},
               'end_to_end': {}}

    print("\n" + "=" * 76)
    print("RESULTS")
    print("=" * 76)
    print(f"\n  Stage 1 alone   : breast-level AUC {s1_auc:.4f}  "
          f"(Normal vs Abnormal)")
    print(f"  Cancers lost    : {cancers_lost} of {n_can} at screening "
          f"({100*cancers_lost/max(n_can,1):.1f}%)")

    for name, sc in (('hard cascade (screened-out = 0)', hard),
                     ('soft cascade (P_abn x P_cancer)', soft)):
        a = auc(y_can, sc)
        ci = boot_ci(y_can, sc)
        se2, sp2, *_ = sens_spec(y_can, sc, 0.5)
        key = 'hard' if name.startswith('hard') else 'soft'
        results['end_to_end'][key] = {'auc': a, 'sens@0.5': se2, 'spec@0.5': sp2,
                                      'ci95': list(ci) if ci else None}
        print(f"\n  {name}")
        print(f"    Cancer vs rest AUC : {a:.4f}" +
              (f"   95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
        print(f"    sens {se2:.3f}  spec {sp2:.3f}  @0.5")

    # ---- screening-threshold sweep ----
    print("\n  Stage 1 threshold sweep (the recall/workload trade-off):")
    print(f"    {'tau':>5} {'fwd':>6} {'lost':>5} {'e2e AUC':>9}")
    sweep = []
    for t in SWEEP_TAUS:
        ps = p_abn >= t
        lost = int(((~ps) & (y_can == 1)).sum())
        sc = np.where(ps, p_can_given, 0.0)
        a = auc(y_can, sc)
        sweep.append({'tau': t, 'forwarded': int(ps.sum()),
                      'cancers_lost': lost, 'auc_hard': a})
        print(f"    {t:5.2f} {int(ps.sum()):6d} {lost:5d} {a:9.4f}")
    results['stage1_sweep'] = sweep
    print("\n    Lowering the screening threshold loses fewer cancers but")
    print("    forwards more breasts to Stage 2. That trade-off, not a single")
    print("    number, is the relevant trade-off.")

    # Default name includes the split so consecutive runs cannot overwrite
    # each other; override with --out for a specific path.
    out = args.out or str(Path(args.stage2_ckpt).parent /
                          f'end_to_end_{args.split}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  saved: {out}")
    print("=" * 76)


if __name__ == '__main__':
    main()
