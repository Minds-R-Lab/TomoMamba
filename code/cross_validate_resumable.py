"""
Cross-Validation Harness for MambaCenterNet v5.2 - CRASH-SAFE / RESUMABLE
=========================================================================
Same protocol as cross_validate.py (patient-grouped, stratified k-fold,
faithful to train_stage2.py), with the durability that version lacked.

WHAT CHANGED, AND WHY IT MATTERS AFTER A CRASH
----------------------------------------------
1. RESUME AT FOLD LEVEL. A completed fold writes fold_N/fold_result.json.
   On --resume those folds are loaded from disk and skipped entirely.

2. RESUME MID-FOLD. Every epoch writes fold_N/last.pt containing the FULL
   training state: model, optimizer, scheduler, AMP scaler, epoch, best
   score, patience counter, history, and RNG states. A killed fold picks
   up at the next epoch instead of restarting from zero. This is the bit
   that turns "lost 14 hours" into "lost 20 minutes".

3. OOF PREDICTIONS PERSISTED. The previous version kept each fold's
   out-of-fold patient predictions in memory until the very end, so a
   crash lost them even for folds that had finished. They now go into
   fold_N/fold_result.json as each fold completes.

4. ATOMIC WRITES. Saves go to a .tmp file then os.replace(), which is
   atomic on the same filesystem. A restart mid-save can no longer leave
   a truncated, unloadable checkpoint - the exact failure mode that
   produces a [CORRUPT] line in crash_triage.py.

5. LEGACY RECOVERY. Folds finished by the OLD cross_validate.py wrote
   fold_N/training_history.json but no fold_result.json. --resume rebuilds
   their per-fold metrics from that history so they do NOT need re-running.
   Their OOF predictions are unrecoverable (they were never written to
   disk), so those folds are excluded from the pooled-OOF section and the
   report says so explicitly. Per-fold mean +/- std - the number you
   actually report in a paper - is fully intact for them.

REPRODUCIBILITY OF FOLD ASSIGNMENT
----------------------------------
Fold membership is deterministic given (data_root, cv_splits, folds, seed):
cases are indexed with sorted() and split by StratifiedGroupKFold with a
fixed random_state. Resuming with the same args reproduces the same folds,
so mixing resumed and fresh folds is sound. A fingerprint of those args is
stored in cv_state.json and checked on resume; a mismatch aborts rather
than silently blending incompatible runs.

Note: with num_workers > 0, resume is statistically equivalent, not
bit-identical (worker RNG streams restart). This does not affect validity.

USAGE
-----
    conda activate tomomamba
    cd ~/DBT/MambaCenterNet_v5.2

    # start (or restart after a crash - same command, add --resume)
    python cross_validate_resumable.py --folds 5 --resume

    # belt and braces against another surprise reboot:
    nohup python -u cross_validate_resumable.py --folds 5 --resume \
        > /mnt/e/cv_run.log 2>&1 &
    tail -f /mnt/e/cv_run.log

The test set is NOT touched: --cv_splits defaults to train + validation.
"""

import os
import sys
import gc
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_stage2 import (DBTStage2Dataset, collate_fn, MetricTracker,
                          train_epoch, eval_epoch)
from centernet_models import (MambaCenterNet, CenterNetLoss,
                              aggregate_per_patient, compute_patient_metrics)

CLASS_MAP = {'Benign': 0, 'Cancer': 1}

# Copied verbatim from train_stage2.py so a fold trains identically.
LOSS_KWARGS = dict(cls_weight=5.0, size_weight=0.3, offset_weight=1.0,
                   peak_reg_weight=0.3, min_radius=3,
                   cls_focal_gamma=2.0, label_smoothing=0.1)

AGG_KEYS = ['patient_auc', 'patient_acc', 'patient_sens', 'patient_spec',
            'gt_auc', 'gt_acc', 'det_auc', 'det_acc',
            'recall@0.1', 'recall@0.25', 'recall@0.5', 'mean_best_iou',
            'val_loss', 'score']


# =============================================================================
# Atomic persistence - the anti-corruption layer
# =============================================================================

def atomic_torch_save(obj, path):
    """Write then rename. os.replace is atomic on the same filesystem, so a
    crash leaves either the old file or the new one, never a half-written
    one. Plain torch.save straight to the target can be truncated."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(to_jsonable(obj), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# =============================================================================
# Helpers (identical semantics to cross_validate.py)
# =============================================================================

def patient_id_of(case_id: str) -> str:
    """Patient id used for grouping. Identical to aggregate_per_patient()."""
    parts = case_id.rsplit('_', 1)
    return parts[0] if len(parts) >= 2 else case_id


def combined_score(patient_auc, gt_auc, det_auc, det_metrics) -> float:
    """The v5.2 combined score best_model.pt is selected on."""
    return (patient_auc * 0.40 +
            gt_auc * 0.10 +
            det_auc * 0.15 +
            det_metrics.get('recall@0.25', 0) * 0.15 +
            det_metrics.get('recall@0.1', 0) * 0.10 +
            det_metrics.get('mean_best_iou', 0) * 0.10)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def capture_rng():
    return {
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def restore_rng(st):
    try:
        torch.set_rng_state(st['torch'])
        if st.get('cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st['cuda'])
        np.random.set_state(st['numpy'])
        random.setstate(st['python'])
    except Exception as e:
        print(f"    [warn] could not restore RNG state ({e}); continuing.")


def index_all_cases(data_root, splits):
    """Mirrors DBTStage2Dataset._index_cases exactly (same class map, same
    metadata resolution incl. lowercase fallback, sorted for determinism)."""
    root = Path(data_root)
    cases, per, seen, dup = [], {}, set(), 0
    for split in splits:
        split_dir, meta_dir = root / split, root / 'metadata' / split
        if not split_dir.exists():
            print(f"  [skip] split directory not found: {split_dir}")
            continue
        for class_name in ['Benign', 'Cancer']:
            class_dir = split_dir / class_name
            meta_class_dir = meta_dir / class_name
            if not class_dir.exists():
                continue
            count = 0
            for npy_path in sorted(class_dir.glob("*.npy")):
                cid = npy_path.stem
                if cid in seen:
                    dup += 1
                    continue
                seen.add(cid)
                meta_path = meta_class_dir / f"{cid}.json"
                if not meta_path.exists():
                    meta_path = meta_class_dir / f"{cid.lower()}.json"
                cases.append({'npy_path': str(npy_path),
                              'meta_path': str(meta_path) if meta_path.exists() else None,
                              'case_id': cid, 'class_name': class_name,
                              'label': CLASS_MAP[class_name], 'split': split})
                count += 1
            per[(split, class_name)] = count
    if dup:
        print(f"  [warn] skipped {dup} duplicate case_id(s) seen in >1 split")
    return cases, per


def make_folds(cases, n_splits, seed):
    labels = np.array([c['label'] for c in cases])
    groups = np.array([patient_id_of(c['case_id']) for c in cases])
    pid_labels = defaultdict(set)
    for c in cases:
        pid_labels[patient_id_of(c['case_id'])].add(int(c['label']))
    mixed = sorted(p for p, s in pid_labels.items() if len(s) > 1)
    if mixed:
        print(f"  [warn] {len(mixed)} patient(s) have mixed Benign/Cancer views; "
              f"stratification uses the per-case label. Example: {mixed[0]}")
    X = np.zeros(len(cases))
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                          random_state=seed).split(X, labels, groups))
        method = "StratifiedGroupKFold"
    except Exception as e:
        from sklearn.model_selection import GroupKFold
        print(f"  [warn] StratifiedGroupKFold unavailable ({e}); using GroupKFold.")
        folds = list(GroupKFold(n_splits=n_splits).split(X, labels, groups))
        method = "GroupKFold"
    return folds, groups, labels, method, mixed


class FoldDataset(DBTStage2Dataset):
    """DBTStage2Dataset built from an explicit case list. All data handling
    is inherited unchanged, so a fold's pipeline matches production."""

    def __init__(self, cases, split, spatial_size=384, max_boxes=20, augment=True):
        self.data_root = None
        self.split = split
        self.spatial_size = spatial_size
        self.max_boxes = max_boxes
        self.augment = augment and (split == 'train')
        self.cases = list(cases)
        self.box_crops = []
        if self.augment:
            self._build_box_crop_pool()
        self._print_stats()


@torch.no_grad()
def collect_oof_patient_predictions(model, loader, device, score_thresh=0.20,
                                    use_amp=True):
    """Same realistic detection -> ROI-classification path eval_epoch uses."""
    model.eval()
    tracker = MetricTracker()
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with autocast(device_type='cuda', enabled=use_amp):
            output = model(batch)
            dets = model.detect(batch['volume'], score_thresh=score_thresh)
            det_probs = model.classify_detections(output['feat_maps'], dets)
        tracker.update(det_probs.argmax(dim=-1), batch['labels'], det_probs,
                       case_ids=batch['case_id'])
    return aggregate_per_patient(tracker.case_ids,
                                 np.array(tracker.probs),
                                 np.array(tracker.labels))


def bootstrap_auc_ci(labels, probs, n_boot=10000, seed=0):
    labels, probs = np.asarray(labels), np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return None
    rng = np.random.default_rng(seed)
    aucs, n = [], len(labels)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            aucs.append(roc_auc_score(labels[idx], probs[idx]))
        except Exception:
            pass
    if not aucs:
        return None
    return (float(np.mean(aucs)), float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def snapshots_from_history(hist):
    """Rebuild best-combined / best-patient-AUC snapshots from a history
    dict. Used to recover folds finished by the OLD script."""
    n = len(hist.get('epoch', []))
    if n == 0:
        return None, None
    def row(i):
        return {k: (v[i] if isinstance(v, list) and i < len(v) else None)
                for k, v in hist.items()}
    scores = [s if isinstance(s, (int, float)) else -1 for s in hist.get('score', [])]
    paucs = [s if isinstance(s, (int, float)) else -1 for s in hist.get('patient_auc', [])]
    best_c = row(int(np.argmax(scores))) if scores else None
    best_p = row(int(np.argmax(paucs))) if paucs else None
    return best_c, best_p


# =============================================================================
# One fold (resumable)
# =============================================================================

def train_one_fold(fold_idx, train_cases, val_cases, args, device):
    fold_num = fold_idx + 1
    fold_dir = Path(args.save_dir) / f"fold_{fold_num}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    result_path = fold_dir / 'fold_result.json'
    last_path = fold_dir / 'last.pt'
    best_path = fold_dir / 'best_model.pt'

    # ---- Already finished by THIS script? Load and skip. ----
    if args.resume and result_path.exists():
        try:
            with open(result_path) as f:
                r = json.load(f)
            print(f"\n### FOLD {fold_num}/{args.folds}: COMPLETE (loaded from disk, "
                  f"{len(r.get('oof_patients', {}))} OOF patients) - skipping")
            return r
        except Exception as e:
            print(f"  [warn] fold_result.json unreadable ({e}); re-running fold.")

    # ---- Finished by the OLD script? Recover metrics, no re-run. ----
    legacy_hist = fold_dir / 'training_history.json'
    if args.resume and legacy_hist.exists() and not result_path.exists():
        try:
            with open(legacy_hist) as f:
                hist = json.load(f)
            bc, bp = snapshots_from_history(hist)
            if bc:
                print(f"\n### FOLD {fold_num}/{args.folds}: recovered from legacy run "
                      f"({len(hist.get('epoch', []))} epochs, "
                      f"PatAUC {bc.get('patient_auc', float('nan')):.4f}). "
                      f"Metrics kept; OOF preds unavailable (never written to disk).")
                return {'best_combined': bc, 'best_patient_auc': bp,
                        'oof_patients': {}, 'oof_recovered': False}
        except Exception as e:
            print(f"  [warn] legacy history unreadable ({e}); re-running fold.")

    print(f"\n{'#'*70}")
    print(f"# FOLD {fold_num}/{args.folds}")
    print(f"{'#'*70}")

    set_seed(args.seed + fold_idx)

    train_ds = FoldDataset(train_cases, 'train', args.spatial_size,
                           max_boxes=args.max_boxes, augment=True)
    val_ds = FoldDataset(val_cases, 'validation', args.spatial_size,
                         max_boxes=args.max_boxes, augment=False)
    if len(set(c['label'] for c in val_cases)) < 2:
        print("  [warn] validation fold has a single class; its AUC will be 0.5.")

    sampler = WeightedRandomSampler(train_ds.get_class_weights(),
                                    len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True)

    model = MambaCenterNet(num_classes=2, dropout=0.7,
                           use_mamba=not args.no_mamba,
                           spatial_size=args.spatial_size,
                           backbone=args.backbone).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = CenterNetLoss(**LOSS_KWARGS)
    backbone_params = list(model.backbone.parameters())
    detect_params = (list(model.cross_slice.parameters()) if model.cross_slice else []) \
        + list(model.detect_head.parameters())
    cls_params = list(model.roi_classifier.parameters()) \
        + list(model.global_classifier.parameters())
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': detect_params, 'lr': args.lr},
        {'params': cls_params, 'lr': args.lr * 5.0},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)
    scaler = GradScaler('cuda')

    start_epoch = 1
    best_score = -1.0
    best_snapshot = None
    best_patient_auc = -1.0
    best_patient_snapshot = None
    patience_ctr = 0
    history = defaultdict(list)

    # ---- Mid-fold resume ----
    if args.resume and last_path.exists():
        try:
            ck = torch.load(last_path, map_location=device, weights_only=False)
            model.load_state_dict(ck['model_state_dict'])
            optimizer.load_state_dict(ck['optimizer_state_dict'])
            scheduler.load_state_dict(ck['scheduler_state_dict'])
            scaler.load_state_dict(ck['scaler_state_dict'])
            start_epoch = ck['epoch'] + 1
            best_score = ck['best_score']
            best_snapshot = ck['best_snapshot']
            best_patient_auc = ck['best_patient_auc']
            best_patient_snapshot = ck['best_patient_snapshot']
            patience_ctr = ck['patience_ctr']
            history = defaultdict(list, ck['history'])
            if ck.get('rng') is not None:
                restore_rng(ck['rng'])
            print(f"  RESUMED mid-fold from epoch {ck['epoch']} "
                  f"(best score so far {best_score:.4f}) -> continuing at "
                  f"epoch {start_epoch}")
        except Exception as e:
            print(f"  [warn] last.pt unusable ({str(e)[:100]}); restarting fold "
                  f"from epoch 1.")

    for epoch in range(start_epoch, args.epochs + 1):
        t_loss, t_comp, t_cls = train_epoch(model, train_loader, loss_fn,
                                            optimizer, scaler, device)
        v_loss, v_comp, v_gt_cls, v_det_cls, v_det, v_patient = eval_epoch(
            model, val_loader, loss_fn, device, spatial_size=args.spatial_size)

        patient_auc = v_patient['auc'] if v_patient else 0.5
        gt_auc, det_auc = v_gt_cls['auc'], v_det_cls['auc']
        scheduler.step(patient_auc)
        score = combined_score(patient_auc, gt_auc, det_auc, v_det)

        snapshot = {
            'fold': fold_num, 'epoch': epoch, 'score': float(score),
            'val_loss': float(v_loss), 'patient_auc': float(patient_auc),
            'patient_acc': float(v_patient['accuracy']) if v_patient else float('nan'),
            'patient_sens': float(v_patient['sensitivity']) if v_patient else float('nan'),
            'patient_spec': float(v_patient['specificity']) if v_patient else float('nan'),
            'patient_n': int(v_patient['n_patients']) if v_patient else 0,
            'gt_auc': float(gt_auc), 'gt_acc': float(v_gt_cls['accuracy']),
            'det_auc': float(det_auc), 'det_acc': float(v_det_cls['accuracy']),
            'recall@0.1': float(v_det.get('recall@0.1', 0)),
            'recall@0.25': float(v_det.get('recall@0.25', 0)),
            'recall@0.5': float(v_det.get('recall@0.5', 0)),
            'mean_best_iou': float(v_det.get('mean_best_iou', 0)),
        }
        for k, v in snapshot.items():
            history[k].append(v)

        print(f"  [F{fold_num} E{epoch:3d}] loss {v_loss:.3f} | "
              f"PatAUC {patient_auc:.3f} "
              f"(sens {snapshot['patient_sens']:.2f} spec {snapshot['patient_spec']:.2f}) | "
              f"GT_AUC {gt_auc:.3f} | Det_AUC {det_auc:.3f} | "
              f"R@0.25 {snapshot['recall@0.25']:.3f} | score {score:.4f}")

        if score > best_score:
            best_score = score
            best_snapshot = dict(snapshot)
            patience_ctr = 0
            atomic_torch_save({'epoch': epoch,
                               'model_state_dict': model.state_dict(),
                               'spatial_size': args.spatial_size,
                               'score': float(score)}, best_path)
            print(f"    * new best (score={score:.4f}, PatAUC={patient_auc:.3f}) -> saved")
        else:
            patience_ctr += 1

        if patient_auc > best_patient_auc:
            best_patient_auc = patient_auc
            best_patient_snapshot = dict(snapshot)

        # Full state, every epoch, atomically. This is the crash insurance.
        atomic_torch_save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_score': best_score, 'best_snapshot': best_snapshot,
            'best_patient_auc': best_patient_auc,
            'best_patient_snapshot': best_patient_snapshot,
            'patience_ctr': patience_ctr, 'history': dict(history),
            'rng': capture_rng(), 'args': vars(args),
        }, last_path)

        if patience_ctr >= args.patience:
            print(f"    early stopping at epoch {epoch}")
            break

    atomic_json_dump(dict(history), fold_dir / 'training_history.json')

    # OOF from the KEPT checkpoint, then persist immediately.
    oof_patients = {}
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        oof_patients = collect_oof_patient_predictions(
            model, val_loader, device, score_thresh=args.oof_score_thresh)
        print(f"  OOF patients collected: {len(oof_patients)} "
              f"(from best epoch {ck.get('epoch', '?')})")

    result = {'best_combined': best_snapshot,
              'best_patient_auc': best_patient_snapshot,
              'oof_patients': oof_patients, 'oof_recovered': True}
    atomic_json_dump(result, result_path)

    if not args.save_fold_checkpoints:
        for p in (best_path, last_path):
            if p.exists():
                p.unlink()

    del model, optimizer, scheduler, scaler, loss_fn
    del train_loader, val_loader, train_ds, val_ds, sampler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# =============================================================================
# Reporting
# =============================================================================

def mean_std_table(snapshots, title):
    snaps = [s for s in snapshots if s]
    print(f"\n  {title} (mean +/- std across {len(snaps)} folds)")
    print(f"  {'metric':<16}{'mean':>10}{'std':>10}")
    print(f"  {'-'*36}")
    agg = {}
    for k in AGG_KEYS:
        vals = [s[k] for s in snaps
                if isinstance(s.get(k), (int, float)) and not np.isnan(s[k])]
        if not vals:
            continue
        m, sd = float(np.mean(vals)), float(np.std(vals))
        agg[k] = {'mean': m, 'std': sd, 'per_fold': vals}
        print(f"  {k:<16}{m:>10.4f}{sd:>10.4f}")
    return agg


def main():
    ap = argparse.ArgumentParser(
        description="Crash-safe patient-grouped stratified k-fold CV.")
    ap.add_argument('--data_root', default='/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15')
    ap.add_argument('--save_dir', default='/mnt/e/DBT_Stage2_MambaCenterNet_v5.2_cv')
    ap.add_argument('--cv_splits', nargs='+', default=['train', 'validation'],
                    help="Split dirs pooled for CV. Default leaves the test set untouched.")
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--val_batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=5e-3)
    ap.add_argument('--spatial_size', type=int, default=384)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--max_boxes', type=int, default=20)
    ap.add_argument('--no_mamba', action='store_true')
    ap.add_argument('--backbone', default='resnet18', choices=['resnet18', 'mobilenet'])
    ap.add_argument('--oof_score_thresh', type=float, default=0.20)
    ap.add_argument('--save_fold_checkpoints', action='store_true')
    ap.add_argument('--resume', action='store_true',
                    help="Skip completed folds, continue an interrupted one.")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MambaCenterNet v5.2 - K-Fold CV (crash-safe)")
    print("=" * 70)
    print(f"  Data root:  {args.data_root}")
    print(f"  Save dir:   {args.save_dir}")
    print(f"  CV splits:  {args.cv_splits}"
          f"{'   (test set untouched)' if 'test' not in args.cv_splits else '   (ALL data pooled)'}")
    print(f"  Folds:      {args.folds}   | resume: {'ON' if args.resume else 'OFF'}")
    print(f"  Device:     {device}")
    print(f"  Batch:      {args.batch_size} (val {args.val_batch_size}) | "
          f"spatial {args.spatial_size}")
    print("=" * 70)

    cases, per = index_all_cases(args.data_root, args.cv_splits)
    if not cases:
        print("\n[fatal] No cases found. Check --data_root and --cv_splits.")
        sys.exit(1)

    n_pat = len(set(patient_id_of(c['case_id']) for c in cases))
    n_pos = sum(c['label'] for c in cases)
    print(f"\n  Pooled cases: {len(cases)} | patients: {n_pat} | "
          f"Cancer {n_pos} / Benign {len(cases) - n_pos}")

    try:
        folds, groups, labels, method, mixed = make_folds(cases, args.folds, args.seed)
    except ValueError as e:
        print(f"\n[fatal] Could not build {args.folds} folds: {e}")
        sys.exit(1)
    print(f"  Split method: {method}")

    # Fingerprint guard: refuse to blend incompatible runs.
    fp = {'data_root': args.data_root, 'cv_splits': list(args.cv_splits),
          'folds': args.folds, 'seed': args.seed, 'spatial_size': args.spatial_size,
          'backbone': args.backbone, 'no_mamba': bool(args.no_mamba),
          'n_cases': len(cases), 'n_patients': n_pat}
    state_path = save_dir / 'cv_state.json'
    if state_path.exists():
        try:
            with open(state_path) as f:
                old_fp = json.load(f).get('fingerprint', {})
            if old_fp and old_fp != fp:
                diff = {k: (old_fp.get(k), fp.get(k)) for k in fp
                        if old_fp.get(k) != fp.get(k)}
                print(f"\n[fatal] {state_path} is from a DIFFERENT configuration.")
                print(f"        Changed (old -> new): {diff}")
                print("        Folds would not match. Use a fresh --save_dir, or "
                      "restore the original args.")
                sys.exit(1)
        except SystemExit:
            raise
        except Exception:
            pass
    atomic_json_dump({'fingerprint': fp, 'split_method': method}, state_path)

    t0 = time.time()
    fold_results = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        tr_pids, va_pids = set(groups[tr_idx]), set(groups[va_idx])
        assert not (tr_pids & va_pids), \
            f"patient leakage in fold {k+1}: {len(tr_pids & va_pids)} shared patients"
        try:
            res = train_one_fold(k, [cases[i] for i in tr_idx],
                                 [cases[i] for i in va_idx], args, device)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            print("\n[fatal] CUDA OOM. Retry the SAME command with "
                  "--batch_size 2 --val_batch_size 1 (or --spatial_size 320). "
                  "Completed folds are on disk; --resume will skip them.")
            sys.exit(1)
        fold_results.append(res)

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    best_combined = [r['best_combined'] for r in fold_results]
    best_patient = [r['best_patient_auc'] for r in fold_results]

    print("\n  Per-fold patient AUC (model kept by combined score):")
    for i, s in enumerate(best_combined):
        if s:
            print(f"    fold {i+1}: PatAUC {s['patient_auc']:.4f} @ epoch {s['epoch']} "
                  f"(sens {s['patient_sens']:.3f}, spec {s['patient_spec']:.3f})")

    agg_combined = mean_std_table(best_combined,
                                  "AT BEST COMBINED SCORE (matches best_model.pt)")
    agg_patient = mean_std_table(best_patient,
                                 "AT BEST PATIENT AUC (matches best_patient_auc.pt)")

    # Pooled OOF, only over folds whose predictions exist on disk.
    merged, n_with_oof = {}, 0
    for r in fold_results:
        if r.get('oof_patients'):
            n_with_oof += 1
            merged.update(r['oof_patients'])

    pooled = None
    if merged:
        pm = compute_patient_metrics(merged)
        ci = bootstrap_auc_ci([d['label'] for d in merged.values()],
                              [d['prob'] for d in merged.values()], seed=args.seed)
        cm = pm['confusion_matrix']
        print(f"\n{'='*70}")
        print(f"POOLED OUT-OF-FOLD PATIENT METRICS ({len(merged)} patients, "
              f"{n_with_oof}/{len(fold_results)} folds)")
        print(f"{'='*70}")
        if n_with_oof < len(fold_results):
            print("  NOTE: folds recovered from the pre-crash run have no stored OOF")
            print("        predictions and are excluded here. Their per-fold metrics")
            print("        above ARE included. Re-run those folds for full pooling.")
        if ci:
            print(f"  Patient AUC: {pm['auc']:.4f}  "
                  f"(bootstrap mean {ci[0]:.3f}, 95% CI {ci[1]:.3f}-{ci[2]:.3f})")
        else:
            print(f"  Patient AUC: {pm['auc']:.4f}")
        print(f"  Accuracy:    {pm['accuracy']:.4f}")
        print(f"  Sensitivity: {pm['sensitivity']:.4f}")
        print(f"  Specificity: {pm['specificity']:.4f}")
        if isinstance(cm, np.ndarray) and cm.shape == (2, 2):
            print(f"  Confusion:   TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
        pooled = {'auc': pm['auc'], 'accuracy': pm['accuracy'],
                  'sensitivity': pm['sensitivity'], 'specificity': pm['specificity'],
                  'confusion_matrix': cm, 'n_patients': len(merged),
                  'folds_with_oof': n_with_oof, 'n_folds': len(fold_results),
                  'auc_bootstrap_mean': ci[0] if ci else None,
                  'auc_ci95_low': ci[1] if ci else None,
                  'auc_ci95_high': ci[2] if ci else None}

    atomic_json_dump({
        'args': vars(args), 'split_method': method, 'fingerprint': fp,
        'n_cases': len(cases), 'n_patients': n_pat,
        'mixed_label_patients': len(mixed), 'elapsed_seconds': elapsed,
        'per_fold': {'best_combined': best_combined,
                     'best_patient_auc': best_patient},
        'aggregate_at_best_combined': agg_combined,
        'aggregate_at_best_patient_auc': agg_patient,
        'pooled_oof_patient': pooled,
    }, save_dir / 'cv_results.json')

    print(f"\n  Wall time this session: {elapsed/3600:.2f} h")
    print(f"  Results saved: {save_dir / 'cv_results.json'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
