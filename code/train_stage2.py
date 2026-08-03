"""
Training Script for MambaCenterNet v5.2
=======================================
Changes from v5.1:
1. Scoring: patient_auc is PRIMARY metric (40% weight, was 10%)
2. Dual checkpoints: best_model.pt (combined) + best_patient_auc.pt (classification)
3. ReduceLROnPlateau tracking patient_auc (was CosineAnnealing)
4. Stronger regularization: weight_decay=5e-3, classifier dropout=0.7
5. Shorter patience: 20 (was 40) — stop before severe overfitting
6. EMA model for stable evaluation
7. Model architecture unchanged — same ROI classification + detection

USAGE:
    python train_stage2.py --spatial_size 384
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import cv2

from centernet_models import (MambaCenterNet, CenterNetLoss,
                               compute_detection_metrics,
                               aggregate_per_patient, compute_patient_metrics)


# =============================================================================
# DATASET (unchanged from v5 — copy-paste aug, max_boxes=20)
# =============================================================================

class DBTStage2Dataset(Dataset):
    CLASS_MAP = {'Benign': 0, 'Cancer': 1}

    def __init__(self, data_root, split='train', spatial_size=384, max_boxes=20, augment=True):
        self.data_root = Path(data_root)
        self.split = split
        self.spatial_size = spatial_size
        self.max_boxes = max_boxes
        self.augment = augment and (split == 'train')
        self.cases = []
        self._index_cases()
        self.box_crops = []
        if self.augment:
            self._build_box_crop_pool()
        self._print_stats()

    def _index_cases(self):
        split_dir = self.data_root / self.split
        meta_dir = self.data_root / 'metadata' / self.split
        for class_name in ['Benign', 'Cancer']:
            class_dir = split_dir / class_name
            meta_class_dir = meta_dir / class_name
            if not class_dir.exists():
                continue
            for npy_path in sorted(class_dir.glob("*.npy")):
                meta_path = meta_class_dir / f"{npy_path.stem}.json"
                if not meta_path.exists():
                    meta_path = meta_class_dir / f"{npy_path.stem.lower()}.json"
                self.cases.append({
                    'npy_path': str(npy_path),
                    'meta_path': str(meta_path) if meta_path.exists() else None,
                    'case_id': npy_path.stem,
                    'class_name': class_name,
                    'label': self.CLASS_MAP[class_name],
                })

        # --- CV manifest filtering (added by patch_dataset_manifest.py) ---
        import os as _os, json as _json
        # When CV_MERGE_SPLITS=1, also index the OTHER biopsied split so that a
        # fold's training set can include patients that originally lived in the
        # 'validation' folder (and vice versa). Manifest filtering below then
        # keeps only this fold's assigned case_ids.
        if _os.environ.get("CV_MERGE_SPLITS") == "1":
            _other = "validation" if self.split == "train" else "train"
            _other_split_dir = self.data_root / _other
            _other_meta_dir = self.data_root / "metadata" / _other
            for _cls in ["Benign", "Cancer"]:
                _cd = _other_split_dir / _cls
                _mcd = _other_meta_dir / _cls
                if not _cd.exists():
                    continue
                _have = {c["case_id"] for c in self.cases}
                for _np in sorted(_cd.glob("*.npy")):
                    if _np.stem in _have:
                        continue
                    _mp = _mcd / f"{_np.stem}.json"
                    if not _mp.exists():
                        _mp = _mcd / f"{_np.stem.lower()}.json"
                    self.cases.append({
                        'npy_path': str(_np),
                        'meta_path': str(_mp) if _mp.exists() else None,
                        'case_id': _np.stem,
                        'class_name': _cls,
                        'label': self.CLASS_MAP[_cls],
                    })
        _mf = _os.environ.get(f"CV_MANIFEST_{self.split.upper()}")
        if _mf and _os.path.exists(_mf):
            with open(_mf) as _f:
                _allowed = set(_json.load(_f).get("case_ids", []))
            _before = len(self.cases)
            self.cases = [c for c in self.cases if c["case_id"] in _allowed]
            print(f"  [CV] {self.split}: filtered {_before} -> {len(self.cases)} "
                  f"cases via manifest {_mf}")

    def _build_box_crop_pool(self):
        print("  Building copy-paste crop pool...")
        for case in self.cases:
            if case['meta_path'] is None:
                continue
            try:
                with open(case['meta_path']) as f:
                    meta = json.load(f)
                vol = np.load(case['npy_path'], mmap_mode='r')
                selected = meta.get('selected_slices', list(range(vol.shape[0])))
                orig_h = meta.get('volume_shape', [1024, 1024])[0]
                orig_w = meta.get('volume_shape', [1024, 1024])[-1]
                for box in meta.get('boxes', []):
                    orig_slice = box.get('slice', box.get('original_slice',
                                         box.get('slice_idx', -1)))
                    if orig_slice in selected:
                        s_idx = selected.index(orig_slice)
                    elif 'slice_idx' in box:
                        s_idx = box['slice_idx']
                    else:
                        continue
                    if s_idx < 0 or s_idx >= vol.shape[0]:
                        continue
                    bx = int(float(box['x']))
                    by = int(float(box['y']))
                    bw = int(float(box['width']))
                    bh = int(float(box['height']))
                    if bx <= 1.0 and by <= 1.0:
                        bx, by = int(bx * orig_w), int(by * orig_h)
                        bw, bh = int(bw * orig_w), int(bh * orig_h)
                    pad = max(bw, bh) // 4
                    y1 = max(0, by - pad)
                    y2 = min(orig_h, by + bh + pad)
                    x1 = max(0, bx - pad)
                    x2 = min(orig_w, bx + bw + pad)
                    crop = np.array(vol[s_idx, y1:y2, x1:x2]).copy()
                    if crop.size > 0:
                        self.box_crops.append({
                            'crop': crop, 'rel_x': bx - x1, 'rel_y': by - y1,
                            'w': bw, 'h': bh, 'class': case['class_name'],
                        })
            except Exception:
                pass
        print(f"  ✓ {len(self.box_crops)} crops extracted for copy-paste augmentation")

    def _print_stats(self):
        counts = defaultdict(int)
        n_boxes = 0
        for c in self.cases:
            counts[c['class_name']] += 1
            if c['meta_path']:
                try:
                    with open(c['meta_path']) as f:
                        n_boxes += len(json.load(f).get('boxes', []))
                except Exception:
                    pass
        print(f"\n  Stage 2 Dataset [{self.split}]: {len(self.cases)} cases")
        for cls in sorted(counts):
            print(f"    {cls}: {counts[cls]}")
        print(f"    Total GT boxes: {n_boxes}")
        print(f"    Spatial size: {self.spatial_size}x{self.spatial_size}")
        print(f"    Augmentation: {'ON' if self.augment else 'OFF'}")

    def __len__(self):
        return len(self.cases)

    def _load_boxes(self, meta_path, num_slices):
        boxes = []
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            selected = meta.get('selected_slices', list(range(num_slices)))
            orig_h = meta.get('volume_shape', [1024, 1024])[0]
            orig_w = meta.get('volume_shape', [1024, 1024])[-1]
            for box in meta.get('boxes', []):
                orig_slice = box.get('slice', box.get('original_slice',
                                     box.get('slice_idx', -1)))
                if orig_slice in selected:
                    s_idx = selected.index(orig_slice)
                elif 'slice_idx' in box:
                    s_idx = box['slice_idx']
                else:
                    continue
                if s_idx < 0 or s_idx >= num_slices:
                    continue
                bx = float(box['x'])
                by = float(box['y'])
                bw = float(box['width'])
                bh = float(box['height'])
                if bx > 1.0 or by > 1.0 or bw > 1.0 or bh > 1.0:
                    bx /= orig_w
                    by /= orig_h
                    bw /= orig_w
                    bh /= orig_h
                bx = max(0, min(1 - 0.001, bx))
                by = max(0, min(1 - 0.001, by))
                bw = max(0.001, min(1 - bx, bw))
                bh = max(0.001, min(1 - by, bh))
                boxes.append([s_idx, bx, by, bw, bh])
        except Exception:
            pass
        return boxes

    def __getitem__(self, idx):
        case = self.cases[idx]
        volume = np.load(case['npy_path']).astype(np.float32)
        boxes = self._load_boxes(case['meta_path'], volume.shape[0]) if case['meta_path'] else []

        S, H, W = volume.shape
        if H != self.spatial_size or W != self.spatial_size:
            resized = np.zeros((S, self.spatial_size, self.spatial_size), dtype=np.float32)
            for s in range(S):
                resized[s] = cv2.resize(volume[s], (self.spatial_size, self.spatial_size),
                                        interpolation=cv2.INTER_LINEAR)
            volume = resized

        if self.augment:
            volume, boxes = self._augment(volume, boxes)

        lo = np.percentile(volume, 1)
        hi = np.percentile(volume, 99)
        if hi - lo > 1e-6:
            volume = (volume - lo) / (hi - lo)
        volume = np.clip(volume, 0, 1)

        box_tensor = torch.zeros(self.max_boxes, 5)
        box_mask = torch.zeros(self.max_boxes)
        for i, b in enumerate(boxes[:self.max_boxes]):
            box_tensor[i] = torch.tensor(b, dtype=torch.float32)
            box_mask[i] = 1.0

        return {
            'volume': torch.from_numpy(volume),
            'labels': torch.tensor(case['label'], dtype=torch.long),
            'boxes': box_tensor,
            'box_masks': box_mask,
            'case_id': case['case_id'],
            'class_name': case['class_name'],
        }

    def _augment(self, volume, boxes):
        S, H, W = volume.shape
        # Horizontal flip
        if random.random() < 0.5:
            volume = volume[:, :, ::-1].copy()
            for i in range(len(boxes)):
                bx = boxes[i][1]
                bw = boxes[i][3]
                boxes[i][1] = 1.0 - bx - bw

        # Vertical flip
        if random.random() < 0.3:
            volume = volume[:, ::-1, :].copy()
            for i in range(len(boxes)):
                by = boxes[i][2]
                bh = boxes[i][4]
                boxes[i][2] = 1.0 - by - bh

        # Intensity augmentation
        if random.random() < 0.5:
            gamma = random.uniform(0.7, 1.3)
            mn, mx = volume.min(), volume.max()
            if mx - mn > 1e-6:
                volume = ((volume - mn) / (mx - mn)) ** gamma * (mx - mn) + mn

        if random.random() < 0.5:
            noise = np.random.normal(0, random.uniform(0.01, 0.05), volume.shape).astype(np.float32)
            volume = volume + noise

        # Copy-paste augmentation
        if self.box_crops and random.random() < 0.5:
            crop_info = random.choice(self.box_crops)
            crop = crop_info['crop']
            target_s = random.randint(0, S - 1)
            gt_slices = set(int(b[0]) for b in boxes)
            if target_s not in gt_slices:
                ch, cw = crop.shape
                target_h = int(ch * (self.spatial_size / 1024))
                target_w = int(cw * (self.spatial_size / 1024))
                if target_h > 4 and target_w > 4:
                    resized_crop = cv2.resize(crop, (target_w, target_h))
                    py = random.randint(0, max(0, H - target_h))
                    px = random.randint(0, max(0, W - target_w))
                    eh = min(target_h, H - py)
                    ew = min(target_w, W - px)
                    alpha = 0.7
                    volume[target_s, py:py+eh, px:px+ew] = (
                        alpha * resized_crop[:eh, :ew] +
                        (1 - alpha) * volume[target_s, py:py+eh, px:px+ew])
                    rel_x = crop_info['rel_x'] * (self.spatial_size / 1024)
                    rel_y = crop_info['rel_y'] * (self.spatial_size / 1024)
                    box_w = crop_info['w'] * (self.spatial_size / 1024)
                    box_h = crop_info['h'] * (self.spatial_size / 1024)
                    new_bx = (px + rel_x) / W
                    new_by = (py + rel_y) / H
                    new_bw = box_w / W
                    new_bh = box_h / H
                    if 0 < new_bx < 1 and 0 < new_by < 1 and new_bw > 0.005 and new_bh > 0.005:
                        boxes.append([target_s, new_bx, new_by,
                                     min(new_bw, 1 - new_bx), min(new_bh, 1 - new_by)])

        return volume, boxes

    def get_class_weights(self):
        labels = [c['label'] for c in self.cases]
        counts = np.bincount(labels)
        w = 1.0 / counts.astype(np.float32)
        return [w[l] for l in labels]


def collate_fn(batch):
    return {
        'volume': torch.stack([b['volume'] for b in batch]),
        'labels': torch.stack([b['labels'] for b in batch]),
        'boxes': torch.stack([b['boxes'] for b in batch]),
        'box_masks': torch.stack([b['box_masks'] for b in batch]),
        'case_id': [b['case_id'] for b in batch],
        'class_name': [b['class_name'] for b in batch],
    }


# =============================================================================
# METRIC TRACKER
# =============================================================================

class MetricTracker:
    def __init__(self):
        self.preds, self.labels, self.probs = [], [], []
        self.case_ids = []

    def update(self, preds, labels, probs, case_ids=None):
        self.preds.extend(preds.detach().cpu().numpy().tolist())
        self.labels.extend(labels.detach().cpu().numpy().tolist())
        self.probs.extend(probs.detach().cpu().numpy().tolist())
        if case_ids:
            self.case_ids.extend(case_ids)

    def compute(self):
        preds, labels = np.array(self.preds), np.array(self.labels)
        probs = np.array(self.probs)
        try:
            auc = roc_auc_score(labels, probs[:, 1])
        except Exception:
            auc = 0.5
        return {
            'accuracy': accuracy_score(labels, preds),
            'auc': auc,
            'f1': f1_score(labels, preds, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(labels, preds, labels=[0, 1]),
        }

    def compute_patient_level(self):
        """Compute per-patient metrics by aggregating views."""
        if not self.case_ids:
            return None
        probs = np.array(self.probs)
        labels = np.array(self.labels)
        patient_results = aggregate_per_patient(self.case_ids, probs, labels)
        return compute_patient_metrics(patient_results), patient_results


# =============================================================================
# TRAIN / EVAL — v5.1: ROI classification in training loop
# =============================================================================

def train_epoch(model, dataloader, loss_fn, optimizer, scaler, device, use_amp=True):
    model.train()
    total_loss = 0
    comps = defaultdict(float)
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=use_amp):
            # Step 1: Forward — backbone + Mamba + detection heads
            output = model(batch)

            # Step 2: ROI classification using GT boxes
            cls_output = model.extract_and_classify_rois(
                output['feat_maps'], batch['boxes'], batch['box_masks'], batch['labels'])

            # Step 3: Loss (detection + ROI classification)
            losses = loss_fn(output, cls_output, batch)
            loss = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k, v in losses.items():
            if k != 'total':
                comps[k] += (v.item() if isinstance(v, torch.Tensor) else float(v))

        # Track volume-level predictions
        vol_preds = cls_output['vol_probs'].argmax(dim=-1)
        tracker.update(vol_preds, batch['labels'], cls_output['vol_probs'])

        pbar.set_postfix(loss=f'{loss.item():.3f}',
                         hm=f'{losses["heatmap"].item():.3f}',
                         cls=f'{losses["cls"].item():.3f}')

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in comps.items()}, tracker.compute()


@torch.no_grad()
def eval_epoch(model, dataloader, loss_fn, device, use_amp=True, spatial_size=384):
    model.eval()
    total_loss = 0
    comps = defaultdict(float)

    # View-level tracker (using GT boxes for classification — consistent with training)
    gt_tracker = MetricTracker()
    # View-level tracker (using DETECTED boxes for classification — realistic)
    det_tracker = MetricTracker()

    all_pb, all_ps, all_psl, all_gb, all_gm = [], [], [], [], []

    pbar = tqdm(dataloader, desc="Validating", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with autocast(device_type='cuda', enabled=use_amp):
            output = model(batch)
            cls_output = model.extract_and_classify_rois(
                output['feat_maps'], batch['boxes'], batch['box_masks'], batch['labels'])
            losses = loss_fn(output, cls_output, batch)

        total_loss += losses['total'].item()
        for k, v in losses.items():
            if k != 'total':
                comps[k] += (v.item() if isinstance(v, torch.Tensor) else float(v))

        # GT-based classification (for loss comparison / upper bound)
        gt_preds = cls_output['vol_probs'].argmax(dim=-1)
        gt_tracker.update(gt_preds, batch['labels'], cls_output['vol_probs'],
                          case_ids=batch['case_id'])

        # Detection + ROI classification (realistic evaluation)
        # Training-time monitoring only. All detection metrics reported in the
        # Reported detection metrics use tau_det = 0.16 via eval_variant.py.
        dets = model.detect(batch['volume'], score_thresh=0.20)
        det_probs = model.classify_detections(output['feat_maps'], dets)
        det_preds = det_probs.argmax(dim=-1)
        det_tracker.update(det_preds, batch['labels'], det_probs,
                           case_ids=batch['case_id'])

        for i in range(batch['volume'].shape[0]):
            all_pb.append(dets[i]['boxes'])
            all_ps.append(dets[i]['scores'])
            all_psl.append(dets[i]['slice_indices'])
            all_gb.append(batch['boxes'][i].cpu())
            all_gm.append(batch['box_masks'][i].cpu())

    n = len(dataloader)
    det_metrics = compute_detection_metrics(
        all_pb, all_ps, all_psl, all_gb, all_gm,
        iou_thresholds=[0.1, 0.25, 0.5], input_size=spatial_size)

    gt_cls = gt_tracker.compute()
    det_cls = det_tracker.compute()

    # Per-patient metrics
    patient_result = det_tracker.compute_patient_level()
    patient_metrics = patient_result[0] if patient_result else None

    return (total_loss / n, {k: v / n for k, v in comps.items()},
            gt_cls, det_cls, det_metrics, patient_metrics)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15')
    parser.add_argument('--save_dir', default='/mnt/e/DBT_Stage2_MambaCenterNet_v5.2')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-3)  # v5.2: 5× stronger
    parser.add_argument('--spatial_size', type=int, default=384)
    parser.add_argument('--patience', type=int, default=20)  # v5.2: stop earlier
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_mamba', action='store_true')
    parser.add_argument('--backbone', default='resnet18', choices=['resnet18', 'mobilenet'])
    parser.add_argument('--ema_decay', type=float, default=0.999)  # v5.2: EMA
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MambaCenterNet v5.2 — Stage 2 Training")
    print("FIX: Patient AUC-focused scoring + stronger regularization + EMA")
    print("Architecture unchanged from v5.1 (ROI classification + detection)")
    print("=" * 70)
    print(f"  Data root:     {args.data_root}")
    print(f"  Save dir:      {args.save_dir}")
    print(f"  Device:        {device}")
    print(f"  Spatial:       {args.spatial_size}x{args.spatial_size}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch:         {args.batch_size}")
    print(f"  LR:            {args.lr}")
    print(f"  Weight decay:  {args.weight_decay}")
    print(f"  Patience:      {args.patience}")
    print(f"  EMA decay:     {args.ema_decay}")
    # print(f"  Mamba:         {'Disabled' if args.no_mamba else 'Enabled'}")
    print(f"  Mamba:         {'Disabled' if args.no_mamba else 'Enabled'}")
    print(f"  Backbone:      {args.backbone}")
    print("=" * 70)

    train_ds = DBTStage2Dataset(args.data_root, 'train', args.spatial_size, augment=True)
    val_ds   = DBTStage2Dataset(args.data_root, 'validation', args.spatial_size, augment=False)

    sampler = WeightedRandomSampler(train_ds.get_class_weights(), len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler, num_workers=4,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_fn, pin_memory=True)

    # model = MambaCenterNet(num_classes=2, dropout=0.7,
    #                        use_mamba=not args.no_mamba,
    #                        spatial_size=args.spatial_size).to(device)

    # # v5.2: EMA model for stable evaluation
    # ema_model = MambaCenterNet(num_classes=2, dropout=0.7,
    #                            use_mamba=not args.no_mamba,
    #                            spatial_size=args.spatial_size).to(device)
    model = MambaCenterNet(num_classes=2, dropout=0.7,
                           use_mamba=not args.no_mamba,
                           spatial_size=args.spatial_size,
                           backbone=args.backbone).to(device)

    # v5.2: EMA model for stable evaluation
    ema_model = MambaCenterNet(num_classes=2, dropout=0.7,
                               use_mamba=not args.no_mamba,
                               spatial_size=args.spatial_size,
                               backbone=args.backbone).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {total_p:,} (trainable: {train_p:,})")

    loss_fn = CenterNetLoss(
        cls_weight=5.0,              # v5.1: strong classification signal
        size_weight=0.3,
        offset_weight=1.0,
        peak_reg_weight=0.3,
        min_radius=3,
        cls_focal_gamma=2.0,
        label_smoothing=0.1,
    )

    # v5.1 param groups (unchanged):
    backbone_params = list(model.backbone.parameters())
    detect_params = (list(model.cross_slice.parameters()) if model.cross_slice else []) + \
                    list(model.detect_head.parameters())
    cls_params = list(model.roi_classifier.parameters()) + \
                 list(model.global_classifier.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': detect_params, 'lr': args.lr},
        {'params': cls_params, 'lr': args.lr * 5.0},
    ], weight_decay=args.weight_decay)

    # v5.2: ReduceLROnPlateau tracking patient AUC (was CosineAnnealing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)
    scaler = GradScaler('cuda')
    writer = SummaryWriter(str(save_dir / 'logs'))

    best_score = 0
    best_patient_auc = 0  # v5.2: track best classification separately
    patience_ctr = 0
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        t_loss, t_comp, t_cls = train_epoch(model, train_loader, loss_fn,
                                             optimizer, scaler, device)

        # v5.2-fix: Evaluate training model directly
        # (EMA per-epoch at 0.999 decay was broken - model stayed ~random)
        v_loss, v_comp, v_gt_cls, v_det_cls, v_det, v_patient = \
            eval_epoch(model, val_loader, loss_fn, device, spatial_size=args.spatial_size)

        patient_auc = v_patient['auc'] if v_patient else 0.5

        # v5.2: LR scheduler tracks patient AUC
        scheduler.step(patient_auc)

        # Print training results
        print(f"  Train — Loss: {t_loss:.4f} | Cls: {t_comp.get('cls',0):.4f} | "
              f"HM: {t_comp.get('heatmap',0):.4f} | Size: {t_comp.get('size',0):.4f} | "
              f"Acc: {t_cls['accuracy']:.4f}")

        # Print validation results — GT-box classification (upper bound)
        print(f"  Val(GT) — Loss: {v_loss:.4f} | Cls: {v_comp.get('cls',0):.4f} | "
              f"Acc: {v_gt_cls['accuracy']:.4f} | AUC: {v_gt_cls['auc']:.4f}")

        # Print validation results — detected-box classification (realistic)
        print(f"  Val(Det)— Acc: {v_det_cls['accuracy']:.4f} | AUC: {v_det_cls['auc']:.4f}")

        # Detection metrics
        print(f"  Det   — R@0.1: {v_det.get('recall@0.1',0):.3f} | "
              f"R@0.25: {v_det.get('recall@0.25',0):.3f} | "
              f"R@0.5: {v_det.get('recall@0.5',0):.3f} | "
              f"mIoU: {v_det.get('mean_best_iou',0):.3f}")

        # Confusion matrices
        cm_gt = v_gt_cls.get('confusion_matrix', np.zeros((2,2)))
        cm_det = v_det_cls.get('confusion_matrix', np.zeros((2,2)))
        if isinstance(cm_gt, np.ndarray) and cm_gt.shape == (2,2):
            print(f"  CM(GT) — TN={cm_gt[0,0]} FP={cm_gt[0,1]} FN={cm_gt[1,0]} TP={cm_gt[1,1]}")
        if isinstance(cm_det, np.ndarray) and cm_det.shape == (2,2):
            print(f"  CM(Det)— TN={cm_det[0,0]} FP={cm_det[0,1]} FN={cm_det[1,0]} TP={cm_det[1,1]}")

        # Per-patient metrics
        if v_patient:
            print(f"  Patient— Acc: {v_patient['accuracy']:.4f} | AUC: {v_patient['auc']:.4f} | "
                  f"Sens: {v_patient['sensitivity']:.4f} | Spec: {v_patient['specificity']:.4f} | "
                  f"N={v_patient['n_patients']}")
            pcm = v_patient.get('confusion_matrix', np.zeros((2,2)))
            if isinstance(pcm, np.ndarray) and pcm.shape == (2,2):
                print(f"  PCM   — TN={pcm[0,0]} FP={pcm[0,1]} FN={pcm[1,0]} TP={pcm[1,1]}")

        # Print current LR
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"  LR    — backbone: {current_lrs[0]:.2e} | det: {current_lrs[1]:.2e} | cls: {current_lrs[2]:.2e}")

        # TensorBoard
        writer.add_scalars('Loss', {'train': t_loss, 'val': v_loss}, epoch)
        writer.add_scalars('Accuracy', {
            'train': t_cls['accuracy'],
            'val_gt': v_gt_cls['accuracy'],
            'val_det': v_det_cls['accuracy'],
        }, epoch)
        writer.add_scalar('AUC/val_gt', v_gt_cls['auc'], epoch)
        writer.add_scalar('AUC/val_det', v_det_cls['auc'], epoch)
        if v_patient:
            writer.add_scalar('Patient/accuracy', v_patient['accuracy'], epoch)
            writer.add_scalar('Patient/auc', v_patient['auc'], epoch)
            writer.add_scalar('Patient/sensitivity', v_patient['sensitivity'], epoch)
            writer.add_scalar('Patient/specificity', v_patient['specificity'], epoch)
        writer.add_scalar('Det/recall@0.1', v_det.get('recall@0.1', 0), epoch)
        writer.add_scalar('Det/recall@0.25', v_det.get('recall@0.25', 0), epoch)
        writer.add_scalar('Det/recall@0.5', v_det.get('recall@0.5', 0), epoch)
        writer.add_scalar('Det/mean_iou', v_det.get('mean_best_iou', 0), epoch)
        writer.add_scalar('LR/backbone', current_lrs[0], epoch)
        writer.add_scalar('LR/cls', current_lrs[2], epoch)

        # History
        history['epoch'].append(epoch)
        history['train_loss'].append(float(t_loss))
        history['val_loss'].append(float(v_loss))
        history['val_gt_acc'].append(float(v_gt_cls['accuracy']))
        history['val_gt_auc'].append(float(v_gt_cls['auc']))
        history['val_det_acc'].append(float(v_det_cls['accuracy']))
        history['val_det_auc'].append(float(v_det_cls['auc']))
        if v_patient:
            history['patient_acc'].append(float(v_patient['accuracy']))
            history['patient_auc'].append(float(v_patient['auc']))
            history['patient_sens'].append(float(v_patient['sensitivity']))
            history['patient_spec'].append(float(v_patient['specificity']))

        # v5.2 SCORING: patient AUC is PRIMARY metric
        det_auc = v_det_cls['auc']
        gt_auc = v_gt_cls['auc']

        score = (patient_auc * 0.40 +             # v5.2: PRIMARY (was 0.10)
                 gt_auc * 0.10 +                   # GT classification (upper bound)
                 det_auc * 0.15 +                  # Realistic classification
                 v_det.get('recall@0.25', 0) * 0.15 +
                 v_det.get('recall@0.1', 0) * 0.10 +
                 v_det.get('mean_best_iou', 0) * 0.10)

        # v5.2: DUAL CHECKPOINT — save both best combined AND best patient AUC
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # v5.2: save training model
            'raw_model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_gt_auc': v_gt_cls['auc'],
            'val_det_auc': v_det_cls['auc'],
            'val_gt_acc': v_gt_cls['accuracy'],
            'val_det_acc': v_det_cls['accuracy'],
            'val_det': v_det,
            'patient_metrics': v_patient,
            'score': score,
            'patient_auc': patient_auc,
            'spatial_size': args.spatial_size,
            'args': vars(args),
        }

        if score > best_score:
            best_score = score
            patience_ctr = 0
            torch.save(checkpoint_data, save_dir / 'best_model.pt')
            print(f"  ★ New best combined! Score={score:.4f} (PatAUC={patient_auc:.3f}, "
                  f"GT_AUC={gt_auc:.3f}, Det_AUC={det_auc:.3f}, "
                  f"R@0.25={v_det.get('recall@0.25',0):.3f})")
        else:
            patience_ctr += 1

        # v5.2: ALWAYS save best patient AUC checkpoint separately
        if patient_auc > best_patient_auc:
            best_patient_auc = patient_auc
            torch.save(checkpoint_data, save_dir / 'best_patient_auc.pt')
            print(f"  ★ New best patient AUC! {patient_auc:.4f} "
                  f"(Sens={v_patient['sensitivity']:.3f}, Spec={v_patient['specificity']:.3f})")

        if epoch % 25 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'spatial_size': args.spatial_size},
                       save_dir / f'checkpoint_epoch{epoch}.pt')

        if patience_ctr >= args.patience:
            print(f"\n  ⚠ Early stopping at epoch {epoch}")
            break

    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(dict(history), f, indent=2, default=float)

    writer.close()
    print(f"\n{'='*70}")
    print(f"Done! Best combined score: {best_score:.4f}")
    print(f"Best patient AUC: {best_patient_auc:.4f}")
    print(f"Checkpoints: {save_dir / 'best_model.pt'}")
    print(f"             {save_dir / 'best_patient_auc.pt'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()