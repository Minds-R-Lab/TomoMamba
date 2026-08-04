"""
Controlled Baseline Training Script
====================================
Train baselines 1, 2, or 3 under IDENTICAL conditions to MambaCenterNet v5.2.
Only the architecture changes. Everything else is fixed:
  - Same data split, preprocessing, augmentation
  - Same optimizer (AdamW), LR schedule (ReduceLROnPlateau)
  - Same loss weights, batch size, seed
  - Same evaluation protocol (patient-level AUC, detection metrics)

USAGE:
    python train_baselines.py --baseline 1  # ResNet-18 Classifier only
    python train_baselines.py --baseline 2  # ResNet-18 + CenterNet (no cross-slice)
    python train_baselines.py --baseline 3  # ResNet-18 + BiGRU + CenterNet

All baselines save to /mnt/e/DBT_Stage2_Baseline_{1,2,3}/
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

# Import baseline models
from centernet_baselines import create_baseline, ResNet18Classifier

# Import shared utilities from main codebase
from centernet_models import (CenterNetLoss, compute_detection_metrics,
                               aggregate_per_patient, compute_patient_metrics)


# =============================================================================
# DATASET — identical to v5.2
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
        if random.random() < 0.5:
            volume = volume[:, :, ::-1].copy()
            for i in range(len(boxes)):
                bx = boxes[i][1]
                bw = boxes[i][3]
                boxes[i][1] = 1.0 - bx - bw
        if random.random() < 0.3:
            volume = volume[:, ::-1, :].copy()
            for i in range(len(boxes)):
                by = boxes[i][2]
                bh = boxes[i][4]
                boxes[i][2] = 1.0 - by - bh
        if random.random() < 0.5:
            gamma = random.uniform(0.7, 1.3)
            mn, mx = volume.min(), volume.max()
            if mx - mn > 1e-6:
                volume = ((volume - mn) / (mx - mn)) ** gamma * (mx - mn) + mn
        if random.random() < 0.5:
            noise = np.random.normal(0, random.uniform(0.01, 0.05), volume.shape).astype(np.float32)
            volume = volume + noise
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
# METRIC TRACKER — identical to v5.2
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
        if not self.case_ids:
            return None
        probs = np.array(self.probs)
        labels = np.array(self.labels)
        patient_results = aggregate_per_patient(self.case_ids, probs, labels)
        return compute_patient_metrics(patient_results), patient_results


# =============================================================================
# BASELINE 1: Classification-only training
# =============================================================================

def train_epoch_cls_only(model, dataloader, optimizer, scaler, device, use_amp=True):
    """Train Baseline 1 — classification only, no detection."""
    model.train()
    total_loss = 0
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Training (cls-only)", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=use_amp):
            output = model(batch)
            # Focal cross-entropy for consistency with v5.2
            logits = output['vol_logits']
            labels = batch['labels']
            # Class weights: cancer is rarer
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
            # Focal modulation
            pt = torch.exp(-F.cross_entropy(logits, labels, reduction='none'))
            focal_loss = ((1 - pt) ** 2.0 * F.cross_entropy(logits, labels, reduction='none',
                                                              label_smoothing=0.1)).mean()
            loss = focal_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = output['vol_probs'].argmax(dim=-1)
        tracker.update(preds, batch['labels'], output['vol_probs'])
        pbar.set_postfix(loss=f'{loss.item():.3f}')

    n = len(dataloader)
    return total_loss / n, tracker.compute()


@torch.no_grad()
def eval_epoch_cls_only(model, dataloader, device, use_amp=True):
    """Evaluate Baseline 1 — classification only."""
    model.eval()
    total_loss = 0
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Validating (cls-only)", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with autocast(device_type='cuda', enabled=use_amp):
            output = model(batch)
            logits = output['vol_logits']
            loss = F.cross_entropy(logits, batch['labels'], label_smoothing=0.1)

        total_loss += loss.item()
        preds = output['vol_probs'].argmax(dim=-1)
        tracker.update(preds, batch['labels'], output['vol_probs'],
                       case_ids=batch['case_id'])

    n = len(dataloader)
    cls_metrics = tracker.compute()
    patient_result = tracker.compute_patient_level()
    patient_metrics = patient_result[0] if patient_result else None

    return total_loss / n, cls_metrics, patient_metrics


# =============================================================================
# BASELINES 2 & 3: Detection + Classification training (same as v5.2)
# =============================================================================

def train_epoch_det(model, dataloader, loss_fn, optimizer, scaler, device, use_amp=True):
    """Train Baselines 2/3 — detection + ROI classification."""
    model.train()
    total_loss = 0
    comps = defaultdict(float)
    tracker = MetricTracker()

    pbar = tqdm(dataloader, desc="Training (det+cls)", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()

        with autocast(device_type='cuda', enabled=use_amp):
            output = model(batch)
            cls_output = model.extract_and_classify_rois(
                output['feat_maps'], batch['boxes'], batch['box_masks'], batch['labels'])
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

        vol_preds = cls_output['vol_probs'].argmax(dim=-1)
        tracker.update(vol_preds, batch['labels'], cls_output['vol_probs'])
        pbar.set_postfix(loss=f'{loss.item():.3f}',
                         hm=f'{losses["heatmap"].item():.3f}',
                         cls=f'{losses["cls"].item():.3f}')

    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in comps.items()}, tracker.compute()


@torch.no_grad()
def eval_epoch_det(model, dataloader, loss_fn, device, use_amp=True, spatial_size=384):
    """Evaluate Baselines 2/3 — detection + classification."""
    model.eval()
    total_loss = 0
    comps = defaultdict(float)
    gt_tracker = MetricTracker()
    det_tracker = MetricTracker()
    all_pb, all_ps, all_psl, all_gb, all_gm = [], [], [], [], []

    pbar = tqdm(dataloader, desc="Validating (det+cls)", leave=False)
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

        gt_preds = cls_output['vol_probs'].argmax(dim=-1)
        gt_tracker.update(gt_preds, batch['labels'], cls_output['vol_probs'],
                          case_ids=batch['case_id'])

        with autocast(device_type='cuda', enabled=use_amp):
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
    patient_result = det_tracker.compute_patient_level()
    patient_metrics = patient_result[0] if patient_result else None

    return (total_loss / n, {k: v / n for k, v in comps.items()},
            gt_cls, det_cls, det_metrics, patient_metrics)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train controlled baselines")
    parser.add_argument('--baseline', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Baseline number: 1=ResNet-18 Cls, 2=ResNet-18+CenterNet, '
                             '3=ResNet-18+BiGRU+CenterNet, 4=ResNet-18+Transformer+CenterNet')
    parser.add_argument('--data_root', default='/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15')
    parser.add_argument('--save_dir', default=None,
                        help='Override save directory (default: /mnt/e/DBT_Stage2_Baseline_{N})')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--spatial_size', type=int, default=384)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Default save dir
    if args.save_dir is None:
        args.save_dir = f'/mnt/e/DBT_Stage2_Baseline_{args.baseline}'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    baseline_names = {
        1: "ResNet-18 Classifier (no detection, no cross-slice)",
        2: "ResNet-18 + CenterNet (no cross-slice)",
        3: "ResNet-18 + BiGRU + CenterNet",
    }

    print("=" * 70)
    print(f"CONTROLLED BASELINE {args.baseline}: {baseline_names.get(args.baseline, "ResNet-18 + Transformer + CenterNet")}")
    print("Identical conditions to MambaCenterNet v5.2 — only architecture differs")
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
    print(f"  Seed:          {args.seed}")
    print("=" * 70)

    # Data — identical to v5.2
    train_ds = DBTStage2Dataset(args.data_root, 'train', args.spatial_size, augment=True)
    val_ds   = DBTStage2Dataset(args.data_root, 'validation', args.spatial_size, augment=False)

    sampler = WeightedRandomSampler(train_ds.get_class_weights(), len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler, num_workers=4,
                              collate_fn=collate_fn, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_fn, pin_memory=True)

    # Model
    model = create_baseline(args.baseline, num_classes=2, dropout=0.7,
                             spatial_size=args.spatial_size).to(device)

    is_cls_only = isinstance(model, ResNet18Classifier)

    # Optimizer — match v5.2 param groups structure
    if is_cls_only:
        # Baseline 1: backbone (slow LR) + classifier (fast LR)
        backbone_params = list(model.backbone.parameters())
        cls_params = list(model.classifier.parameters()) + list(model.fc.parameters())
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': cls_params, 'lr': args.lr * 5.0},
        ], weight_decay=args.weight_decay)
    else:
        # Baselines 2/3: backbone (slow) + detection (normal) + classifier (fast)
        backbone_params = list(model.backbone.parameters())
        detect_params = list(model.detect_head.parameters())
        if model.cross_slice is not None:
            detect_params += list(model.cross_slice.parameters())
        cls_params = list(model.roi_classifier.parameters()) + \
                     list(model.global_classifier.parameters())
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},
            {'params': detect_params, 'lr': args.lr},
            {'params': cls_params, 'lr': args.lr * 5.0},
        ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)
    scaler = GradScaler('cuda')
    writer = SummaryWriter(str(save_dir / 'logs'))

    # Loss function for detection baselines
    loss_fn = None
    if not is_cls_only:
        loss_fn = CenterNetLoss(
            cls_weight=5.0,
            size_weight=0.3,
            offset_weight=1.0,
            peak_reg_weight=0.3,
            min_radius=3,
            cls_focal_gamma=2.0,
            label_smoothing=0.1,
        )

    best_score = 0
    best_patient_auc = 0
    patience_ctr = 0
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # =================== TRAINING ===================
        if is_cls_only:
            t_loss, t_cls = train_epoch_cls_only(model, train_loader, optimizer, scaler, device)
            print(f"  Train — Loss: {t_loss:.4f} | Acc: {t_cls['accuracy']:.4f} | AUC: {t_cls['auc']:.4f}")
        else:
            t_loss, t_comp, t_cls = train_epoch_det(model, train_loader, loss_fn,
                                                     optimizer, scaler, device)
            print(f"  Train — Loss: {t_loss:.4f} | Cls: {t_comp.get('cls',0):.4f} | "
                  f"HM: {t_comp.get('heatmap',0):.4f} | Acc: {t_cls['accuracy']:.4f}")

        # =================== VALIDATION ===================
        if is_cls_only:
            v_loss, v_cls, v_patient = eval_epoch_cls_only(model, val_loader, device)
            print(f"  Val   — Loss: {v_loss:.4f} | Acc: {v_cls['accuracy']:.4f} | AUC: {v_cls['auc']:.4f}")
            v_det = {}  # No detection metrics
            patient_auc = v_patient['auc'] if v_patient else 0.5
        else:
            v_loss, v_comp, v_gt_cls, v_det_cls, v_det, v_patient = \
                eval_epoch_det(model, val_loader, loss_fn, device, spatial_size=args.spatial_size)
            print(f"  Val(GT) — Loss: {v_loss:.4f} | Acc: {v_gt_cls['accuracy']:.4f} | AUC: {v_gt_cls['auc']:.4f}")
            print(f"  Val(Det)— Acc: {v_det_cls['accuracy']:.4f} | AUC: {v_det_cls['auc']:.4f}")
            print(f"  Det   — R@0.1: {v_det.get('recall@0.1',0):.3f} | "
                  f"R@0.25: {v_det.get('recall@0.25',0):.3f} | "
                  f"R@0.5: {v_det.get('recall@0.5',0):.3f} | "
                  f"mIoU: {v_det.get('mean_best_iou',0):.3f}")
            patient_auc = v_patient['auc'] if v_patient else 0.5

        # Patient metrics
        if v_patient:
            print(f"  Patient— Acc: {v_patient['accuracy']:.4f} | AUC: {v_patient['auc']:.4f} | "
                  f"Sens: {v_patient['sensitivity']:.4f} | Spec: {v_patient['specificity']:.4f} | "
                  f"N={v_patient['n_patients']}")
            pcm = v_patient.get('confusion_matrix', np.zeros((2,2)))
            if isinstance(pcm, np.ndarray) and pcm.shape == (2,2):
                print(f"  PCM   — TN={pcm[0,0]} FP={pcm[0,1]} FN={pcm[1,0]} TP={pcm[1,1]}")

        # LR
        scheduler.step(patient_auc)
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"  LR    — {' | '.join(f'{lr:.2e}' for lr in current_lrs)}")

        # TensorBoard
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        if v_patient:
            writer.add_scalar('Patient/auc', v_patient['auc'], epoch)
            writer.add_scalar('Patient/sensitivity', v_patient['sensitivity'], epoch)
            writer.add_scalar('Patient/specificity', v_patient['specificity'], epoch)
        if v_det:
            writer.add_scalar('Det/recall@0.25', v_det.get('recall@0.25', 0), epoch)

        # History
        history['epoch'].append(epoch)
        history['train_loss'].append(float(t_loss))
        history['val_loss'].append(float(v_loss))
        if v_patient:
            history['patient_auc'].append(float(v_patient['auc']))
            history['patient_sens'].append(float(v_patient['sensitivity']))
            history['patient_spec'].append(float(v_patient['specificity']))

        # Scoring — same formula as v5.2 (detection terms = 0 for Baseline 1)
        if is_cls_only:
            score = patient_auc  # Only classification available
        else:
            det_auc = v_det_cls['auc']
            gt_auc = v_gt_cls['auc']
            score = (patient_auc * 0.40 +
                     gt_auc * 0.10 +
                     det_auc * 0.15 +
                     v_det.get('recall@0.25', 0) * 0.15 +
                     v_det.get('recall@0.1', 0) * 0.10 +
                     v_det.get('mean_best_iou', 0) * 0.10)

        # Checkpointing
        checkpoint_data = {
            'epoch': epoch,
            'baseline': args.baseline,
            'baseline_name': baseline_names.get(args.baseline, "ResNet-18 + Transformer + CenterNet"),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'patient_metrics': v_patient,
            'det_metrics': v_det if v_det else None,
            'score': score,
            'patient_auc': patient_auc,
            'spatial_size': args.spatial_size,
            'args': vars(args),
        }

        if score > best_score:
            best_score = score
            patience_ctr = 0
            torch.save(checkpoint_data, save_dir / 'best_model.pt')
            print(f"  ★ New best! Score={score:.4f} (PatAUC={patient_auc:.3f})")
        else:
            patience_ctr += 1

        if patient_auc > best_patient_auc:
            best_patient_auc = patient_auc
            torch.save(checkpoint_data, save_dir / 'best_patient_auc.pt')
            print(f"  ★ New best patient AUC! {patient_auc:.4f}")

        if patience_ctr >= args.patience:
            print(f"\n  ⚠ Early stopping at epoch {epoch}")
            break

    # Save history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(dict(history), f, indent=2, default=float)

    writer.close()
    print(f"\n{'='*70}")
    print(f"BASELINE {args.baseline} COMPLETE: {baseline_names.get(args.baseline, "ResNet-18 + Transformer + CenterNet")}")
    print(f"Best combined score: {best_score:.4f}")
    print(f"Best patient AUC:    {best_patient_auc:.4f}")
    print(f"Checkpoints:         {save_dir / 'best_model.pt'}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()