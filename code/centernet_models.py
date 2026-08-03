"""
MambaCenterNet v5.1: ROI-Based Classification + Per-Patient Aggregation
========================================================================

ROOT CAUSE of v5 classification failure (20 epochs, cls_loss stuck at 0.693):
  Detection works great (R@0.1=0.707). Classifier outputs random 50/50.
  
  The dual-path classifier is dead because:
  1. Long gradient path: backbone → Mamba → pool → attention → fusion → classifier
  2. Detection loss dominates backbone (per-pixel vs 1 label/volume)
  3. Slice attention frozen 5 epochs → random when unfrozen → deadlock
  4. Classifier averages features across entire volume instead of looking at the LESION
  
  The SAME root cause explains v3's mediocre 64% accuracy:
  v3's heatmap-weighted path helped somewhat, but still pooled globally.

v5.1 FIX — Two key changes:
  1. ROI-based classification: crop features WHERE THE LESION IS, classify THOSE.
     Short gradient path: feat_map → crop → small CNN → loss. No attention/fusion/gate.
  2. Per-patient aggregation: group views by patient, max cancer prob across views.
     Clinically correct — radiologists look at all views together.
"""

import math
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# 1. BACKBONE — ResNet18 + FPN-lite (unchanged)
# =============================================================================

class SliceBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.up5 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.up4 = nn.Sequential(nn.Conv2d(256, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(128, 64, 1),  nn.BatchNorm2d(64),  nn.ReLU(True))
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.out_channels = 128

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        t = c2.shape[2:]
        c5_up = F.interpolate(self.up5(c5), size=t, mode='bilinear', align_corners=False)
        c4_up = F.interpolate(self.up4(c4), size=t, mode='bilinear', align_corners=False)
        c3_up = F.interpolate(self.up3(c3), size=t, mode='bilinear', align_corners=False)
        return self.fusion(torch.cat([c2, c3_up, c4_up, c5_up], dim=1))


# =============================================================================
# 2. MAMBA CROSS-SLICE (unchanged)
# =============================================================================

class MambaCrossSlicePropagation(nn.Module):
    def __init__(self, d_model=128, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.use_mamba = False
        try:
            from mamba_ssm import Mamba
            self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
            self.use_mamba = True
            print("  ✓ Using Mamba SSM for cross-slice propagation")
        except ImportError:
            print("  ⚠ mamba_ssm not available, using BiGRU fallback")
            self.ssm = nn.GRU(d_model, d_model // 2, 1, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, features):
        B, S, C, H, W = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, S, C)
        out = self.ssm(x) if self.use_mamba else self.ssm(x)[0]
        out = self.norm(out)
        gate = self.gate(torch.cat([x, out], dim=-1))
        enhanced = x + gate * out
        return enhanced.reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2)


# =============================================================================
# 3. CENTERNET HEAD — detection only (unchanged)
# =============================================================================

class CenterNetHead(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
        )
        self.heatmap = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 1, 1))
        self.heatmap[-1].bias.data.fill_(-2.19)
        self.size = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 2, 1), nn.Sigmoid())
        self.offset = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 2, 1))

    def forward(self, features):
        shared = self.shared(features)
        return {
            'heatmap': torch.sigmoid(self.heatmap(shared)),
            'size': self.size(shared),
            'offset': self.offset(shared),
        }


# =============================================================================
# 4. ROI CLASSIFIER (v5.1 NEW)
# =============================================================================

class ROIClassifier(nn.Module):
    """
    Classifies lesion ROIs extracted from feature maps.
    Short gradient path: crop → conv → pool → FC → loss
    """
    def __init__(self, in_channels=128, roi_size=7, num_classes=2, dropout=0.7):
        super().__init__()
        self.roi_size = roi_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # spatial dropout between convs
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, roi_features):
        return self.head(roi_features)


class GlobalClassifier(nn.Module):
    """Fallback when no boxes are detected at inference."""
    def __init__(self, in_channels=128, num_classes=2, dropout=0.7):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, feat_maps_5d):
        x = feat_maps_5d.mean(dim=1)  # avg across slices
        return self.head(x)


# =============================================================================
# 5. MAIN MODEL
# =============================================================================

class MambaCenterNet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.7, use_mamba=True,
                 spatial_size=384, roi_size=7, backbone='resnet18'):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_size = spatial_size
        self.stride = 4
        self.roi_size = roi_size
        if backbone == 'mobilenet':
            from mobilenet_backbone import MobileNetBackbone
            self.backbone = MobileNetBackbone(pretrained=True)
        else:
            self.backbone = SliceBackbone(pretrained=True)
        # self.backbone = SliceBackbone(pretrained=True)
        feat_dim = self.backbone.out_channels  # 128
        self.cross_slice = MambaCrossSlicePropagation(d_model=feat_dim) if use_mamba else None
        self.detect_head = CenterNetHead(in_channels=feat_dim)

        # v5.1: ROI-based classification
        self.roi_classifier = ROIClassifier(
            in_channels=feat_dim, roi_size=roi_size,
            num_classes=num_classes, dropout=dropout)
        self.global_classifier = GlobalClassifier(
            in_channels=feat_dim, num_classes=num_classes, dropout=dropout)

    def forward(self, batch):
        volume = batch['volume']
        B, S, H, W = volume.shape

        slices = volume.reshape(B * S, 1, H, W)
        slice_feats = self.backbone(slices)
        _, C, fH, fW = slice_feats.shape
        slice_feats = slice_feats.reshape(B, S, C, fH, fW)

        enhanced = self.cross_slice(slice_feats) if self.cross_slice is not None else slice_feats

        det_input = enhanced.reshape(B * S, C, fH, fW)
        det_out = self.detect_head(det_input)

        return {
            'heatmaps': det_out['heatmap'].reshape(B, S, 1, fH, fW),
            'sizes':    det_out['size'].reshape(B, S, 2, fH, fW),
            'offsets':  det_out['offset'].reshape(B, S, 2, fH, fW),
            'feat_maps': enhanced,  # (B, S, C, fH, fW) for ROI extraction
        }

    def _extract_single_roi(self, feat_map, bx, by, bw, bh, fH, fW, context_pad=0.3):
        """Extract one ROI from a single slice's feature map.
        bx, by = top-left corner, bw, bh = width/height, all in [0,1]."""
        pad_w = bw * context_pad
        pad_h = bh * context_pad
        x1 = max(0, int((bx - pad_w) * fW))
        y1 = max(0, int((by - pad_h) * fH))
        x2 = min(fW, int(math.ceil((bx + bw + pad_w) * fW)))
        y2 = min(fH, int(math.ceil((by + bh + pad_h) * fH)))
        # Ensure minimum 2×2
        if x2 - x1 < 2:
            cx = int((bx + bw / 2) * fW)
            x1, x2 = max(0, cx - 1), min(fW, cx + 2)
        if y2 - y1 < 2:
            cy = int((by + bh / 2) * fH)
            y1, y2 = max(0, cy - 1), min(fH, cy + 2)
        crop = feat_map[:, y1:y2, x1:x2]
        return F.interpolate(crop.unsqueeze(0), size=(self.roi_size, self.roi_size),
                             mode='bilinear', align_corners=False).squeeze(0)

    def extract_and_classify_rois(self, feat_maps, boxes, box_masks, labels):
        """
        Extract ROIs from GT box locations and classify them.

        Args:
            feat_maps: (B, S, C, fH, fW)
            boxes: (B, max_boxes, 5) — [slice_idx, bx, by, bw, bh] normalized
            box_masks: (B, max_boxes)
            labels: (B,) volume-level labels

        Returns dict with:
            roi_logits: (N_total_rois, num_classes)
            roi_labels: (N_total_rois,)
            vol_logits: (B, num_classes) — aggregated
            vol_probs:  (B, num_classes)
        """
        B, S, C, fH, fW = feat_maps.shape
        device = feat_maps.device

        all_rois = []
        all_labels = []
        roi_sample_idx = []

        for b in range(B):
            for i in range(boxes.shape[1]):
                if box_masks[b, i] < 0.5:
                    continue
                s_idx = int(boxes[b, i, 0].item())
                if s_idx < 0 or s_idx >= S:
                    continue
                bx, by, bw, bh = [boxes[b, i, j].item() for j in range(1, 5)]
                roi = self._extract_single_roi(
                    feat_maps[b, s_idx], bx, by, bw, bh, fH, fW)
                all_rois.append(roi)
                all_labels.append(labels[b].item())
                roi_sample_idx.append(b)

        if all_rois:
            rois_tensor = torch.stack(all_rois)  # (N, C, roi_size, roi_size)
            roi_logits = self.roi_classifier(rois_tensor)
            roi_labels = torch.tensor(all_labels, device=device, dtype=torch.long)
            roi_probs = F.softmax(roi_logits, dim=-1)

            # Volume-level: max cancer prob per sample
            vol_logits, vol_probs = [], []
            for b in range(B):
                idxs = [j for j, s in enumerate(roi_sample_idx) if s == b]
                if idxs:
                    sp = roi_probs[idxs]
                    best = sp[:, 1].argmax().item()
                    vol_logits.append(roi_logits[idxs[best]])
                    vol_probs.append(sp[best])
                else:
                    fb = self.global_classifier(feat_maps[b:b+1])
                    vol_logits.append(fb.squeeze(0))
                    vol_probs.append(F.softmax(fb.squeeze(0), dim=-1))
            vol_logits = torch.stack(vol_logits)
            vol_probs = torch.stack(vol_probs)
        else:
            roi_logits = torch.zeros(0, self.num_classes, device=device)
            roi_labels = torch.zeros(0, device=device, dtype=torch.long)
            vol_logits = self.global_classifier(feat_maps)
            vol_probs = F.softmax(vol_logits, dim=-1)

        return {
            'roi_logits': roi_logits,
            'roi_labels': roi_labels,
            'vol_logits': vol_logits,
            'vol_probs': vol_probs,
        }

    @torch.no_grad()
    def classify_detections(self, feat_maps, det_results):
        """Classify detected boxes at inference time."""
        B, S, C, fH, fW = feat_maps.shape
        device = feat_maps.device
        H = W = self.spatial_size
        vol_probs_list = []

        for b in range(B):
            dets = det_results[b]
            det_boxes = dets['boxes']
            det_slices = dets['slice_indices']

            if len(det_boxes) == 0:
                fb = self.global_classifier(feat_maps[b:b+1])
                vol_probs_list.append(F.softmax(fb.squeeze(0), dim=-1))
                continue

            rois = []
            for j in range(len(det_boxes)):
                s_idx = int(det_slices[j].item())
                if s_idx < 0 or s_idx >= S:
                    continue
                x1p, y1p, x2p, y2p = det_boxes[j].tolist()
                bx_n, by_n = x1p / W, y1p / H
                bw_n, bh_n = (x2p - x1p) / W, (y2p - y1p) / H
                roi = self._extract_single_roi(
                    feat_maps[b, s_idx], bx_n, by_n, bw_n, bh_n, fH, fW)
                rois.append(roi)

            if rois:
                rois_batch = torch.stack(rois)
                logits = self.roi_classifier(rois_batch)
                probs = F.softmax(logits, dim=-1)
                best = probs[:, 1].argmax()
                vol_probs_list.append(probs[best])
            else:
                fb = self.global_classifier(feat_maps[b:b+1])
                vol_probs_list.append(F.softmax(fb.squeeze(0), dim=-1))

        return torch.stack(vol_probs_list)

    # @torch.no_grad()
    # def detect(self, volume, score_thresh=0.20, nms_kernel=5, max_dets=15):
    @torch.no_grad()
    def detect(self, volume, score_thresh=0.20, nms_kernel=5, max_dets=15,
            box_method='size', rel_thresh=0.25, abs_thresh=0.05, max_radius=12):
        self.eval()
        output = self.forward({'volume': volume})
        B, S = volume.shape[:2]
        H, W = volume.shape[2], volume.shape[3]
        results = []

        for b in range(B):
            all_boxes, all_scores, all_slices = [], [], []
            for s in range(S):
                hm  = output['heatmaps'][b, s]
                sz  = output['sizes'][b, s]
                off = output['offsets'][b, s]
                hm_pool = F.max_pool2d(hm.unsqueeze(0), nms_kernel, stride=1,
                                       padding=nms_kernel // 2).squeeze(0)
                keep = (hm == hm_pool) & (hm >= score_thresh)
                positions = keep[0].nonzero()
                for pos in positions:
                    fy, fx = pos[0].item(), pos[1].item()
                    # score = hm[0, fy, fx].item()
                    # cx = (fx + off[0, fy, fx].item()) * self.stride
                    # cy = (fy + off[1, fy, fx].item()) * self.stride
                    # bw = sz[0, fy, fx].item() * W
                    # bh = sz[1, fy, fx].item() * H
                    # x1 = max(0, cx - bw / 2)
                    # y1 = max(0, cy - bh / 2)
                    # x2 = min(W, cx + bw / 2)
                    # y2 = min(H, cy + bh / 2)
                    # if (x2 - x1) > 2 and (y2 - y1) > 2:
                    #     all_boxes.append([x1, y1, x2, y2])
                    #     all_scores.append(score)
                    #     all_slices.append(s)
                    score = hm[0, fy, fx].item()

                    if box_method == 'size':
                        cx = (fx + off[0, fy, fx].item()) * self.stride
                        cy = (fy + off[1, fy, fx].item()) * self.stride
                        bw = sz[0, fy, fx].item() * W
                        bh = sz[1, fy, fx].item() * H
                        x1 = max(0, cx - bw / 2)
                        y1 = max(0, cy - bh / 2)
                        x2 = min(W, cx + bw / 2)
                        y2 = min(H, cy + bh / 2)
                        box = [x1, y1, x2, y2] if (x2 - x1) > 2 and (y2 - y1) > 2 else None

                    elif box_method == 'blob':
                        box = self._extract_box_from_heatmap_blob(
                            hm=hm[0],
                            peak_y=fy,
                            peak_x=fx,
                            peak_score=score,
                            stride=self.stride,
                            img_h=H,
                            img_w=W,
                            rel_thresh=rel_thresh,
                            abs_thresh=abs_thresh,
                            max_radius=max_radius,
                            min_box_size=2,
                        )

                    else:
                        raise ValueError(f"Unknown box_method: {box_method}")

                    if box is not None:
                        all_boxes.append(box)
                        all_scores.append(score)
                        all_slices.append(s)
            if all_boxes:
                scores_t = torch.tensor(all_scores)
                boxes_t = torch.tensor(all_boxes)
                slices_t = torch.tensor(all_slices)
                # Cross-slice NMS
                keep_mask = torch.ones(len(scores_t), dtype=torch.bool)
                order = scores_t.argsort(descending=True)
                for i in range(len(order)):
                    if not keep_mask[order[i]]:
                        continue
                    for j in range(i + 1, len(order)):
                        if not keep_mask[order[j]]:
                            continue
                        if abs(int(slices_t[order[i]]) - int(slices_t[order[j]])) <= 1:
                            iou = _compute_iou(boxes_t[order[i]].tolist(),
                                               boxes_t[order[j]].tolist())
                            if iou > 0.3:
                                keep_mask[order[j]] = False
                scores_t = scores_t[keep_mask]
                boxes_t = boxes_t[keep_mask]
                slices_t = slices_t[keep_mask]
                if len(scores_t) > max_dets:
                    topk = scores_t.argsort(descending=True)[:max_dets]
                    scores_t, boxes_t, slices_t = scores_t[topk], boxes_t[topk], slices_t[topk]
                results.append({'boxes': boxes_t, 'scores': scores_t, 'slice_indices': slices_t})
            else:
                results.append({'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0),
                                'slice_indices': torch.zeros(0, dtype=torch.long)})
        return results
    
    def _extract_box_from_heatmap_blob(
        self,
        hm,          # (fH, fW) single-channel heatmap for one slice
        peak_y,
        peak_x,
        peak_score,
        stride,
        img_h,
        img_w,
        rel_thresh=0.25,
        abs_thresh=0.05,
        max_radius=12,
        min_box_size=2,
    ):
        """
        Build a box from the connected heatmap blob around one peak.

        Returns:
            [x1, y1, x2, y2] in input-image pixel coordinates
            or None if blob is invalid.
        """
        fH, fW = hm.shape

        # Local crop around peak so one lesion does not swallow the whole map
        y1w = max(0, peak_y - max_radius)
        y2w = min(fH, peak_y + max_radius + 1)
        x1w = max(0, peak_x - max_radius)
        x2w = min(fW, peak_x + max_radius + 1)

        patch = hm[y1w:y2w, x1w:x2w]

        # Threshold: relative to peak, but not below abs_thresh
        thr = max(abs_thresh, rel_thresh * peak_score)
        binary = (patch >= thr)

        # Must include the peak itself
        py = peak_y - y1w
        px = peak_x - x1w
        if not binary[py, px]:
            binary[py, px] = True

        # Connected component containing the peak
        visited = torch.zeros_like(binary, dtype=torch.bool)
        q = [(py, px)]
        visited[py, px] = True
        coords = []

        while q:
            cy, cx = q.pop()
            if not binary[cy, cx]:
                continue
            coords.append((cy, cx))

            for ny, nx in [(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1),
                        (cy-1, cx-1), (cy-1, cx+1), (cy+1, cx-1), (cy+1, cx+1)]:
                if 0 <= ny < binary.shape[0] and 0 <= nx < binary.shape[1]:
                    if not visited[ny, nx]:
                        visited[ny, nx] = True
                        if binary[ny, nx]:
                            q.append((ny, nx))

        if len(coords) == 0:
            return None

        ys = [c[0] for c in coords]
        xs = [c[1] for c in coords]

        fx1 = x1w + min(xs)
        fy1 = y1w + min(ys)
        fx2 = x1w + max(xs) + 1
        fy2 = y1w + max(ys) + 1

        # Convert feature-map box -> image box
        x1 = fx1 * stride
        y1 = fy1 * stride
        x2 = fx2 * stride
        y2 = fy2 * stride

        # Clamp
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            return None

        return [float(x1), float(y1), float(x2), float(y2)]


# =============================================================================
# 6. LOSS — detection unchanged, ROI focal CE for classification
# =============================================================================

class CenterNetLoss(nn.Module):
    def __init__(self, hm_weight=1.0, size_weight=0.3, offset_weight=1.0,
                 cls_weight=5.0, focal_alpha=2.0, focal_beta=4.0,
                 peak_reg_weight=0.3, min_radius=3,
                 cls_focal_gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.hm_weight = hm_weight
        self.size_weight = size_weight
        self.offset_weight = offset_weight
        self.cls_weight = cls_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.peak_reg_weight = peak_reg_weight
        self.min_radius = min_radius
        self.cls_focal_gamma = cls_focal_gamma
        self.label_smoothing = label_smoothing

    def focal_loss_heatmap(self, pred, gt):
        pred = pred.clamp(1e-6, 1 - 1e-6)
        pos = gt.eq(1).float()
        neg = gt.lt(1).float()
        pos_loss = -((1 - pred) ** self.focal_alpha) * torch.log(pred) * pos
        neg_loss = -((1 - gt) ** self.focal_beta) * (pred ** self.focal_alpha) * torch.log(1 - pred) * neg
        return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1)

    def focal_cross_entropy(self, logits, targets):
        n_classes = logits.shape[1]
        with torch.no_grad():
            smooth = torch.full_like(logits, self.label_smoothing / n_classes)
            smooth.scatter_(1, targets.unsqueeze(1),
                           1.0 - self.label_smoothing + self.label_smoothing / n_classes)
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        ce = -(smooth * log_probs).sum(dim=1)
        pt = (smooth * probs).sum(dim=1)
        focal = ((1 - pt) ** self.cls_focal_gamma) * ce
        return focal.mean()

    def forward(self, output, cls_output, batch):
        device = output['heatmaps'].device
        heatmaps = output['heatmaps']
        sizes, offsets = output['sizes'], output['offsets']
        B, S = heatmaps.shape[0], heatmaps.shape[1]
        fH, fW = heatmaps.shape[3], heatmaps.shape[4]
        H, W = batch['volume'].shape[2], batch['volume'].shape[3]
        stride = H // fH

        gt_hm   = torch.zeros(B, S, 1, fH, fW, device=device)
        gt_sz   = torch.zeros(B, S, 2, fH, fW, device=device)
        gt_off  = torch.zeros(B, S, 2, fH, fW, device=device)
        gt_mask = torch.zeros(B, S, 1, fH, fW, device=device)
        boxes, box_masks = batch['boxes'], batch['box_masks']

        for b in range(B):
            for i in range(boxes.shape[1]):
                if box_masks[b, i] < 0.5:
                    continue
                box = boxes[b, i]
                s_idx = int(box[0].item())
                if s_idx < 0 or s_idx >= S:
                    continue
                bx, by = box[1].item() * W, box[2].item() * H
                bw, bh = box[3].item() * W, box[4].item() * H
                if bw < 1 or bh < 1:
                    continue
                cx, cy = bx + bw / 2, by + bh / 2
                fcx, fcy = cx / stride, cy / stride
                icx, icy = int(fcx), int(fcy)
                if icx < 0 or icx >= fW or icy < 0 or icy >= fH:
                    continue
                computed_r = self._gaussian_radius(bh / stride, bw / stride)
                radius = max(self.min_radius, int(computed_r))
                self._draw_gaussian(gt_hm[b, s_idx, 0], (icx, icy), radius)
                gt_sz[b, s_idx, 0, icy, icx] = bw / W
                gt_sz[b, s_idx, 1, icy, icx] = bh / H
                gt_off[b, s_idx, 0, icy, icx] = fcx - icx
                gt_off[b, s_idx, 1, icy, icx] = fcy - icy
                gt_mask[b, s_idx, 0, icy, icx] = 1.0

        hm_loss = self.focal_loss_heatmap(
            heatmaps.reshape(-1, 1, fH, fW), gt_hm.reshape(-1, 1, fH, fW))
        num_pos = gt_mask.sum().clamp(min=1)
        size_loss = F.l1_loss(sizes * gt_mask, gt_sz * gt_mask, reduction='sum') / num_pos
        offset_loss = F.l1_loss(offsets * gt_mask, gt_off * gt_mask, reduction='sum') / num_pos

        hm_flat = heatmaps.reshape(B * S, -1)
        gt_flat = gt_hm.reshape(B * S, -1)
        has_gt = gt_flat.max(dim=1)[0] > 0.5
        no_gt = ~has_gt
        peak_reg = hm_flat[no_gt].mean() if no_gt.any() else torch.tensor(0.0, device=device)

        # Classification: focal CE on ROI predictions
        roi_logits = cls_output['roi_logits']
        roi_labels = cls_output['roi_labels']
        if roi_logits.shape[0] > 0:
            cls_loss = self.focal_cross_entropy(roi_logits, roi_labels)
        else:
            cls_loss = F.cross_entropy(cls_output['vol_logits'], batch['labels'],
                                       label_smoothing=self.label_smoothing)

        total = (self.cls_weight * cls_loss +
                 self.hm_weight * hm_loss +
                 self.size_weight * size_loss +
                 self.offset_weight * offset_loss +
                 self.peak_reg_weight * peak_reg)

        return {
            'total': total,
            'cls': cls_loss.detach(),
            'heatmap': hm_loss.detach(),
            'size': size_loss.detach(),
            'offset': offset_loss.detach(),
            'peak_reg': peak_reg.detach(),
        }

    @staticmethod
    def _gaussian_radius(height, width, min_overlap=0.7):
        a1, b1 = 1, height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = max(0, b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1 ** 0.5) / 2
        a2, b2 = 4, 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = max(0, b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2 ** 0.5) / 2
        a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = max(0, b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3 ** 0.5) / 2
        return min(r1, r2, r3)

    @staticmethod
    def _draw_gaussian(heatmap, center, radius, k=1):
        d = 2 * radius + 1
        sigma = max(radius / 2.0, 0.5)
        coords = torch.arange(d, device=heatmap.device, dtype=torch.float32) - radius
        g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g_1d.unsqueeze(0) * g_1d.unsqueeze(1)
        x, y = center
        h, w = heatmap.shape
        l = min(x, radius)
        r = min(w - x, radius + 1)
        t = min(y, radius)
        b = min(h - y, radius + 1)
        if l + r <= 0 or t + b <= 0:
            return
        mh = heatmap[y - t:y + b, x - l:x + r]
        mg = g[radius - t:radius + b, radius - l:radius + r]
        if mh.shape == mg.shape:
            torch.max(mh, mg * k, out=mh)


# =============================================================================
# 7. PER-PATIENT AGGREGATION
# =============================================================================

def aggregate_per_patient(case_ids, probs, labels):
    """Group views by patient, take max cancer probability."""
    patients = defaultdict(lambda: {'views': [], 'view_probs': [], 'label': None})
    for i, case_id in enumerate(case_ids):
        parts = case_id.rsplit('_', 1)
        pid = parts[0] if len(parts) >= 2 else case_id
        cancer_prob = float(probs[i, 1]) if probs.ndim == 2 else float(probs[i])
        patients[pid]['views'].append(case_id)
        patients[pid]['view_probs'].append(cancer_prob)
        patients[pid]['label'] = int(labels[i])
    for pid in patients:
        patients[pid]['prob'] = max(patients[pid]['view_probs'])
        patients[pid]['pred'] = 1 if patients[pid]['prob'] > 0.5 else 0
    return dict(patients)


def compute_patient_metrics(patient_results):
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    preds = np.array([d['pred'] for d in patient_results.values()])
    labels = np.array([d['label'] for d in patient_results.values()])
    probs_arr = np.array([d['prob'] for d in patient_results.values()])
    try:
        auc = roc_auc_score(labels, probs_arr)
    except Exception:
        auc = 0.5
    acc = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return {
        'accuracy': acc, 'auc': auc,
        'sensitivity': tp / max(tp + fn, 1),
        'specificity': tn / max(tn + fp, 1),
        'confusion_matrix': cm,
        'n_patients': len(patient_results),
    }


# =============================================================================
# 8. DETECTION METRICS (unchanged)
# =============================================================================

def _compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def compute_detection_metrics(pred_boxes, pred_scores, pred_slices,
                               gt_boxes_list, gt_masks_list,
                               iou_thresholds=(0.1, 0.25, 0.5), input_size=384):
    metrics = {}
    all_ious = []
    for iou_thresh in iou_thresholds:
        tp = fp = fn = 0
        for pb, ps, psl, gb, gm in zip(pred_boxes, pred_scores, pred_slices,
                                        gt_boxes_list, gt_masks_list):
            gt_valid = []
            for i in range(len(gb)):
                if gm[i] > 0.5:
                    b = gb[i]
                    bx = float(b[1]) * input_size
                    by = float(b[2]) * input_size
                    bw = float(b[3]) * input_size
                    bh = float(b[4]) * input_size
                    gt_valid.append({'slice': int(b[0]),
                                     'box': [bx, by, bx + bw, by + bh]})
            if not gt_valid:
                fp += len(pb)
                continue
            if len(pb) == 0:
                fn += len(gt_valid)
                continue
            matched = set()
            order = ps.argsort(descending=True) if len(ps) > 0 else []
            for idx in order:
                p_box = pb[idx]
                p_sl = int(psl[idx].item())
                best_iou, best_gt = 0, -1
                for gi, g in enumerate(gt_valid):
                    if gi in matched:
                        continue
                    if abs(p_sl - g['slice']) > 2:
                        continue
                    iou = _compute_iou(p_box.tolist(), g['box'])
                    if iou > best_iou:
                        best_iou, best_gt = iou, gi
                all_ious.append(best_iou)
                if best_iou >= iou_thresh and best_gt >= 0:
                    tp += 1
                    matched.add(best_gt)
                else:
                    fp += 1
            fn += len(gt_valid) - len(matched)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-6)
        metrics[f'precision@{iou_thresh}'] = prec
        metrics[f'recall@{iou_thresh}'] = rec
        metrics[f'f1@{iou_thresh}'] = f1
    metrics['mean_best_iou'] = float(np.mean(all_ious)) if all_ious else 0.0
    return metrics
