"""
Controlled Baselines for MambaCenterNet Ablation
=================================================
Three architectures, identical data/training/evaluation.
Only the model architecture changes.

Baseline 1: ResNet-18 Classifier (classification only, no detection)
Baseline 2: ResNet-18 + CenterNet (detection + cls, no cross-slice)
Baseline 3: ResNet-18 + BiGRU + CenterNet (BiGRU replaces Mamba)

All share the same SliceBackbone, same ROI classifier, same loss weights.
"""

import math
from typing import Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# =============================================================================
# SHARED: SliceBackbone (identical to MambaCenterNet)
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
# SHARED: CenterNet Head (for Baselines 2 & 3)
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
# SHARED: ROI Classifier (for Baselines 2 & 3)
# =============================================================================

class ROIClassifier(nn.Module):
    def __init__(self, in_channels=128, roi_size=7, num_classes=2, dropout=0.7):
        super().__init__()
        self.roi_size = roi_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
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
# SHARED: BiGRU Cross-Slice (for Baseline 3)
# =============================================================================

class BiGRUCrossSlice(nn.Module):
    """Bidirectional GRU for cross-slice propagation. Replaces Mamba."""
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, features):
        B, S, C, H, W = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, S, C)
        out, _ = self.gru(x)
        out = self.norm(out)
        gate = self.gate(torch.cat([x, out], dim=-1))
        enhanced = x + gate * out
        return enhanced.reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2)


# =============================================================================
# BASELINE 1: ResNet-18 Classifier Only
# =============================================================================

class ResNet18Classifier(nn.Module):
    """
    Classification only. No detection heads, no cross-slice propagation.
    ResNet-18 → FPN → GAP across slices → linear classifier.

    Tests: Does joint detection learning help classification?
    """
    def __init__(self, num_classes=2, dropout=0.7, spatial_size=384):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_size = spatial_size
        self.stride = 4
        self.baseline_name = "ResNet-18 Classifier"

        self.backbone = SliceBackbone(pretrained=True)
        feat_dim = self.backbone.out_channels  # 128

        # Volume-level classifier: pool across spatial + slices
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B*S, 128, 1, 1)
            nn.Flatten(),              # (B*S, 128)
        )
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch):
        volume = batch['volume']
        B, S, H, W = volume.shape

        slices = volume.reshape(B * S, 1, H, W)
        slice_feats = self.backbone(slices)  # (B*S, 128, fH, fW)
        pooled = self.classifier(slice_feats)  # (B*S, 128)
        pooled = pooled.reshape(B, S, -1)      # (B, S, 128)

        # Max-pool across slices (like radiologist looking at worst slice)
        vol_feats = pooled.max(dim=1)[0]       # (B, 128)
        logits = self.fc(vol_feats)            # (B, num_classes)
        probs = F.softmax(logits, dim=-1)

        return {
            'vol_logits': logits,
            'vol_probs': probs,
            # No detection outputs
            'heatmaps': None,
            'sizes': None,
            'offsets': None,
            'feat_maps': None,
        }


# =============================================================================
# BASELINE 2: ResNet-18 + CenterNet (no cross-slice)
# =============================================================================

class ResNet18CenterNet(nn.Module):
    """
    Detection + classification, but NO cross-slice propagation.
    Each slice processed independently. Same CenterNet + ROI classifier.

    Tests: Does cross-slice propagation improve performance?
    """
    def __init__(self, num_classes=2, dropout=0.7, spatial_size=384, roi_size=7):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_size = spatial_size
        self.stride = 4
        self.roi_size = roi_size
        self.baseline_name = "ResNet-18 + CenterNet"

        self.backbone = SliceBackbone(pretrained=True)
        feat_dim = self.backbone.out_channels
        # NO cross-slice module
        self.cross_slice = None
        self.detect_head = CenterNetHead(in_channels=feat_dim)
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

        # No cross-slice propagation — use raw backbone features
        det_input = slice_feats.reshape(B * S, C, fH, fW)
        det_out = self.detect_head(det_input)

        return {
            'heatmaps': det_out['heatmap'].reshape(B, S, 1, fH, fW),
            'sizes':    det_out['size'].reshape(B, S, 2, fH, fW),
            'offsets':  det_out['offset'].reshape(B, S, 2, fH, fW),
            'feat_maps': slice_feats,
        }

    # Copy ROI extraction / classification / detection methods from main model
    def _extract_single_roi(self, feat_map, bx, by, bw, bh, fH, fW, context_pad=0.3):
        pad_w = bw * context_pad
        pad_h = bh * context_pad
        x1 = max(0, int((bx - pad_w) * fW))
        y1 = max(0, int((by - pad_h) * fH))
        x2 = min(fW, int(math.ceil((bx + bw + pad_w) * fW)))
        y2 = min(fH, int(math.ceil((by + bh + pad_h) * fH)))
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
        B, S, C, fH, fW = feat_maps.shape
        device = feat_maps.device
        all_rois, all_labels, roi_sample_idx = [], [], []

        for b in range(B):
            for i in range(boxes.shape[1]):
                if box_masks[b, i] < 0.5:
                    continue
                s_idx = int(boxes[b, i, 0].item())
                if s_idx < 0 or s_idx >= S:
                    continue
                bx, by, bw, bh = [boxes[b, i, j].item() for j in range(1, 5)]
                roi = self._extract_single_roi(feat_maps[b, s_idx], bx, by, bw, bh, fH, fW)
                all_rois.append(roi)
                all_labels.append(labels[b].item())
                roi_sample_idx.append(b)

        if all_rois:
            rois_tensor = torch.stack(all_rois)
            roi_logits = self.roi_classifier(rois_tensor)
            roi_labels = torch.tensor(all_labels, device=device, dtype=torch.long)
            roi_probs = F.softmax(roi_logits, dim=-1)

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
            'roi_logits': roi_logits, 'roi_labels': roi_labels,
            'vol_logits': vol_logits, 'vol_probs': vol_probs,
        }

    @torch.no_grad()
    def detect(self, volume, score_thresh=0.20, nms_kernel=3, topk=50):
        """NMS-based detection on heatmaps."""
        B, S, H, W = volume.shape
        slices = volume.reshape(B * S, 1, H, W)
        feats = self.backbone(slices)
        _, C, fH, fW = feats.shape
        feats5d = feats.reshape(B, S, C, fH, fW)
        det_in = feats5d.reshape(B * S, C, fH, fW)
        det_out = self.detect_head(det_in)

        hm = det_out['heatmap'].reshape(B, S, 1, fH, fW)
        sz = det_out['size'].reshape(B, S, 2, fH, fW)
        off = det_out['offset'].reshape(B, S, 2, fH, fW)

        results = []
        for b in range(B):
            boxes, scores, slice_indices = [], [], []
            for s in range(S):
                h = hm[b, s, 0]
                pad = nms_kernel // 2
                hmax = F.max_pool2d(h.unsqueeze(0).unsqueeze(0), nms_kernel,
                                     stride=1, padding=pad).squeeze()
                keep = (h == hmax) & (h >= score_thresh)
                ys, xs = torch.where(keep)
                if len(xs) == 0:
                    continue
                sc = h[ys, xs]
                if len(sc) > topk:
                    topk_idx = sc.argsort(descending=True)[:topk]
                    xs, ys, sc = xs[topk_idx], ys[topk_idx], sc[topk_idx]
                w_pred = sz[b, s, 0, ys, xs] * W
                h_pred = sz[b, s, 1, ys, xs] * H
                ox = off[b, s, 0, ys, xs]
                oy = off[b, s, 1, ys, xs]
                cx = (xs.float() + ox) * self.stride
                cy = (ys.float() + oy) * self.stride
                x1 = cx - w_pred / 2
                y1 = cy - h_pred / 2
                x2 = cx + w_pred / 2
                y2 = cy + h_pred / 2
                for i in range(len(xs)):
                    boxes.append([x1[i].item(), y1[i].item(),
                                  x2[i].item(), y2[i].item()])
                    scores.append(sc[i].item())
                    slice_indices.append(s)
            results.append({
                'boxes': torch.tensor(boxes, device=volume.device) if boxes
                         else torch.zeros(0, 4, device=volume.device),
                'scores': torch.tensor(scores, device=volume.device) if scores
                          else torch.zeros(0, device=volume.device),
                'slice_indices': torch.tensor(slice_indices, device=volume.device) if slice_indices
                                 else torch.zeros(0, dtype=torch.long, device=volume.device),
            })
        return results

    @torch.no_grad()
    def classify_detections(self, feat_maps, det_results):
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

            roi_list = []
            for j in range(len(det_boxes)):
                box = det_boxes[j]
                s_idx = int(det_slices[j].item())
                if s_idx < 0 or s_idx >= S:
                    continue
                bx = box[0].item() / W
                by = box[1].item() / H
                bw = (box[2].item() - box[0].item()) / W
                bh = (box[3].item() - box[1].item()) / H
                bx, by = max(0, bx), max(0, by)
                bw, bh = max(0.001, bw), max(0.001, bh)
                roi = self._extract_single_roi(feat_maps[b, s_idx], bx, by, bw, bh, fH, fW)
                roi_list.append(roi)

            if roi_list:
                rois = torch.stack(roi_list)
                logits = self.roi_classifier(rois)
                probs = F.softmax(logits, dim=-1)
                best = probs[:, 1].argmax().item()
                vol_probs_list.append(probs[best])
            else:
                fb = self.global_classifier(feat_maps[b:b+1])
                vol_probs_list.append(F.softmax(fb.squeeze(0), dim=-1))

        return torch.stack(vol_probs_list)


# =============================================================================
# BASELINE 3: ResNet-18 + BiGRU + CenterNet
# =============================================================================

class ResNet18BiGRUCenterNet(ResNet18CenterNet):
    """
    Same as Baseline 2 but with BiGRU cross-slice propagation instead of Mamba.

    Tests: Does Mamba outperform a standard recurrent alternative?
    """
    def __init__(self, num_classes=2, dropout=0.7, spatial_size=384, roi_size=7):
        super().__init__(num_classes, dropout, spatial_size, roi_size)
        self.baseline_name = "ResNet-18 + BiGRU + CenterNet"
        feat_dim = self.backbone.out_channels
        self.cross_slice = BiGRUCrossSlice(d_model=feat_dim)

    def forward(self, batch):
        volume = batch['volume']
        B, S, H, W = volume.shape

        slices = volume.reshape(B * S, 1, H, W)
        slice_feats = self.backbone(slices)
        _, C, fH, fW = slice_feats.shape
        slice_feats = slice_feats.reshape(B, S, C, fH, fW)

        # BiGRU cross-slice propagation
        enhanced = self.cross_slice(slice_feats)

        det_input = enhanced.reshape(B * S, C, fH, fW)
        det_out = self.detect_head(det_input)

        return {
            'heatmaps': det_out['heatmap'].reshape(B, S, 1, fH, fW),
            'sizes':    det_out['size'].reshape(B, S, 2, fH, fW),
            'offsets':  det_out['offset'].reshape(B, S, 2, fH, fW),
            'feat_maps': enhanced,
        }

    @torch.no_grad()
    def detect(self, volume, score_thresh=0.20, nms_kernel=3, topk=50):
        """Detection with BiGRU cross-slice features."""
        B, S, H, W = volume.shape
        slices = volume.reshape(B * S, 1, H, W)
        feats = self.backbone(slices)
        _, C, fH, fW = feats.shape
        feats5d = feats.reshape(B, S, C, fH, fW)
        enhanced = self.cross_slice(feats5d)

        det_in = enhanced.reshape(B * S, C, fH, fW)
        det_out = self.detect_head(det_in)

        hm = det_out['heatmap'].reshape(B, S, 1, fH, fW)
        sz = det_out['size'].reshape(B, S, 2, fH, fW)
        off = det_out['offset'].reshape(B, S, 2, fH, fW)

        results = []
        for b in range(B):
            boxes, scores, slice_indices = [], [], []
            for s in range(S):
                h = hm[b, s, 0]
                pad = nms_kernel // 2
                hmax = F.max_pool2d(h.unsqueeze(0).unsqueeze(0), nms_kernel,
                                     stride=1, padding=pad).squeeze()
                keep = (h == hmax) & (h >= score_thresh)
                ys, xs = torch.where(keep)
                if len(xs) == 0:
                    continue
                sc = h[ys, xs]
                if len(sc) > topk:
                    topk_idx = sc.argsort(descending=True)[:topk]
                    xs, ys, sc = xs[topk_idx], ys[topk_idx], sc[topk_idx]
                w_pred = sz[b, s, 0, ys, xs] * W
                h_pred = sz[b, s, 1, ys, xs] * H
                ox = off[b, s, 0, ys, xs]
                oy = off[b, s, 1, ys, xs]
                cx = (xs.float() + ox) * self.stride
                cy = (ys.float() + oy) * self.stride
                x1 = cx - w_pred / 2
                y1 = cy - h_pred / 2
                x2 = cx + w_pred / 2
                y2 = cy + h_pred / 2
                for i in range(len(xs)):
                    boxes.append([x1[i].item(), y1[i].item(),
                                  x2[i].item(), y2[i].item()])
                    scores.append(sc[i].item())
                    slice_indices.append(s)
            results.append({
                'boxes': torch.tensor(boxes, device=volume.device) if boxes
                         else torch.zeros(0, 4, device=volume.device),
                'scores': torch.tensor(scores, device=volume.device) if scores
                          else torch.zeros(0, device=volume.device),
                'slice_indices': torch.tensor(slice_indices, device=volume.device) if slice_indices
                                 else torch.zeros(0, dtype=torch.long, device=volume.device),
            })
        return results


# =============================================================================
# FACTORY: Create baseline by number
# =============================================================================

# =============================================================================
# BASELINE 4: Transformer cross-slice propagation
# =============================================================================

class TransformerCrossSlice(nn.Module):
    """Self-attention over the depth axis. Drop-in replacement for
    BiGRUCrossSlice and for Mamba: identical input/output shape, identical
    LayerNorm and gated residual, so the only difference is the sequence model.

    A learned positional embedding is included because self-attention is
    permutation-invariant. Without it the transformer would have no access to
    depth order at all, which would not be a fair comparison against an
    ordered recurrence.
    """
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256,
                 num_layers=1, max_slices=64, attn_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pos = nn.Parameter(torch.zeros(1, max_slices, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=attn_dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, features):
        B, S, C, H, W = features.shape
        x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, S, C)
        out = self.encoder(x + self.pos[:, :S, :])
        out = self.norm(out)
        gate = self.gate(torch.cat([x, out], dim=-1))
        enhanced = x + gate * out
        return enhanced.reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2)


class ResNet18TransformerCenterNet(ResNet18BiGRUCenterNet):
    """Baseline 2 with transformer cross-slice propagation instead of Mamba.

    Tests: does self-attention over 15 slices outperform selective
    state-space propagation, and at what computational cost?

    forward() and detect() are inherited from ResNet18BiGRUCenterNet without
    modification; only self.cross_slice differs.
    """
    def __init__(self, num_classes=2, dropout=0.7, spatial_size=384,
                 roi_size=7, nhead=4, dim_feedforward=256, num_layers=1):
        super().__init__(num_classes, dropout, spatial_size, roi_size)
        self.baseline_name = "ResNet-18 + Transformer + CenterNet"
        feat_dim = self.backbone.out_channels
        self.cross_slice = TransformerCrossSlice(
            d_model=feat_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            num_layers=num_layers)


def create_baseline(baseline_id, num_classes=2, dropout=0.7, spatial_size=384):
    """
    Create a baseline model.
    baseline_id: 1, 2, or 3
    """
    if baseline_id == 1:
        model = ResNet18Classifier(num_classes=num_classes, dropout=dropout,
                                    spatial_size=spatial_size)
    elif baseline_id == 2:
        model = ResNet18CenterNet(num_classes=num_classes, dropout=dropout,
                                   spatial_size=spatial_size)
    elif baseline_id == 3:
        model = ResNet18BiGRUCenterNet(num_classes=num_classes, dropout=dropout,
                                        spatial_size=spatial_size)


    elif baseline_id == 4:
        model = ResNet18TransformerCenterNet(num_classes=num_classes,
                                             dropout=dropout,
                                             spatial_size=spatial_size)
    else:
        raise ValueError(f"Unknown baseline_id: {baseline_id}")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Baseline {baseline_id}: {model.baseline_name}")
    print(f"  Parameters: {total_p:,} (trainable: {train_p:,})")
    return model
