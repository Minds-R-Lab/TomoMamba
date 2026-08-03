# mobilenet_backbone.py — Drop-in replacement for SliceBackbone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MobileNetBackbone(nn.Module):
    """MobileNet-v2 backbone with FPN-lite, same interface as SliceBackbone."""
    def __init__(self, pretrained=True):
        super().__init__()
        mob = models.mobilenet_v2(weights='IMAGENET1K_V1' if pretrained else None)
        feats = mob.features
        
        # Adapt first conv: 3ch → 1ch
        old_conv = feats[0][0]
        new_conv = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        if pretrained:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        feats[0][0] = new_conv
        
        # Split into 4 stages matching ResNet FPN hookpoints
        # c2: stride 4, 24ch  (features 0-3)
        # c3: stride 8, 32ch  (features 4-6)
        # c4: stride 16, 96ch (features 7-13)
        # c5: stride 32, 320ch (features 14-17)
        self.stage1 = nn.Sequential(*feats[0:4])    # → 24ch, stride 4
        self.stage2 = nn.Sequential(*feats[4:7])     # → 32ch, stride 8
        self.stage3 = nn.Sequential(*feats[7:14])    # → 96ch, stride 16
        self.stage4 = nn.Sequential(*feats[14:18])   # → 320ch, stride 32
        
        # FPN lateral convs (match channel dims)
        self.up5 = nn.Sequential(nn.Conv2d(320, 128, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.up4 = nn.Sequential(nn.Conv2d(96, 64, 1),   nn.BatchNorm2d(64),  nn.ReLU(True))
        self.up3 = nn.Sequential(nn.Conv2d(32, 32, 1),   nn.BatchNorm2d(32),  nn.ReLU(True))
        
        # Fusion: 24 + 32 + 64 + 128 = 248 → 128
        self.fusion = nn.Sequential(
            nn.Conv2d(248, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.out_channels = 128
    
    def forward(self, x):
        c2 = self.stage1(x)    # (B, 24, H/4, W/4)
        c3 = self.stage2(c2)   # (B, 32, H/8, W/8)
        c4 = self.stage3(c3)   # (B, 96, H/16, W/16)
        c5 = self.stage4(c4)   # (B, 320, H/32, W/32)
        
        t = c2.shape[2:]  # target size = stride-4
        c5_up = F.interpolate(self.up5(c5), size=t, mode='bilinear', align_corners=False)
        c4_up = F.interpolate(self.up4(c4), size=t, mode='bilinear', align_corners=False)
        c3_up = F.interpolate(self.up3(c3), size=t, mode='bilinear', align_corners=False)
        
        return self.fusion(torch.cat([c2, c3_up, c4_up, c5_up], dim=1))