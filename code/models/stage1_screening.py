"""
Stage 1 Model: Dual-Branch 2.5D CNN for DBT Screening
=====================================================

Task: Normal vs Abnormal classification at BREAST level

Architecture:
1. Two parallel branches (CC and MLO)
2. Each branch: ResNet backbone → Slice attention → View embedding
3. Late fusion: Concatenate + MLP → Single prediction

Key innovations for DBT:
- 2.5D approach: Process each slice with 2D CNN, aggregate with attention
- Slice attention: Learn which slices are diagnostically relevant
- Explicit CC+MLO fusion: No averaging hacks, proper learned combination
- View-aware processing: Handle missing views gracefully

Paper justification:
- 2.5D is standard for volumetric medical imaging with limited data
- Slice attention shown effective in CT/MRI literature (refs available)
- Late fusion preserves view-specific information before combination
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple


class SliceAttentionPool(nn.Module):
    """
    Attention-weighted pooling across slices.
    
    Learns which slices are most informative for the classification task.
    Returns weighted sum of slice features.
    
    For DBT: Lesions may only be visible in certain slices.
    Attention helps focus on diagnostically relevant depths.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, S, D) - Batch, Slices, Features
            mask: (B, S) - Valid slice mask (optional)
        
        Returns:
            pooled: (B, D) - Attention-weighted features
            weights: (B, S) - Attention weights (for visualization)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (B, S)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax over slices
        weights = F.softmax(scores, dim=-1)  # (B, S)
        
        # Weighted sum
        pooled = torch.einsum('bs,bsd->bd', weights, x)  # (B, D)
        
        return pooled, weights


class ViewBranch(nn.Module):
    """
    Single view processing branch.
    
    Architecture:
    1. ResNet backbone (removes final FC)
    2. Slice-wise feature extraction
    3. Attention pooling across slices
    
    Input: (B, S, H, W) volume
    Output: (B, D) view-level features
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
        pool_method: str = "attention"
    ):
        super().__init__()
        self.pool_method = pool_method
        
        # Load backbone
        if backbone == "resnet18":
            base = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_dim = 512
        elif backbone == "resnet34":
            base = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_dim = 512
        elif backbone == "resnet50":
            base = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Modify first conv to accept 1 channel (grayscale)
        # Average ImageNet weights across RGB channels
        original_conv = base.conv1
        self.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            self.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        
        # Keep all layers except final FC
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        
        # Project to consistent feature dim if needed
        self.feature_proj = nn.Linear(backbone_dim, feature_dim) if backbone_dim != feature_dim else nn.Identity()
        
        # Slice pooling
        if pool_method == "attention":
            self.slice_pool = SliceAttentionPool(feature_dim)
        elif pool_method == "max":
            self.slice_pool = None  # Use torch.max
        elif pool_method == "avg":
            self.slice_pool = None  # Use torch.mean
        else:
            raise ValueError(f"Unknown pool method: {pool_method}")
        
        self.feature_dim = feature_dim
    
    def extract_slice_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single 2D slice.
        
        Args:
            x: (B, 1, H, W) single slice
        
        Returns:
            features: (B, D)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.feature_proj(x)
        
        return x
    
    def forward(self, volume: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            volume: (B, S, H, W) - DBT volume
            valid_mask: (B,) - Whether this view is valid (for missing view handling)
        
        Returns:
            features: (B, D) - View-level features
            attention: (B, S) - Slice attention weights (if using attention pooling)
        """
        B, S, H, W = volume.shape
        
        # Extract features for each slice
        # Reshape to process all slices in parallel
        x = volume.view(B * S, 1, H, W)  # (B*S, 1, H, W)
        x = self.extract_slice_features(x)  # (B*S, D)
        x = x.view(B, S, -1)  # (B, S, D)
        
        # Pool across slices
        attention = None
        if self.pool_method == "attention":
            features, attention = self.slice_pool(x)
        elif self.pool_method == "max":
            features, _ = x.max(dim=1)
        else:  # avg
            features = x.mean(dim=1)
        
        return features, attention


class CCMLOFusion(nn.Module):
    """
    Fusion module for CC and MLO view features.
    
    Options:
    1. Concatenation + MLP (default)
    2. Cross-attention between views
    3. Bilinear pooling
    
    Handles missing views by using learned default embedding.
    """
    
    def __init__(
        self,
        input_dim: int,
        fusion_dim: int,
        fusion_method: str = "concat_mlp"
    ):
        super().__init__()
        self.fusion_method = fusion_method
        
        if fusion_method == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(input_dim * 2, fusion_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.ReLU(inplace=True)
            )
        elif fusion_method == "cross_attention":
            self.cc_to_mlo = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
            self.mlo_to_cc = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
            self.fusion = nn.Linear(input_dim * 2, fusion_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Learned embeddings for missing views
        self.missing_cc_emb = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.missing_mlo_emb = nn.Parameter(torch.randn(input_dim) * 0.01)
        
        self.output_dim = fusion_dim
    
    def forward(
        self,
        cc_features: torch.Tensor,
        mlo_features: torch.Tensor,
        cc_valid: torch.Tensor,
        mlo_valid: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            cc_features: (B, D) CC view features
            mlo_features: (B, D) MLO view features
            cc_valid: (B,) binary mask for valid CC
            mlo_valid: (B,) binary mask for valid MLO
        
        Returns:
            fused: (B, fusion_dim) fused features
        """
        B = cc_features.shape[0]
        
        # Handle missing views
        cc_mask = cc_valid.unsqueeze(-1)  # (B, 1)
        mlo_mask = mlo_valid.unsqueeze(-1)
        
        cc_features = cc_features * cc_mask + self.missing_cc_emb * (1 - cc_mask)
        mlo_features = mlo_features * mlo_mask + self.missing_mlo_emb * (1 - mlo_mask)
        
        if self.fusion_method == "concat_mlp":
            concat = torch.cat([cc_features, mlo_features], dim=-1)  # (B, 2D)
            fused = self.fusion(concat)
        else:
            # Cross-attention
            cc_attended, _ = self.cc_to_mlo(
                cc_features.unsqueeze(1),
                mlo_features.unsqueeze(1),
                mlo_features.unsqueeze(1)
            )
            mlo_attended, _ = self.mlo_to_cc(
                mlo_features.unsqueeze(1),
                cc_features.unsqueeze(1),
                cc_features.unsqueeze(1)
            )
            concat = torch.cat([cc_attended.squeeze(1), mlo_attended.squeeze(1)], dim=-1)
            fused = self.fusion(concat)
        
        return fused


class DBTScreeningModel(nn.Module):
    """
    Stage 1: DBT Screening Model
    
    Complete architecture for breast-level Normal vs Abnormal classification
    using paired CC and MLO views.
    
    Flow:
    1. CC branch processes CC volume → CC features
    2. MLO branch processes MLO volume → MLO features
    3. Fusion module combines CC + MLO → Fused features
    4. Classifier head → Normal/Abnormal probability
    
    Output: Single prediction per breast (NOT per view)
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 512,
        fusion_dim: int = 256,
        pool_method: str = "attention",
        num_classes: int = 2
    ):
        super().__init__()
        
        # Shared backbone for both views (parameter sharing)
        self.cc_branch = ViewBranch(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=feature_dim,
            pool_method=pool_method
        )
        
        # Use same weights for MLO branch (weight sharing for efficiency)
        # Can also use separate branches if desired
        self.mlo_branch = self.cc_branch  # Weight sharing
        
        # Fusion
        self.fusion = CCMLOFusion(
            input_dim=feature_dim,
            fusion_dim=fusion_dim,
            fusion_method="concat_mlp"
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, num_classes)
        )
        
        self.feature_dim = feature_dim
        self.fusion_dim = fusion_dim
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dict with keys:
                - cc_volume: (B, S, H, W)
                - mlo_volume: (B, S, H, W)
                - cc_valid: (B,) 
                - mlo_valid: (B,)
        
        Returns:
            Dict with keys:
                - logits: (B, 2) class logits
                - probs: (B, 2) class probabilities
                - cc_attention: (B, S) CC slice attention
                - mlo_attention: (B, S) MLO slice attention
        """
        cc_vol = batch['cc_volume']
        mlo_vol = batch['mlo_volume']
        cc_valid = batch['cc_valid']
        mlo_valid = batch['mlo_valid']
        
        # Extract view features
        cc_features, cc_attention = self.cc_branch(cc_vol, cc_valid)
        mlo_features, mlo_attention = self.mlo_branch(mlo_vol, mlo_valid)
        
        # Fuse views
        fused = self.fusion(cc_features, mlo_features, cc_valid, mlo_valid)
        
        # Classify
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'cc_attention': cc_attention,
            'mlo_attention': mlo_attention,
            'cc_features': cc_features,
            'mlo_features': mlo_features,
            'fused_features': fused
        }
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simple prediction interface.
        
        Returns:
            predictions: (B,) predicted class (0=Normal, 1=Abnormal)
            confidences: (B,) confidence scores
        """
        output = self.forward(batch)
        probs = output['probs']
        predictions = probs.argmax(dim=-1)
        confidences = probs.max(dim=-1).values
        return predictions, confidences


class ScreeningLoss(nn.Module):
    """
    Loss function for Stage 1 screening with FOCAL LOSS.
    
    Focal Loss specifically designed for severe class imbalance.
    Downweights easy examples (normals), focuses on hard examples (abnormals).
    """
    
    def __init__(self, pos_weight: float = 15.0, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma  # Focusing parameter
        self.alpha = alpha  # Balance parameter (weight for positive class)
        
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal Loss: -alpha * (1-p)^gamma * log(p) for positive class
                    -(1-alpha) * p^gamma * log(1-p) for negative class
        """
        # Get probability for positive class (abnormal)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # Prob of abnormal
        
        # Binary targets
        targets_float = targets.float()
        
        # Focal weights
        p_t = torch.where(targets_float == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting (higher for minority class)
        alpha_t = torch.where(targets_float == 1, self.alpha, 1 - self.alpha)
        
        # Cross entropy
        ce = F.cross_entropy(logits, targets, reduction='none')
        
        # Combined focal loss
        loss = alpha_t * focal_weight * ce
        
        return loss.mean()
    
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            output: Model output dict
            labels: (B,) ground truth labels
        
        Returns:
            Dict with loss components
        """
        logits = output['logits']
        
        # Focal loss for classification
        focal = self.focal_loss(logits, labels)
        
        return {
            'total': focal,
            'ce': focal,
            'attn_reg': torch.tensor(0.0, device=logits.device)
        }


def create_stage1_model(config) -> Tuple[DBTScreeningModel, ScreeningLoss]:
    """Factory function to create Stage 1 model and loss."""
    model = DBTScreeningModel(
        backbone=config.stage1.backbone,
        pretrained=config.stage1.pretrained,
        feature_dim=config.stage1.feature_dim,
        fusion_dim=config.stage1.fusion_dim,
        pool_method=config.stage1.slice_pool_method,
        num_classes=2
    )
    
    # Focal loss with aggressive weighting for severe imbalance
    loss_fn = ScreeningLoss(
        pos_weight=config.stage1.pos_weight,
        gamma=2.0,   # Focusing parameter
        alpha=0.8    # Weight for abnormal class (minority)
    )
    
    return model, loss_fn


# Quick architecture summary
if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    model, loss_fn = create_stage1_model(config)
    
    print("\n" + "="*60)
    print("STAGE 1 MODEL ARCHITECTURE")
    print("="*60)
    print(model)
    
    # Test forward pass
    batch = {
        'cc_volume': torch.randn(2, 15, 512, 512),
        'mlo_volume': torch.randn(2, 15, 512, 512),
        'cc_valid': torch.ones(2),
        'mlo_valid': torch.ones(2),
        'label': torch.tensor([0, 1])
    }
    
    with torch.no_grad():
        output = model(batch)
        print(f"\nOutput shapes:")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
