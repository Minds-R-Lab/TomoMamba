"""
Configuration for Two-Stage DBT Classification Pipeline
========================================================
Stage 1: Normal vs Abnormal (breast-level, CC+MLO fusion)
Stage 2: Benign vs Cancer + Localization (abnormal cases only)

Designed to match the preprocessing output:
- Volume shape: (15, 1024, 1024)
- Z-score normalized
- Bounding box annotations in metadata JSON
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import torch


@dataclass
class DataConfig:
    """Data configuration matching preprocessing script output."""
    
    # Root directory containing train/validation/test splits
    data_root: str = "/path/to/DBT_CancerBenignNormal_Gradient_1024_15"  # EDIT THIS
    
    # Volume dimensions from preprocessing
    num_slices: int = 15
    height: int = 1024
    width: int = 1024
    
    # For training efficiency, we downsample spatially
    # Original 1024x1024 -> 512x512 for Stage 1 (memory constraints)
    # Stage 2 uses 256x256 crops around lesion regions
    stage1_spatial_size: Tuple[int, int] = (512, 512)
    stage2_spatial_size: Tuple[int, int] = (256, 256)
    
    # Class mappings
    # Stage 1: Normal=0, Abnormal=1 (Abnormal = Cancer OR Benign)
    # Stage 2: Benign=0, Cancer=1
    stage1_classes: List[str] = field(default_factory=lambda: ["Normal", "Abnormal"])
    stage2_classes: List[str] = field(default_factory=lambda: ["Benign", "Cancer"])
    
    # Number of workers for data loading
    num_workers: int = 4
    
    # Views to fuse
    views: List[str] = field(default_factory=lambda: ["CC", "MLO"])


@dataclass
class Stage1Config:
    """
    Stage 1: Screening - Normal vs Abnormal
    
    Architecture: Dual-branch 2.5D CNN with explicit CC+MLO fusion
    - Each branch processes one view (CC or MLO)
    - Slice-level features via ResNet18 backbone
    - Slice attention pooling
    - Late fusion of CC and MLO representations
    - Single breast-level prediction
    """
    
    # Architecture
    backbone: str = "resnet18"  # Lightweight, sufficient for screening
    pretrained: bool = True
    
    # Slice handling
    slice_attention: bool = True  # Learn which slices are most informative
    slice_pool_method: str = "attention"  # "attention", "max", "avg"
    
    # Feature dimensions
    feature_dim: int = 512  # Output dim after backbone
    fusion_dim: int = 256   # Dimension for CC+MLO fusion
    
    # Training
    batch_size: int = 4  # Per-breast (each sample = CC + MLO volumes)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    
    # Class imbalance handling
    # Dataset: Normal=1357, Abnormal=105 → ratio ~13:1
    # Need aggressive weighting to prevent collapse
    use_weighted_loss: bool = True
    pos_weight: float = 15.0  # Upweight abnormal class (higher than ratio)
    
    # Augmentation
    use_augmentation: bool = True
    
    # Checkpointing
    save_dir: str = "./checkpoints/stage1"


@dataclass 
class Stage2Config:
    """
    Stage 2: Diagnostic + Localization (Abnormal cases only)
    
    IMPROVED VERSION - combat overfitting on small dataset (200 cases):
    - Smaller backbone (ResNet18)
    - Heavy regularization (dropout, weight decay, label smoothing)
    - Weighted sampling for class balance
    - Focal loss for hard example mining
    """
    
    # Architecture - SMALLER to prevent overfitting
    backbone: str = "resnet18"  # Changed from resnet34 - less capacity
    pretrained: bool = True
    
    # FPN configuration - reduced
    fpn_channels: int = 128  # Reduced from 256
    num_anchor_scales: int = 3
    num_anchor_ratios: int = 3
    
    # Detection
    score_threshold: float = 0.3
    nms_threshold: float = 0.5
    max_detections_per_slice: int = 10
    
    # Classification
    classification_method: str = "detection_weighted"
    
    # Training - HEAVY REGULARIZATION
    batch_size: int = 4  # Smaller for more stochasticity
    learning_rate: float = 3e-4  # Higher LR, will use scheduler
    weight_decay: float = 5e-3  # Strong L2 regularization
    dropout: float = 0.6  # Heavy dropout
    label_smoothing: float = 0.15  # Prevent overconfident predictions
    epochs: int = 150  # More epochs, rely on early stopping
    
    # Class balance (124 Benign vs 76 Cancer)
    use_weighted_sampling: bool = True
    focal_gamma: float = 2.0  # Focal loss focusing parameter
    focal_alpha: float = 0.6  # Weight for Cancer (minority)
    
    # Multi-task loss weights
    cls_loss_weight: float = 1.0
    box_loss_weight: float = 1.0
    obj_loss_weight: float = 0.5
    
    # Checkpointing
    save_dir: str = "./checkpoints/stage2"


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Mixed precision training
    use_amp: bool = True
    
    # Early stopping
    patience: int = 10
    
    # Logging
    log_dir: str = "./logs"
    log_interval: int = 10  # Log every N batches
    
    # Evaluation
    eval_interval: int = 1  # Evaluate every N epochs
    
    # Stage configs
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)


def get_config() -> TrainingConfig:
    """Get default configuration."""
    return TrainingConfig()


def print_config(config: TrainingConfig):
    """Pretty print configuration."""
    print("\n" + "="*60)
    print("TWO-STAGE DBT PIPELINE CONFIGURATION")
    print("="*60)
    
    print("\n📁 DATA CONFIG:")
    print(f"  Root: {config.data.data_root}")
    print(f"  Volume shape: ({config.data.num_slices}, {config.data.height}, {config.data.width})")
    print(f"  Stage 1 spatial: {config.data.stage1_spatial_size}")
    print(f"  Stage 2 spatial: {config.data.stage2_spatial_size}")
    
    print("\n🔍 STAGE 1 - SCREENING (Normal vs Abnormal):")
    print(f"  Backbone: {config.stage1.backbone}")
    print(f"  Slice pooling: {config.stage1.slice_pool_method}")
    print(f"  CC+MLO fusion: Late fusion at dim={config.stage1.fusion_dim}")
    print(f"  Batch size: {config.stage1.batch_size}")
    print(f"  Learning rate: {config.stage1.learning_rate}")
    print(f"  Epochs: {config.stage1.epochs}")
    
    print("\n🎯 STAGE 2 - DIAGNOSTIC + LOCALIZATION:")
    print(f"  Backbone: {config.stage2.backbone}")
    print(f"  Detection: FPN with {config.stage2.fpn_channels} channels")
    print(f"  Batch size: {config.stage2.batch_size}")
    print(f"  Learning rate: {config.stage2.learning_rate}")
    print(f"  Epochs: {config.stage2.epochs}")
    
    print("\n" + "="*60)
