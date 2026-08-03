# dbt_cancer_benign_normal_gradient.py
# GRADIENT-BASED slice selection for Cancer, Benign, AND Normal classes
# - Cancer/Benign: Annotated slices FIRST, then gradient-based fill
# - Normal: Pure gradient-based selection (no annotations)

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torch.nn.functional as F
from skimage.exposure import rescale_intensity
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json

warnings.filterwarnings('ignore')

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

# =================== Utilities ===================

def _finite_min_max(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0, 0.0
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 0.0
    return float(x.min()), float(x.max())

def _nan_to_num32(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def _safe_percentile_u8(img: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.uint8:
    x = _nan_to_num32(img)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)
    lo = np.percentile(x[finite], p_low)
    hi = np.percentile(x[finite], p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        vmin, vmax = _finite_min_max(x)
        if vmax <= vmin:
            return np.zeros_like(x, dtype=np.uint8)
        lo, hi = vmin, vmax
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)

# ============= DICOM Processing =============

def _get_image_laterality(pixel_array: np.ndarray) -> str:
    """Official Duke method"""
    left_edge = np.sum(pixel_array[:, 0])
    right_edge = np.sum(pixel_array[:, -1])
    return "R" if left_edge < right_edge else "L"

def _get_window_center(ds: dicom.dataset.FileDataset) -> np.float32:
    """Official Duke method"""
    try:
        return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1050].value)
    except:
        wc = getattr(ds, "WindowCenter", 1024)
        if isinstance(wc, (list, tuple)) and len(wc) > 0:
            wc = wc[0]
        return np.float32(wc)

def _get_window_width(ds: dicom.dataset.FileDataset) -> np.float32:
    """Official Duke method"""
    try:
        return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1051].value)
    except:
        ww = getattr(ds, "WindowWidth", 4096)
        if isinstance(ww, (list, tuple)) and len(ww) > 0:
            ww = ww[0]
        return np.float32(ww)

def dcmread_full_volume(fp: Union[str, Path], view: str) -> Tuple[np.ndarray, Tuple[int, int], bool]:
    """
    Read FULL DBT volume using OFFICIAL Duke method.
    Returns: (volume, original_shape, was_flipped)
    """
    ds = dicom.dcmread(fp)
    try:
        ds.decompress(handler_name="pylibjpeg")
    except Exception:
        try:
            ds.decompress()
        except Exception:
            pass

    pixel_array = ds.pixel_array
    pixel_array = _nan_to_num32(pixel_array)
    
    if pixel_array.ndim == 2:
        pixel_array = pixel_array[None, ...]
    
    original_shape = (pixel_array.shape[1], pixel_array.shape[2])

    # Official Duke laterality fix
    view_laterality = view[0].upper()
    image_laterality = _get_image_laterality(pixel_array[0])
    was_flipped = (image_laterality != view_laterality)
    
    if was_flipped:
        pixel_array = np.flip(pixel_array, axis=(-1, -2)).copy()

    # Official VOI windowing
    window_center = _get_window_center(ds)
    window_width = _get_window_width(ds)
    low = (2 * window_center - window_width) / 2
    high = (2 * window_center + window_width) / 2

    out = np.empty_like(pixel_array, dtype=np.float32)
    for i in range(pixel_array.shape[0]):
        out[i] = rescale_intensity(
            pixel_array[i], 
            in_range=(low, high), 
            out_range="dtype"
        ).astype(np.float32, copy=False)

    return _nan_to_num32(out), original_shape, was_flipped

# ============= Box coordinate transformation =============

def transform_boxes_after_flip(boxes: List[Dict], original_shape: Tuple[int, int], was_flipped: bool) -> List[Dict]:
    """Transform box coordinates after np.flip(axis=(-1, -2))"""
    if not was_flipped:
        return boxes
    
    H, W = original_shape
    transformed = []
    
    for box in boxes:
        new_box = box.copy()
        new_box['x'] = W - (box['x'] + box['width'])
        new_box['y'] = H - (box['y'] + box['height'])
        transformed.append(new_box)
    
    return transformed

def expand_boxes(boxes: List[Dict], expansion_factor: float, image_shape: Tuple[int, int]) -> List[Dict]:
    """Expand bounding boxes by a factor"""
    if expansion_factor <= 1.0:
        return boxes
    
    H, W = image_shape
    expanded = []
    
    for box in boxes:
        cx = box['x'] + box['width'] / 2
        cy = box['y'] + box['height'] / 2
        
        new_width = box['width'] * expansion_factor
        new_height = box['height'] * expansion_factor
        
        new_x = cx - new_width / 2
        new_y = cy - new_height / 2
        
        new_x = max(0, new_x)
        new_y = max(0, new_y)
        new_width = min(new_width, W - new_x)
        new_height = min(new_height, H - new_y)
        
        expanded_box = box.copy()
        expanded_box['x'] = int(new_x)
        expanded_box['y'] = int(new_y)
        expanded_box['width'] = int(new_width)
        expanded_box['height'] = int(new_height)
        expanded_box['expanded'] = True
        expanded_box['original_x'] = box['x']
        expanded_box['original_y'] = box['y']
        expanded_box['original_width'] = box['width']
        expanded_box['original_height'] = box['height']
        
        expanded.append(expanded_box)
    
    return expanded

# ============= GRADIENT-BASED slice selection =============

def _calculate_gradient_magnitude(slice_f32: np.ndarray) -> float:
    """
    Calculate gradient magnitude using Sobel operators.
    Higher values = more edges/contrast/detail in the slice.
    """
    # Normalize to 0-1 for stable gradient calculation
    slice_norm = _safe_percentile_u8(slice_f32, 0.5, 99.5).astype(np.float32) / 255.0
    
    # Sobel gradients in X and Y
    grad_x = cv2.Sobel(slice_norm, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(slice_norm, cv2.CV_32F, 0, 1, ksize=3)
    
    # Gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Return mean gradient magnitude (higher = more edges/detail)
    return float(grad_mag.mean())

def _calculate_laplacian_variance(slice_f32: np.ndarray) -> float:
    """
    Calculate Laplacian variance - detects rapid intensity changes.
    Higher values = more sharp transitions/edges.
    """
    slice_norm = _safe_percentile_u8(slice_f32, 0.5, 99.5).astype(np.float32)
    laplacian = cv2.Laplacian(slice_norm, cv2.CV_32F, ksize=3)
    return float(laplacian.var())

def _gradient_score(slice_f32: np.ndarray) -> float:
    """
    Combined score: gradient magnitude + laplacian variance.
    Prioritizes slices with strong edges and sharp transitions.
    """
    grad_mag = _calculate_gradient_magnitude(slice_f32)
    lap_var = _calculate_laplacian_variance(slice_f32)
    
    # Normalize and combine (adjust weights as needed)
    # Higher gradient magnitude + higher laplacian = more diagnostic content
    return 0.6 * grad_mag + 0.4 * (lap_var / 1000.0)  # Scale laplacian to similar range

def select_annotated_plus_gradient_based(
    volume: np.ndarray, 
    annotated_slices: List[int],
    keep: int = 15, 
    min_sep: int = 1
) -> List[int]:
    """
    GRADIENT-BASED STRATEGY for Cancer/Benign:
    1. First, include ALL annotated slices (the ones with bounding boxes)
    2. Then, select remaining slices based on STRONGEST GRADIENTS
    3. No depth constraints - purely based on edge/contrast strength
    """
    S = volume.shape[0]
    keep = max(1, min(keep, S))
    
    # Start with annotated slices
    chosen = sorted(list(set(annotated_slices)))
    
    print(f"      Starting with {len(chosen)} annotated slices: {chosen}")
    
    # If we already have enough slices, just return them
    if len(chosen) >= keep:
        return sorted(chosen[:keep])
    
    # Calculate gradient scores for ALL slices
    print(f"      Calculating gradient scores for {S} slices...")
    gradient_scores = []
    for i in range(S):
        if i not in chosen:  # Don't re-score already chosen slices
            score = _gradient_score(volume[i])
            gradient_scores.append((i, score))
    
    # Sort by gradient score (descending - highest first)
    gradient_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"      Top 5 gradient slices: {[(idx, f'{score:.4f}') for idx, score in gradient_scores[:5]]}")
    
    # Select top gradient slices
    for idx, score in gradient_scores:
        if len(chosen) >= keep:
            break
        
        # Check minimal separation constraint
        too_close = False
        if min_sep > 0:
            for existing_idx in chosen:
                if abs(idx - existing_idx) < min_sep:
                    too_close = True
                    break
        
        if not too_close:
            chosen.append(idx)
    
    # If we still need more slices (due to min_sep constraints), relax and add anyway
    if len(chosen) < keep:
        print(f"      Relaxing min_sep constraint to reach {keep} slices...")
        for idx, score in gradient_scores:
            if idx not in chosen:
                chosen.append(idx)
                if len(chosen) >= keep:
                    break
    
    final = sorted(chosen)
    print(f"      Final selection: {len(final)} slices (annotated + gradient-based)")
    print(f"      Selected indices: {final}")
    
    return final

def select_pure_gradient_based(
    volume: np.ndarray, 
    keep: int = 15, 
    min_sep: int = 1
) -> List[int]:
    """
    PURE GRADIENT-BASED SELECTION for Normals (no annotated slices).
    Simply picks the slices with strongest gradient scores.
    """
    S = volume.shape[0]
    keep = max(1, min(keep, S))
    
    # Calculate gradient scores for ALL slices
    print(f"      Calculating gradient scores for {S} slices...")
    gradient_scores = []
    for i in range(S):
        score = _gradient_score(volume[i])
        gradient_scores.append((i, score))
    
    # Sort by gradient score (descending)
    gradient_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"      Top 5 gradient slices: {[(idx, f'{score:.4f}') for idx, score in gradient_scores[:5]]}")
    
    chosen = []
    for idx, score in gradient_scores:
        if len(chosen) >= keep:
            break
        
        # Check minimal separation
        too_close = False
        if min_sep > 0:
            for existing_idx in chosen:
                if abs(idx - existing_idx) < min_sep:
                    too_close = True
                    break
        
        if not too_close:
            chosen.append(idx)
    
    # Relax constraint if needed
    if len(chosen) < keep:
        print(f"      Relaxing min_sep constraint...")
        for idx, score in gradient_scores:
            if idx not in chosen:
                chosen.append(idx)
                if len(chosen) >= keep:
                    break
    
    final = sorted(chosen)
    print(f"      Selected {len(final)} slices (pure gradient): {final}")
    return final

# ============= Annotation Processing =============

def parse_box_annotations_from_df(boxes_df: pd.DataFrame, patient_id: str, study_uid: str, view: str) -> List[Dict]:
    """Parse box annotations from the boxes dataframe."""
    case_boxes = boxes_df[(boxes_df['PatientID'] == patient_id) & 
                          (boxes_df['StudyUID'] == study_uid) & 
                          (boxes_df['View'] == view)]
    
    boxes = []
    for _, row in case_boxes.iterrows():
        boxes.append({
            'slice': int(row['Slice']),
            'x': int(row['X']),
            'y': int(row['Y']),
            'width': int(row['Width']),
            'height': int(row['Height']),
            'class': row['Class'] if 'Class' in row else 'Unknown'
        })
    
    return boxes

# ============= Main Preprocessor =============

class DBTCancerBenignNormalPreprocessor:
    KEEP_SLICES = 15
    MIN_SEP = 1
    
    def __init__(self,
                 manifest_dir: str,
                 output_dir: str,
                 target_size: Tuple[int, int] = (1024, 1024),
                 skip_existing: bool = True,
                 bbox_expansion: float = 1.2):
        
        self.is_wsl = self._detect_wsl()
        if self.is_wsl:
            manifest_dir = self._convert_to_wsl_path(manifest_dir)
            output_dir = self._convert_to_wsl_path(output_dir)
        
        self.manifest_dir = Path(manifest_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.skip_existing = skip_existing
        self.bbox_expansion = bbox_expansion
        
        # Create directories for all three classes
        for split in ['train', 'validation', 'test']:
            for class_name in ['Benign', 'Cancer', 'Normal']:
                (self.output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
                (self.output_dir / 'viz' / split / class_name).mkdir(parents=True, exist_ok=True)
                (self.output_dir / 'metadata' / split / class_name).mkdir(parents=True, exist_ok=True)
        
        base_path = "/mnt/e" if self.is_wsl else "E:"
        self.csv_paths = {
            'train': {
                'labels': Path(base_path) / "BCS-DBT-labels-train-v2.csv",
                'filepaths': Path(base_path) / "BCS-DBT-file-paths-train-v2.csv",
                'boxes': Path(base_path) / "BCS-DBT-boxes-train-v2.csv"
            },
            'validation': {
                'labels': Path(base_path) / "BCS-DBT-labels-validation-PHASE-2-Jan-2024.csv",
                'filepaths': Path(base_path) / "BCS-DBT-file-paths-validation-v2.csv",
                'boxes': Path(base_path) / "BCS-DBT-boxes-validation-v2-PHASE-2-Jan-2024.csv"
            },
            'test': {
                'labels': Path(base_path) / "BCS-DBT-labels-test-PHASE-2.csv",
                'filepaths': Path(base_path) / "BCS-DBT-file-paths-test-v2.csv",
                'boxes': Path(base_path) / "BCS-DBT-boxes-test-v2-PHASE-2-Jan-2024.csv"
            }
        }
    
    def _detect_wsl(self) -> bool:
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except Exception:
            return False
    
    def _convert_to_wsl_path(self, windows_path: str) -> str:
        if windows_path.startswith("E:\\") or windows_path.startswith("E:/"):
            return windows_path.replace("E:\\", "/mnt/e/").replace("E:/", "/mnt/e/").replace("\\", "/")
        elif windows_path.startswith(r"E:"):
            return windows_path.replace(r"E:", "/mnt/e").replace("\\", "/")
        return windows_path
    
    def load_cancer_benign_cases(self, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Cancer and Benign cases."""
        print(f"\nLoading {split} Cancer/Benign cases...")
        
        labels_df = pd.read_csv(self.csv_paths[split]['labels'])
        filepaths_df = pd.read_csv(self.csv_paths[split]['filepaths'])
        boxes_df = pd.read_csv(self.csv_paths[split]['boxes'])
        
        print(f"  Loaded {len(boxes_df)} box annotations")
        
        labels_df = labels_df[labels_df['Actionable'] == 0]
        
        cancer_col = benign_col = None
        for col in labels_df.columns:
            if col.lower() == 'cancer':
                cancer_col = col
            elif col.lower() == 'benign':
                benign_col = col
        
        if not cancer_col or not benign_col:
            raise ValueError(f"Columns not found. Available: {list(labels_df.columns)}")
        
        cancer_cases = labels_df[labels_df[cancer_col] == 1].copy()
        benign_cases = labels_df[labels_df[benign_col] == 1].copy()
        
        cancer_cases['Class'] = 'Cancer'
        benign_cases['Class'] = 'Benign'
        
        annotated_cases = pd.concat([cancer_cases, benign_cases])
        df = pd.merge(annotated_cases, filepaths_df, on=['PatientID', 'StudyUID', 'View'])
        
        if 'Class' in boxes_df.columns:
            boxes_df = boxes_df.drop(columns=['Class'])
        
        boxes_df = pd.merge(
            boxes_df, 
            annotated_cases[['PatientID', 'StudyUID', 'View', 'Class']], 
            on=['PatientID', 'StudyUID', 'View'],
            how='inner'
        )
        
        print(f"  Cancer: {(df['Class'] == 'Cancer').sum()}")
        print(f"  Benign: {(df['Class'] == 'Benign').sum()}")
        
        return df, boxes_df
    
    def load_normal_cases(self, split: str, sample_n: Optional[int] = None) -> pd.DataFrame:
        """Load Normal cases (non-actionable, Cancer=0, Benign=0)."""
        print(f"\nLoading {split} Normal cases...")
        
        labels_df = pd.read_csv(self.csv_paths[split]['labels'])
        filepaths_df = pd.read_csv(self.csv_paths[split]['filepaths'])
        
        # Find column names
        cancer_col = benign_col = None
        for col in labels_df.columns:
            if col.lower() == 'cancer':
                cancer_col = col
            elif col.lower() == 'benign':
                benign_col = col
        
        if not cancer_col or not benign_col:
            raise ValueError(f"Columns not found. Available: {list(labels_df.columns)}")
        
        # Normal = non-actionable AND Cancer=0 AND Benign=0
        normal_cases = labels_df[
            (labels_df['Actionable'] == 0) & 
            (labels_df[cancer_col] == 0) & 
            (labels_df[benign_col] == 0)
        ].copy()
        
        normal_cases['Class'] = 'Normal'
        
        print(f"  Total Normal cases available: {len(normal_cases)}")
        
        # Random sample if specified
        if sample_n and sample_n < len(normal_cases):
            normal_cases = normal_cases.sample(n=sample_n, random_state=42)
            print(f"  Sampled: {len(normal_cases)}")
        
        df = pd.merge(normal_cases, filepaths_df, on=['PatientID', 'StudyUID', 'View'])
        print(f"  After merge with filepaths: {len(df)}")
        
        return df
    
    def find_file_path(self, descriptive_path: str) -> Optional[Path]:
        p = descriptive_path.replace("\\", "/")
        import re
        p_na = re.sub(r'(\d+\.000000)-(\d+)/', r'\1-NA-\2/', p)
        
        variants = [
            p_na,
            p_na.replace("MAMMO DIAGNOSTIC DIGITAL BILATERAL", "MAMMO diagnostic digital bilateral"),
            p_na.replace("MAMMO SCREENING DIGITAL BILATERAL", "MAMMO screening digital bilateral"),
            p,
            p.replace("MAMMO DIAGNOSTIC DIGITAL BILATERAL", "MAMMO diagnostic digital bilateral"),
            p.replace("MAMMO SCREENING DIGITAL BILATERAL", "MAMMO screening digital bilateral"),
        ]
        
        for v in set(variants):
            full = self.manifest_dir / v
            if full.exists():
                return full
        return None
    
    @torch.inference_mode()
    def process_volume_with_annotations(
        self, 
        dicom_path: Path, 
        view: str,
        boxes: List[Dict]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int], Tuple[int, int], List[Dict]]:
        """
        Process Cancer/Benign volume with GRADIENT-BASED selection.
        Returns: (z_scored_vol, preview_vol, selected_indices, original_shape, corrected_boxes)
        """
        try:
            vol, original_shape, was_flipped = dcmread_full_volume(dicom_path, view)
            total_slices = vol.shape[0]
            
            corrected_boxes = boxes
            expanded_boxes = expand_boxes(corrected_boxes, self.bbox_expansion, original_shape)
            
            # Get annotated slice indices
            annotated_slices = sorted(list(set([b['slice'] for b in expanded_boxes])))
            
            # GRADIENT-BASED SELECTION (annotated first, then gradient fill)
            selected_indices = select_annotated_plus_gradient_based(
                vol, 
                annotated_slices,
                keep=self.KEEP_SLICES, 
                min_sep=self.MIN_SEP
            )
            
            print(f"    Full vol: {total_slices} slices → Selected: {len(selected_indices)}")
            print(f"    Annotated slices: {annotated_slices}")
            if was_flipped:
                print(f"    ⚠️  Flipped - boxes transformed")
            if self.bbox_expansion > 1.0:
                print(f"    📦 Boxes expanded by {self.bbox_expansion}x")
            
            # Extract selected slices
            selected_volume = vol[selected_indices]
            
            # Resize
            tensor = torch.from_numpy(_nan_to_num32(selected_volume)).unsqueeze(1)
            if device.type == 'cuda':
                tensor = tensor.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            resized = F.interpolate(tensor, size=self.target_size, mode='bilinear', align_corners=False)
            output = resized.squeeze(1).cpu().numpy().astype(np.float32)
            output = _nan_to_num32(output)
            
            preview_volume = output.copy()
            
            # Z-score
            std = float(output.std())
            if not np.isfinite(std) or std < 1e-6:
                output[:] = 0.0
            else:
                mean = float(output.mean())
                output = (output - mean) / std
                output = _nan_to_num32(output)
            
            return output, preview_volume, selected_indices, original_shape, expanded_boxes
            
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, [], (0, 0), []
    
    @torch.inference_mode()
    def process_normal_volume(
        self, 
        dicom_path: Path, 
        view: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[int], Tuple[int, int]]:
        """
        Process Normal volume with PURE gradient-based selection.
        Returns: (z_scored_vol, preview_vol, selected_indices, original_shape)
        """
        try:
            vol, original_shape, was_flipped = dcmread_full_volume(dicom_path, view)
            total_slices = vol.shape[0]
            
            # PURE GRADIENT SELECTION (no annotated slices)
            selected_indices = select_pure_gradient_based(
                vol, 
                keep=self.KEEP_SLICES, 
                min_sep=self.MIN_SEP
            )
            
            print(f"    Full vol: {total_slices} slices → Selected: {len(selected_indices)}")
            
            # Extract selected slices
            selected_volume = vol[selected_indices]
            
            # Resize
            tensor = torch.from_numpy(_nan_to_num32(selected_volume)).unsqueeze(1)
            if device.type == 'cuda':
                tensor = tensor.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
            resized = F.interpolate(tensor, size=self.target_size, mode='bilinear', align_corners=False)
            output = resized.squeeze(1).cpu().numpy().astype(np.float32)
            output = _nan_to_num32(output)
            
            preview_volume = output.copy()
            
            # Z-score
            std = float(output.std())
            if not np.isfinite(std) or std < 1e-6:
                output[:] = 0.0
            else:
                mean = float(output.mean())
                output = (output - mean) / std
                output = _nan_to_num32(output)
            
            return output, preview_volume, selected_indices, original_shape
            
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, [], (0, 0)
    
    def create_visualization_with_boxes(
        self,
        preview_volume: np.ndarray,
        selected_indices: List[int],
        boxes: List[Dict],
        original_shape: Tuple[int, int],
        case_png: Path,
        case_id: str
    ):
        """Create visualization for Cancer/Benign with boxes."""
        if preview_volume is None or preview_volume.ndim != 3:
            return
        
        n_slices = preview_volume.shape[0]
        n_cols = 5
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        cancer_count = sum(1 for b in boxes if b.get('class') == 'Cancer')
        benign_count = sum(1 for b in boxes if b.get('class') == 'Benign')
        
        fig.suptitle(f'Case: {case_id} | Benign: {benign_count} (YELLOW), Cancer: {cancer_count} (RED)', 
                     fontsize=12, fontweight='bold')
        
        scale_h = self.target_size[0] / original_shape[0]
        scale_w = self.target_size[1] / original_shape[1]
        
        for idx in range(n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            if idx < n_slices:
                slice_img = preview_volume[idx]
                orig_slice_idx = selected_indices[idx]
                
                u8 = _safe_percentile_u8(slice_img, 0.5, 99.5)
                ax.imshow(u8, cmap='gray')
                
                for box in boxes:
                    x_scaled = box['x'] * scale_w
                    y_scaled = box['y'] * scale_h
                    w_scaled = box['width'] * scale_w
                    h_scaled = box['height'] * scale_h
                    
                    box_class = box.get('class', 'Unknown')
                    
                    if box_class == 'Benign':
                        color = 'yellow'
                    elif box_class == 'Cancer':
                        color = 'red'
                    else:
                        color = 'cyan'
                    
                    is_annotated_slice = (box['slice'] == orig_slice_idx)
                    
                    linestyle = '-' if is_annotated_slice else '--'
                    linewidth = 2.5 if is_annotated_slice else 1.5
                    alpha = 0.8 if is_annotated_slice else 0.4
                    
                    rect = Rectangle(
                        (x_scaled, y_scaled),
                        w_scaled, h_scaled,
                        linewidth=linewidth,
                        edgecolor=color,
                        linestyle=linestyle,
                        facecolor='none',
                        alpha=alpha
                    )
                    ax.add_patch(rect)
                
                slice_boxes = [b for b in boxes if b['slice'] == orig_slice_idx]
                
                title = f'Slice {orig_slice_idx}'
                if slice_boxes:
                    title += f' ★ ({len(slice_boxes)})'
                    ax.set_title(title, fontsize=9, color='lime', fontweight='bold')
                else:
                    ax.set_title(title, fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        case_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(case_png, dpi=100, bbox_inches='tight')
        plt.close()
    
    def create_normal_visualization(
        self,
        preview_volume: np.ndarray,
        selected_indices: List[int],
        case_png: Path,
        case_id: str
    ):
        """Create simple visualization for Normals (no boxes)."""
        if preview_volume is None or preview_volume.ndim != 3:
            return
        
        n_slices = preview_volume.shape[0]
        n_cols = 5
        n_rows = (n_slices + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'NORMAL: {case_id}', fontsize=12, fontweight='bold', color='green')
        
        for idx in range(n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            if idx < n_slices:
                slice_img = preview_volume[idx]
                orig_slice_idx = selected_indices[idx]
                
                u8 = _safe_percentile_u8(slice_img, 0.5, 99.5)
                ax.imshow(u8, cmap='gray')
                ax.set_title(f'Slice {orig_slice_idx}', fontsize=9)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        case_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(case_png, dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_annotation_metadata(
        self, 
        case_id: str, 
        boxes: List[Dict], 
        selected_indices: List[int], 
        split: str, 
        class_name: str,
        original_shape: Tuple[int, int]
    ):
        """Save metadata for Cancer/Benign with scaled coordinates."""
        scale_h = self.target_size[0] / original_shape[0]
        scale_w = self.target_size[1] / original_shape[1]
        
        slice_idx_map = {orig: new for new, orig in enumerate(selected_indices)}
        
        scaled_boxes = []
        for box in boxes:
            orig_slice = box['slice']
            if orig_slice in slice_idx_map:
                scaled_boxes.append({
                    'slice_idx': slice_idx_map[orig_slice],
                    'original_slice': orig_slice,
                    'x': int(box['x'] * scale_w),
                    'y': int(box['y'] * scale_h),
                    'width': int(box['width'] * scale_w),
                    'height': int(box['height'] * scale_h),
                    'class': box.get('class', 'Unknown'),
                    'x_original': box['x'],
                    'y_original': box['y'],
                    'width_original': box['width'],
                    'height_original': box['height']
                })
        
        metadata = {
            'case_id': case_id,
            'class': class_name,
            'volume_shape': list(self.target_size),
            'num_slices': len(selected_indices),
            'original_shape': list(original_shape),
            'scale_factors': {'scale_h': float(scale_h), 'scale_w': float(scale_w)},
            'selected_slices': selected_indices,
            'total_boxes': len(scaled_boxes),
            'boxes': scaled_boxes,
            'annotated_slice_indices': sorted(list(set([b['slice_idx'] for b in scaled_boxes])))
        }
        
        meta_dir = self.output_dir / 'metadata' / split / class_name
        with open(meta_dir / f"{case_id}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_normal_metadata(
        self, 
        case_id: str, 
        selected_indices: List[int], 
        split: str,
        original_shape: Tuple[int, int]
    ):
        """Save metadata for Normal case (no boxes)."""
        scale_h = self.target_size[0] / original_shape[0]
        scale_w = self.target_size[1] / original_shape[1]
        
        metadata = {
            'case_id': case_id,
            'class': 'Normal',
            'volume_shape': list(self.target_size),
            'num_slices': len(selected_indices),
            'original_shape': list(original_shape),
            'scale_factors': {'scale_h': float(scale_h), 'scale_w': float(scale_w)},
            'selected_slices': selected_indices,
            'total_boxes': 0,
            'boxes': [],
            'annotated_slice_indices': []
        }
        
        meta_dir = self.output_dir / 'metadata' / split / 'Normal'
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / f"{case_id}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def preprocess_cancer_benign(self, split: str):
        """Preprocess Cancer and Benign cases."""
        print(f"\n{'='*50}\nProcessing {split} CANCER/BENIGN\n{'='*50}")
        
        df, boxes_df = self.load_cancer_benign_cases(split)
        stats = {'Benign': 0, 'Cancer': 0, 'skipped': 0, 'already_exists': 0, 'no_boxes': 0}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split} Cancer/Benign"):
            case_id = f"{row['PatientID']}_{row['StudyUID']}_{row['View']}"
            class_name = row['Class']
            
            npy_path = self.output_dir / split / class_name / f"{case_id}.npy"
            viz_png = self.output_dir / 'viz' / split / class_name / f"{case_id}.png"
            meta_json = self.output_dir / 'metadata' / split / class_name / f"{case_id}.json"
            
            if self.skip_existing and npy_path.exists() and viz_png.exists() and meta_json.exists():
                stats['already_exists'] += 1
                continue
            
            boxes = parse_box_annotations_from_df(boxes_df, row['PatientID'], 
                                                 row['StudyUID'], row['View'])
            
            if not boxes:
                print(f"\n  ⚠️  No boxes: {case_id}")
                stats['no_boxes'] += 1
                continue
            
            dicom_path = row['descriptive_path'].replace("\\", "/")
            full_path = self.find_file_path(dicom_path)
            if not full_path:
                stats['skipped'] += 1
                continue
            
            print(f"\n  {class_name}: {case_id}")
            
            vol_z, vol_p, sel_idx, orig_shape, corr_boxes = \
                self.process_volume_with_annotations(full_path, row['View'], boxes)
            
            if vol_z is not None:
                np.save(npy_path, _nan_to_num32(vol_z))
                
                self.create_visualization_with_boxes(
                    vol_p, sel_idx, corr_boxes, orig_shape, viz_png, case_id
                )
                
                self.save_annotation_metadata(
                    case_id, corr_boxes, sel_idx, split, class_name, orig_shape
                )
                
                stats[class_name] += 1
            else:
                stats['skipped'] += 1
        
        print(f"\n{split} Cancer/Benign stats: {stats}")
        return stats
    
    def preprocess_normals(self, split: str, sample_n: int = 200):
        """Preprocess Normal cases."""
        print(f"\n{'='*50}\nProcessing {split} NORMALS (sampling {sample_n})\n{'='*50}")
        
        df = self.load_normal_cases(split, sample_n=sample_n)
        stats = {'Normal': 0, 'skipped': 0, 'already_exists': 0}
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split} Normals"):
            case_id = f"{row['PatientID']}_{row['StudyUID']}_{row['View']}"
            
            npy_path = self.output_dir / split / 'Normal' / f"{case_id}.npy"
            viz_png = self.output_dir / 'viz' / split / 'Normal' / f"{case_id}.png"
            meta_json = self.output_dir / 'metadata' / split / 'Normal' / f"{case_id}.json"
            
            if self.skip_existing and npy_path.exists() and viz_png.exists() and meta_json.exists():
                stats['already_exists'] += 1
                continue
            
            dicom_path = row['descriptive_path'].replace("\\", "/")
            full_path = self.find_file_path(dicom_path)
            if not full_path:
                stats['skipped'] += 1
                continue
            
            print(f"\n  Normal: {case_id}")
            
            vol_z, vol_p, sel_idx, orig_shape = self.process_normal_volume(full_path, row['View'])
            
            if vol_z is not None:
                np.save(npy_path, _nan_to_num32(vol_z))
                self.create_normal_visualization(vol_p, sel_idx, viz_png, case_id)
                self.save_normal_metadata(case_id, sel_idx, split, orig_shape)
                stats['Normal'] += 1
            else:
                stats['skipped'] += 1
        
        print(f"\n{split} Normal stats: {stats}")
        return stats
    
    def run(self, 
            include_cancer_benign: bool = True,
            include_normals: bool = True, 
            normal_sample_n: int = 200):
        """
        Run preprocessing pipeline.
        
        Args:
            include_cancer_benign: Process Cancer and Benign cases
            include_normals: Process Normal cases
            normal_sample_n: Number of Normal cases to sample per split
        """
        print("\n" + "="*60)
        print("DBT CANCER/BENIGN/NORMAL: GRADIENT-BASED SELECTION")
        print("="*60)
        print(f"Manifest: {self.manifest_dir}")
        print(f"Output:   {self.output_dir}")
        print(f"Target:   {self.target_size}, {self.KEEP_SLICES} slices")
        print(f"\n✨ Features:")
        print("  - OFFICIAL Duke laterality & windowing")
        print("  - Cancer/Benign: Annotated slices FIRST, then gradient fill")
        print("  - Normal: Pure gradient-based selection")
        print("  - Color mapping: Benign=YELLOW, Cancer=RED, Normal=GREEN")
        print(f"\n📊 Processing plan:")
        print(f"  - Cancer/Benign: {'YES' if include_cancer_benign else 'NO'}")
        print(f"  - Normals: {'YES' if include_normals else 'NO'} (sample_n={normal_sample_n})")
        
        if not self.manifest_dir.exists():
            print(f"\nERROR: Manifest not found at {self.manifest_dir}")
            return
        
        all_stats = {}
        
        for split in ['train', 'validation', 'test']:
            all_stats[split] = {}
            
            # Check if CSV files exist
            if not self.csv_paths[split]['labels'].exists():
                print(f"\n⚠️  Skipping {split}: labels CSV not found")
                continue
            
            if include_cancer_benign and self.csv_paths[split]['boxes'].exists():
                cb_stats = self.preprocess_cancer_benign(split)
                all_stats[split].update(cb_stats)
            
            if include_normals:
                n_stats = self.preprocess_normals(split, sample_n=normal_sample_n)
                all_stats[split].update(n_stats)
        
        # Print final summary
        print("\n" + "="*60)
        print("COMPLETE! FINAL SUMMARY")
        print("="*60)
        
        for split in ['train', 'validation', 'test']:
            print(f"\n{split.upper()}:")
            for cls in ['Cancer', 'Benign', 'Normal']:
                class_dir = self.output_dir / split / cls
                if class_dir.exists():
                    n = len(list(class_dir.glob("*.npy")))
                    print(f"  {cls}: {n}")
        
        # Total counts
        print("\n" + "-"*40)
        print("TOTAL ACROSS ALL SPLITS:")
        for cls in ['Cancer', 'Benign', 'Normal']:
            total = 0
            for split in ['train', 'validation', 'test']:
                class_dir = self.output_dir / split / cls
                if class_dir.exists():
                    total += len(list(class_dir.glob("*.npy")))
            print(f"  {cls}: {total}")


def main():
    # ============= CONFIGURATION =============
    MANIFEST_DIR = r"E:\manifest-1617905855234"
    OUTPUT_DIR = r"E:\DBT_CancerBenignNormal_Gradient_1024_15"
    
    # Bbox expansion for Cancer/Benign
    BBOX_EXPANSION = 1.2
    
    # Number of Normal cases to sample per split
    # Adjust to balance the dataset
    # Cancer ~124, Benign ~70, so Normal ~200 gives reasonable balance
    NORMAL_SAMPLE_N = 9000
    
    # =========================================
    
    preprocessor = DBTCancerBenignNormalPreprocessor(
        manifest_dir=MANIFEST_DIR,
        output_dir=OUTPUT_DIR,
        target_size=(1024, 1024),
        skip_existing=True,
        bbox_expansion=BBOX_EXPANSION
    )
    
    # Run full pipeline
    preprocessor.run(
        include_cancer_benign=False,
        include_normals=True,
        normal_sample_n=NORMAL_SAMPLE_N
    )


if __name__ == "__main__":
    main()