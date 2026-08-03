#!/usr/bin/env python3
"""
preprocess_test_normals.py
==========================
Preprocess the BCS-DBT test-set NORMAL volumes with the same pipeline
used for train and validation, so the test folder finally has Normals
and end-to-end evaluation becomes possible.

Reuses the exact DICOM reading, laterality correction, VOI windowing,
gradient-based slice selection, resizing, and z-scoring logic from the
existing preprocessing pipeline. NO annotation seeding is used for
Normals because they have no boxes.

Manifest source:
    /mnt/e/test_dbt/manifest-1617905855234
    /mnt/e/test_dbt/manifest-1617905855234/Breast-Cancer-Screening-DBT

Output:
    /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15/test/Normal/*.npy
    /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15/metadata/test/Normal/*.json

Run:
    python preprocess_test_normals.py
"""

import os, re, json, warnings
from pathlib import Path
from typing import Tuple, List, Union

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torch.nn.functional as F
from skimage.exposure import rescale_intensity
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ---------- config ----------
MANIFEST_DIR = Path("/mnt/e/test_dbt/manifest-1617905855234")
LABELS_CSV   = Path("/mnt/e/BCS-DBT-labels-test-PHASE-2.csv")
FILEPATHS_CSV = Path("/mnt/e/BCS-DBT-file-paths-test-v2.csv")

OUTPUT_ROOT  = Path("/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15")
NPY_DIR      = OUTPUT_ROOT / "test" / "Normal"
META_DIR     = OUTPUT_ROOT / "metadata" / "test" / "Normal"

TARGET_SIZE = (1024, 1024)   # matches the train/val Normal preprocessing
KEEP_SLICES = 15
SKIP_EXISTING = True

NPY_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# ---------- vendored utilities (identical to the training pipeline) ----------
def _nan_to_num32(x):
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

def _finite_min_max(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size == 0: return 0.0, 0.0
    return float(x.min()), float(x.max())

def _safe_percentile_u8(img, p_low=0.5, p_high=99.5):
    x = _nan_to_num32(img); finite = np.isfinite(x)
    if not finite.any(): return np.zeros_like(x, dtype=np.uint8)
    lo = np.percentile(x[finite], p_low); hi = np.percentile(x[finite], p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        vmin, vmax = _finite_min_max(x)
        if vmax <= vmin: return np.zeros_like(x, dtype=np.uint8)
        lo, hi = vmin, vmax
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)

def _get_image_laterality(pa):
    return "R" if np.sum(pa[:, 0]) < np.sum(pa[:, -1]) else "L"

def _get_window_center(ds):
    try:
        return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1050].value)
    except Exception:
        wc = getattr(ds, "WindowCenter", 1024)
        if isinstance(wc, (list, tuple)) and wc: wc = wc[0]
        return np.float32(wc)

def _get_window_width(ds):
    try:
        return np.float32(ds[0x5200, 0x9229][0][0x0028, 0x9132][0][0x0028, 0x1051].value)
    except Exception:
        ww = getattr(ds, "WindowWidth", 4096)
        if isinstance(ww, (list, tuple)) and ww: ww = ww[0]
        return np.float32(ww)

def dcmread_full_volume(fp, view):
    ds = dicom.dcmread(fp)
    try:
        ds.decompress(handler_name="pylibjpeg")
    except Exception:
        try: ds.decompress()
        except Exception: pass
    pa = _nan_to_num32(ds.pixel_array)
    if pa.ndim == 2: pa = pa[None, ...]
    original_shape = (pa.shape[1], pa.shape[2])
    view_lat = view[0].upper()
    was_flipped = _get_image_laterality(pa[0]) != view_lat
    if was_flipped:
        pa = np.flip(pa, axis=(-1, -2)).copy()
    wc, ww = _get_window_center(ds), _get_window_width(ds)
    low, high = (2*wc - ww)/2, (2*wc + ww)/2
    out = np.empty_like(pa, dtype=np.float32)
    for i in range(pa.shape[0]):
        out[i] = rescale_intensity(pa[i], in_range=(low, high), out_range="dtype").astype(np.float32, copy=False)
    return _nan_to_num32(out), original_shape, was_flipped

# ---------- gradient scoring (identical) ----------
def _gradient_magnitude(s):
    n = _safe_percentile_u8(s, 0.5, 99.5).astype(np.float32) / 255.0
    gx = cv2.Sobel(n, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx**2 + gy**2).mean())

def _laplacian_var(s):
    n = _safe_percentile_u8(s, 0.5, 99.5).astype(np.float32)
    return float(cv2.Laplacian(n, cv2.CV_32F, ksize=3).var())

def _gradient_score(s):
    return 0.6 * _gradient_magnitude(s) + 0.4 * (_laplacian_var(s) / 1000.0)

def select_gradient_only(volume, keep=15, min_sep=1):
    """No annotation seeding — Normals don't have boxes anyway."""
    S = volume.shape[0]
    keep = max(1, min(keep, S))
    scores = [(i, _gradient_score(volume[i])) for i in range(S)]
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen = []
    for idx, _ in scores:
        if len(chosen) >= keep: break
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)
    if len(chosen) < keep:  # relax min_sep if needed
        for idx, _ in scores:
            if idx not in chosen:
                chosen.append(idx)
                if len(chosen) >= keep: break
    return sorted(chosen)

# ---------- path resolver (identical) ----------
def find_file_path(descriptive_path):
    p = descriptive_path.replace("\\", "/")
    p_na = re.sub(r'(\d+\.000000)-(\d+)/', r'\1-NA-\2/', p)
    variants = {
        p_na,
        p_na.replace("MAMMO DIAGNOSTIC DIGITAL BILATERAL", "MAMMO diagnostic digital bilateral"),
        p_na.replace("MAMMO SCREENING DIGITAL BILATERAL",  "MAMMO screening digital bilateral"),
        p,
        p.replace("MAMMO DIAGNOSTIC DIGITAL BILATERAL", "MAMMO diagnostic digital bilateral"),
        p.replace("MAMMO SCREENING DIGITAL BILATERAL",  "MAMMO screening digital bilateral"),
    }
    for v in variants:
        full = MANIFEST_DIR / v
        if full.exists():
            return full
    return None

# ---------- main ----------
def main():
    print("loading test labels + filepaths...")
    labels = pd.read_csv(LABELS_CSV)
    fp = pd.read_csv(FILEPATHS_CSV)

    labels = labels[labels['Actionable'] == 0]
    normal_col = next(c for c in labels.columns if c.lower() == 'normal')
    normals = labels[labels[normal_col] == 1].copy()
    df = pd.merge(normals, fp, on=['PatientID', 'StudyUID', 'View'])
    print(f"  test Normals to process: {len(df)}")

    stats = {'ok': 0, 'skipped_existing': 0, 'file_not_found': 0, 'read_error': 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="test/Normal"):
        case_id = f"{row['PatientID']}_{row['StudyUID']}_{row['View']}"
        npy_path = NPY_DIR / f"{case_id}.npy"
        meta_path = META_DIR / f"{case_id}.json"

        if SKIP_EXISTING and npy_path.exists() and meta_path.exists():
            stats['skipped_existing'] += 1
            continue

        full = find_file_path(str(row['descriptive_path']))
        if full is None:
            stats['file_not_found'] += 1
            continue

        try:
            vol, orig_shape, was_flipped = dcmread_full_volume(full, row['View'])
        except Exception as e:
            print(f"  read error {case_id}: {e}")
            stats['read_error'] += 1
            continue

        selected = select_gradient_only(vol, keep=KEEP_SLICES, min_sep=1)
        sel_vol = vol[selected]

        # resize
        t = torch.from_numpy(_nan_to_num32(sel_vol)).unsqueeze(1)
        if device.type == 'cuda':
            t = t.pin_memory().to(device, dtype=torch.float32, non_blocking=True)
        r = F.interpolate(t, size=TARGET_SIZE, mode='bilinear', align_corners=False)
        out = _nan_to_num32(r.squeeze(1).cpu().numpy().astype(np.float32))

        # z-score
        std = float(out.std())
        if not np.isfinite(std) or std < 1e-6:
            out[:] = 0.0
        else:
            out = _nan_to_num32((out - float(out.mean())) / std)

        np.save(npy_path, out)
        with open(meta_path, "w") as f:
            json.dump({
                'case_id': case_id,
                'class': 'Normal',
                'volume_shape': list(TARGET_SIZE),
                'num_slices': len(selected),
                'original_shape': list(orig_shape),
                'selected_slices': selected,
                'total_boxes': 0,
                'boxes': [],
                'annotated_slice_indices': [],
                'was_flipped': bool(was_flipped),
            }, f, indent=2)
        stats['ok'] += 1

    print("\n=== done ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  outputs -> {NPY_DIR}")

if __name__ == "__main__":
    main()