"""
Produces the illustrative panels of Figure 4 from a small number of
representative cases, selected by hand for visual clarity.

The quantitative gate statistics reported in Table 1 are NOT
computed here: they come from gate_stats.py, which measures every feature cell
inside a ground-truth box across 72 validation volumes (13,167 lesion cells)
against a matched background sample.

Spatial gate selectivity map - shows WHERE on the feature map the gate is most
active. Overlay on breast image.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json, sys, glob, cv2, types
sys.path.insert(0, '.')
from centernet_models import MambaCenterNet

device = torch.device('cuda')
model = MambaCenterNet(num_classes=2, dropout=0.7, use_mamba=True, spatial_size=384).to(device)
ckpt = torch.load('/mnt/e/DBT_Stage2_MambaCenterNet_v5.2/best_model.pt',
                   map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

data_root = '/mnt/e/DBT_CancerBenignNormal_Gradient_1024_15'

captured = {}
def patched_forward(self, features):
    B, S, C, H, W = features.shape
    x = features.permute(0, 3, 4, 1, 2).reshape(B * H * W, S, C)
    out = self.ssm(x) if self.use_mamba else self.ssm(x)[0]
    out = self.norm(out)
    gate = self.gate(torch.cat([x, out], dim=-1))
    enhanced = x + gate * out
    captured['x'] = x.reshape(B, H, W, S, C).detach().cpu()
    captured['gate'] = gate.reshape(B, H, W, S, C).detach().cpu()
    captured['enhanced'] = enhanced.reshape(B, H, W, S, C).detach().cpu()
    return enhanced.reshape(B, H, W, S, C).permute(0, 3, 4, 1, 2)

model.cross_slice.forward = types.MethodType(patched_forward, model.cross_slice)

# Pick 4 good cases (middle slice lesions, mix of cancer and benign)
cases_to_show = [
    ('Cancer', 'DBT-P01700_DBT-S01353_lcc'),    # slice 4, strong lesion
    ('Cancer', 'DBT-P01207_DBT-S03000_lcc1'),   # slice 5
    ('Cancer', 'DBT-P03027_DBT-S03974_lcc'),    # slice 4
    ('Benign', 'DBT-P03728_DBT-S03978_rmlo'),   # slice 6
]

fig, axes = plt.subplots(4, 4, figsize=(14, 14), dpi=200)

for row, (cls, case_id) in enumerate(cases_to_show):
    vol_path = f'{data_root}/validation/{cls}/{case_id}.npy'
    meta_path = f'{data_root}/metadata/validation/{cls}/{case_id}.json'
    
    with open(meta_path) as f:
        meta = json.load(f)
    box = meta['boxes'][0]
    bx, by, bw, bh = float(box['x']), float(box['y']), float(box['width']), float(box['height'])
    orig_h = meta.get('volume_shape', [1024, 1024])[0]
    orig_w = meta.get('volume_shape', [1024, 1024])[-1]
    if bx > 1.0:
        bx, by, bw, bh = bx/orig_w, by/orig_h, bw/orig_w, bh/orig_h
    
    cx_feat = int((bx + bw/2) * 96)
    cy_feat = int((by + bh/2) * 96)
    cx_feat = min(95, max(0, cx_feat))
    cy_feat = min(95, max(0, cy_feat))
    lesion_slice = box.get('slice', box.get('original_slice', 7))
    selected = meta.get('selected_slices', list(range(15)))
    lesion_s_idx = selected.index(lesion_slice) if lesion_slice in selected else 7
    
    vol = np.load(vol_path).astype(np.float32)
    S = vol.shape[0]
    resized = np.zeros((S, 384, 384), dtype=np.float32)
    for s in range(S):
        resized[s] = cv2.resize(vol[s], (384, 384))
    lo, hi = np.percentile(resized, 1), np.percentile(resized, 99)
    if hi - lo > 1e-6:
        resized = (resized - lo) / (hi - lo)
    resized = np.clip(resized, 0, 1)
    volume = torch.from_numpy(resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model({'volume': volume})
    g = captured['gate'][0, cy_feat, cx_feat, lesion_s_idx, :].numpy()
    print(f"\n{'='*50}")
    print(f"CASE: {case_id} — 128 gate values at lesion position, slice {lesion_s_idx}")
    print(f"{'='*50}")
    for i in range(128):
        marker = ""
        if g[i] > 0.7: marker = " ← KEEP"
        elif g[i] < 0.3: marker = " ← BLOCK"
        print(f"  ch{i:3d}: {g[i]:.3f}{marker}")
    print(f"\nMin: {g.min():.3f}  Max: {g.max():.3f}  Mean: {g.mean():.3f}")
    print(f"Channels > 0.7 (keep):  {(g > 0.7).sum()}/128")
    print(f"Channels < 0.3 (block): {(g < 0.3).sum()}/128")
    print(f"Channels 0.3-0.7 (neutral): {((g >= 0.3) & (g <= 0.7)).sum()}/128")
    
    # Also print background for comparison
    g_bg = captured['gate'][0, 5, 5, lesion_s_idx, :].numpy()
    print(f"\nBackground (5,5) same slice:")
    print(f"Min: {g_bg.min():.3f}  Max: {g_bg.max():.3f}  Mean: {g_bg.mean():.3f}")
    print(f"Channels > 0.7: {(g_bg > 0.7).sum()}/128")
    print(f"Channels < 0.3: {(g_bg < 0.3).sum()}/128")
    # Get the breast image at lesion slice
    breast_img = resized[lesion_s_idx]
    
    # Feature magnitudes at lesion slice: (96, 96)
    x_slice = captured['x'][0, :, :, lesion_s_idx, :].numpy()  # (96, 96, 128)
    x_mag = np.linalg.norm(x_slice, axis=2)
    
    # Gate channel std at lesion slice: how selective is each position
    gate_slice = captured['gate'][0, :, :, lesion_s_idx, :].numpy()  # (96, 96, 128)
    gate_std = gate_slice.std(axis=2)  # (96, 96)
    gate_mean = gate_slice.mean(axis=2)
    
    # Enhanced magnitude
    enh_slice = captured['enhanced'][0, :, :, lesion_s_idx, :].numpy()
    enh_mag = np.linalg.norm(enh_slice, axis=2)
    
    # Enhancement ratio
    enh_ratio = (enh_mag - x_mag) / (x_mag + 1e-8)
    
    # GT box in pixel coords for overlay (384x384)
    bx_px = bx * 384
    by_px = by * 384
    bw_px = bw * 384
    bh_px = bh * 384
    
    # GT box in feature coords (96x96)  
    bx_f = bx * 96
    by_f = by * 96
    bw_f = bw * 96
    bh_f = bh * 96
    
    label_color = '#e74c3c' if cls == 'Cancer' else '#3498db'
    
    # Col 0: Breast image with GT box
    ax = axes[row, 0]
    ax.imshow(breast_img, cmap='gray')
    rect = patches.Rectangle((bx_px, by_px), bw_px, bh_px, 
                               linewidth=2, edgecolor=label_color, facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.set_title('Breast slice' if row == 0 else '', fontsize=9, fontfamily='serif')
    ax.set_ylabel(f'{cls}\nslice {lesion_s_idx}', fontsize=8, fontweight='bold', 
                  color=label_color, fontfamily='serif')
    ax.axis('off')
    
    # Col 1: Input feature magnitude |x|
    ax = axes[row, 1]
    im = ax.imshow(x_mag, cmap='hot', interpolation='bilinear')
    rect = patches.Rectangle((bx_f, by_f), bw_f, bh_f,
                               linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.set_title('Input $\\|\\mathbf{x}\\|$' if row == 0 else '', fontsize=9, fontfamily='serif')
    ax.axis('off')
    
    # Col 2: Gate channel std (selectivity map)
    ax = axes[row, 2]
    im = ax.imshow(gate_std, cmap='magma', interpolation='bilinear')
    rect = patches.Rectangle((bx_f, by_f), bw_f, bh_f,
                               linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.set_title('Gate selectivity (ch. std)' if row == 0 else '', fontsize=9, fontfamily='serif')
    ax.axis('off')
    
    # Col 3: Enhanced feature magnitude
    ax = axes[row, 3]
    im = ax.imshow(enh_mag, cmap='hot', interpolation='bilinear')
    rect = patches.Rectangle((bx_f, by_f), bw_f, bh_f,
                               linewidth=1.5, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.set_title('Enhanced $\\|\\tilde{\\mathbf{F}}\\|$' if row == 0 else '', fontsize=9, fontfamily='serif')
    ax.axis('off')

plt.suptitle('Mamba cross-slice module: spatial activation maps at lesion slices\n'
             'Cyan boxes = ground truth lesion location',
             fontsize=11, fontweight='bold', fontfamily='serif', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('/mnt/e/DBT_Stage2_MambaCenterNet_v5.2/fig4_spatial_maps.png', dpi=200, bbox_inches='tight')
plt.savefig('/mnt/e/DBT_Stage2_MambaCenterNet_v5.2/fig4_spatial_maps.pdf', dpi=200, bbox_inches='tight')
print("Saved fig4_spatial_maps.png/pdf")
print("DONE!")