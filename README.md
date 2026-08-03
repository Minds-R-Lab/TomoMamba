# TomoMamba

Code accompanying *TomoMamba: A Two-Stage Model with State-Space Cross-Slice
Propagation for Breast Cancer Diagnosis in Digital Breast Tomosynthesis*.

Everything here operates on the public
[BCS-DBT](https://www.cancerimagingarchive.net/collection/breast-cancer-screening-dbt/)
dataset. No external training data is used, and no private data is included in
this repository.

---

## Before you start

**Paths are hardcoded.** Every script assumes the dataset lives on an `E:`
drive, written as `E:\...` in the preprocessing scripts (Windows) and
`/mnt/e/...` elsewhere (WSL2). Edit the constants near the top of each script,
or set them via the command-line flags where available. The two you will need
most often:

```
DATA_ROOT   = /mnt/e/DBT_CancerBenignNormal_Gradient_1024_15
MANIFEST_DIR = /mnt/e/manifest-1617905855234        # raw TCIA download
```

**Environment.** Python 3.11+, PyTorch 2.8, CUDA 12.8. The reported results were
produced on a single NVIDIA RTX 4090 Laptop GPU (16 GB).

```bash
pip install torch torchvision numpy scipy scikit-learn opencv-python \
            pydicom pylibjpeg pylibjpeg-libjpeg matplotlib pandas tqdm thop
pip install mamba-ssm causal-conv1d          # requires a CUDA toolchain
```

If `mamba-ssm` will not build, `centernet_models.py` falls back to a pure
PyTorch scan. It is slower and numerically close but not bit-identical.

**Data layout.** After preprocessing:

```
DBT_CancerBenignNormal_Gradient_1024_15/
├── train/      {Normal,Benign,Cancer}/*.npy      # (15, 1024, 1024) float32
├── validation/ {Normal,Benign,Cancer}/*.npy
├── test/       {Normal,Benign,Cancer}/*.npy
└── metadata/   {train,validation,test}/{Benign,Cancer}/*.json
```

Filenames are `{PatientID}_{StudyUID}_{view}.npy`, where view is one of
`lcc`, `rcc`, `lmlo`, `rmlo`, occasionally with a trailing digit
(`lcc1`) when a study contains more than one acquisition of the same view.
`splits.csv` in the repository root lists the split, class, patient, study and
view of every volume, so the partitions can be reconstructed without rerunning
preprocessing.

---

## 1. Preprocessing

```bash
# Cancer, Benign and Normal for train and validation.
# Despite the historical filename, this script handles all three classes
# through the include_cancer_benign / include_normals flags.
python preprocessing_pipeline.py

# Normal volumes for the test partition (needed for the end-to-end evaluation).
python preprocess_test_normals.py
```

Key settings, which must not be changed if you intend to reproduce the reported
numbers: `KEEP_SLICES = 15`, `MIN_SEP = 1`, `BBOX_EXPANSION = 1.2`, target size
`1024 x 1024`.

Slice selection retains any slice carrying a radiologist bounding box, fills the
remaining slots by a gradient and Laplacian score
(`0.6 * grad + 0.4 * lap/1000`), then sorts the selection by original depth
index so anatomical order is preserved. Alternative selection strategies used in
the ablation are produced by:

```bash
python make_slice_variants.py --mode random       # or uniform, contiguous,
python make_slice_variants.py --mode pure_grad    # or gradient (the control)
```

`pure_grad` performs gradient selection with no retention of annotated slices.
It is the annotation-independent condition reported in Section 5.8.

---

## 2. Training

```bash
# Stage 1: breast-level screening
python stage1_screening.py

# Stage 2: joint detection and classification
python train_stage2.py --data_root <DATA_ROOT> --save_dir <OUT> --seed 42

# Controlled baselines (1 = ResNet-18 classifier, 2 = + CenterNet,
#                       3 = + BiGRU,            4 = + Transformer)
python train_baselines.py --baseline 3 --seed 42
```

All reported runs use `--epochs 200 --patience 20 --seed 42` unless stated
otherwise. Note that `train_stage2.py` evaluates during training at a detection
threshold of 0.20; every detection metric reported in the manuscript uses
0.16 and is computed separately by `eval_variant.py`.

---

## 3. Evaluation

```bash
# Per-case predictions for the main model on any dataset root
python eval_variant.py --tag gradient --split validation \
    --checkpoint <CKPT> --data_root <DATA_ROOT> --score_thresh 0.16

# Per-case predictions for the four baselines
python dump_baselines.py --split validation --baselines 1 2 3 4 \
    --dump_dir dumps_val --score_thresh 0.16

# Bootstrap CIs, max-J, Brier, matched operating points, DeLong tests
python stats_package.py --dump_dir dumps_val --ref "BASELINE_(v5.2)"
```

---

## 4. Reproducing each table and figure

Numbering follows the revised manuscript.

| Manuscript item | Script |
|---|---|
| Table 1, gate statistics | `gate_stats.py --checkpoint <CKPT> --bigru_ckpt <CKPT>` |
| Table 2, dataset counts | `recount_table2.py` |
| Table 3, evaluation units (flow) | counts derived from `splits.csv`; biopsied-breast counts include six bilateral patients |
| Table 4, main results | `eval_variant.py` |
| Table 5, performance across partitions | `eval_variant.py --split train/validation/test` |
| Table 6, seed stability | `seed_sweep.py --epochs 200 --patience 20` |
| Table 7, screening-threshold sweep | `end_to_end_eval.py` |
| Table 8, end-to-end on both partitions | `end_to_end_eval.py --split validation`, then `--split test` (writes `end_to_end_<split>.json`) |
| Table 9, controlled baselines | `dump_baselines.py` then `stats_package.py` |
| Table 10, ablations | `eval_ablations.py`, `eval_variant.py`, `multiview_fusion.py` |
| Table 11, FROC | `froc.py --generate` then `froc.py --plot` |
| Table 12, computational cost | `compute_cost.py` (run on an idle GPU) |
| Table 13, published methods | from the cited literature; no script |
| Figure 4 panels | `mamba_activations.py` |
| Figure 7, FROC curves | `froc.py --plot` |
| Cross-validation (Section 5.2) | `cross_validate_resumable.py`, then `recollect_oof.py --score_thresh 0.16` |
| Failure analysis (Section 5.7) | `failure_cases.py --dump <per_case.jsonl> --meta_root <META>` |
| Fallback classifier (Section 5.7) | `fallback_analysis.py --dump <per_case.jsonl>` |
| Depth continuity (Section 5.8) | `slice_gaps.py`, `position_check.py` |
| Annotated-slice position test (Section 5.8) | `position_leakage_test.py` |
| Annotation-independent results (Section 5.8) | `make_slice_variants.py --mode pure_grad` then `eval_variant.py` |

`results/` contains the per-case prediction dumps and summary JSON files behind
the classification, detection, ablation and compute results, so those tables can
be regenerated with the analysis scripts alone, without retraining or access to
the raw DICOM data. The cross-validation, gate-statistic, depth-continuity and
Stage 1 outputs are reproduced by running the scripts listed above.

---

## Citation

```bibtex
@article{soliman2026tomomamba,
  title   = {TomoMamba: A Two-Stage Model with State-Space Cross-Slice
             Propagation for Breast Cancer Diagnosis in Digital Breast
             Tomosynthesis},
  author  = {Soliman, Shahd and Zafari, Yalda and Rashed, Essam A. and
             Mabrok, Mohamed},
  journal = {Multimedia Tools and Applications},
  year    = {2026}
}
```

## Acknowledgements

Supported by the International Research Collaboration Co-Fund (IRCC),
grant IRCC-2025-633, a joint initiative between Qatar University and the
University of Hyogo.
