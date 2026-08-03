#!/usr/bin/env python3
"""
recount_table2.py  ->  recounts the dataset category totals from the label CSVs.

Read-only. Recounts BCS-DBT category totals at the PATIENT level directly
from the official label CSVs, so the corrected Table 2 is grounded in the
data rather than patched by hand.

The raw label files are per-volume and a patient can carry more than one
finding, so counts are collapsed to one row per patient and a category is
assigned by precedence:  cancer > benign > actionable > normal.

Run:
    conda/venv active, then
    python recount_table2.py
"""
import pandas as pd
from pathlib import Path

BASE = Path("/mnt/e")
FILES = {
    "train":      "BCS-DBT-labels-train-v2.csv",
    "validation": "BCS-DBT-labels-validation-PHASE-2-Jan-2024.csv",
    "test":       "BCS-DBT-labels-test-PHASE-2.csv",
}

def col(df, name):
    for c in df.columns:
        if c.lower() == name:
            return c
    raise KeyError(f"column {name!r} not found in {list(df.columns)}")

print(f"{'Split':11s} {'Patients':>9} {'Normal':>8} {'Action.':>8} {'Benign':>8} {'Cancer':>8} {'Sum':>8}")
print("-" * 64)

rows = {}
for split, fname in FILES.items():
    fp = BASE / fname
    if not fp.exists():
        print(f"{split:11s}  [file not found: {fp}]")
        continue
    df = pd.read_csv(fp)
    cc, bc, ac = col(df, "cancer"), col(df, "benign"), col(df, "actionable")

    # collapse to one row per patient: a patient is positive for a category
    # if ANY of their volumes is.
    pat = df.groupby("PatientID").agg({cc: "max", bc: "max", ac: "max"})

    n       = len(pat)
    cancer  = int((pat[cc] == 1).sum())
    benign  = int(((pat[bc] == 1) & (pat[cc] == 0)).sum())
    action  = int(((pat[ac] == 1) & (pat[cc] == 0) & (pat[bc] == 0)).sum())
    normal  = n - cancer - benign - action
    s = normal + action + benign + cancer

    rows[split] = (n, normal, action, benign, cancer)
    print(f"{split:11s} {n:9d} {normal:8d} {action:8d} {benign:8d} {cancer:8d} {s:8d}")

print("-" * 64)
print("Percentages are of the split patient total. Precedence: cancer > benign > actionable > normal.")
print()
print("LaTeX rows for Table 2 (\\label{tab:dataset_full}):")
for split, key in [("Training","train"), ("Validation","validation"), ("Test","test")]:
    if key not in rows:
        continue
    n, normal, action, benign, cancer = rows[key]
    def pct(x): return f"{100*x/n:.1f}\\%"
    print(f"  {split:11s}& {n:,} & {normal:,} ({pct(normal)}) & {action} ({pct(action)}) "
          f"& {benign} ({pct(benign)}) & {cancer} ({pct(cancer)}) \\\\")
