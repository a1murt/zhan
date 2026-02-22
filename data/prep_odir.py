"""
ODIR-5K preprocessing script.

Input  : data/raw/full_df.csv       (one row per patient, 6 392 rows)
Output : data/processed/odir_clean.csv  (one row per eye, ~12 784 rows)

Columns in output
-----------------
Patient_ID   : int       original 'ID'
Eye          : str       'Left' | 'Right'
Image_Name   : str       fundus filename (e.g. '0_left.jpg')
Age          : int       patient age in years
Gender       : int       1 = Male, 0 = Female
Myopia       : int       target label — 'M' column (0 = no myopia, 1 = myopia)
"""

import os
import sys
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))   # …/data/
RAW_CSV       = os.path.join(BASE_DIR, "raw", "full_df.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
OUT_CSV       = os.path.join(PROCESSED_DIR, "odir_clean.csv")

# ── Sanity check ──────────────────────────────────────────────────────────────
if not os.path.exists(RAW_CSV):
    print(f"[ERROR] Source file not found: {RAW_CSV}", file=sys.stderr)
    sys.exit(1)

os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"[INFO] Reading {RAW_CSV} …")
raw = pd.read_csv(RAW_CSV)
print(f"[INFO] Loaded {len(raw):,} patient rows, {raw.shape[1]} columns.")

# ── Deduplicate by Patient ID (keep first occurrence) ────────────────────────
before = len(raw)
raw.drop_duplicates(subset=["ID"], keep="first", inplace=True)
raw.reset_index(drop=True, inplace=True)
dupes = before - len(raw)
if dupes:
    print(f"[WARN] Removed {dupes:,} duplicate patient rows. {len(raw):,} unique patients remain.")
else:
    print(f"[INFO] No duplicate Patient IDs found.")

# ── Flatten: one row per eye ──────────────────────────────────────────────────
# Build Left-eye frame
left = pd.DataFrame({
    "Patient_ID": raw["ID"],
    "Eye":        "Left",
    "Image_Name": raw["Left-Fundus"],
    "Age":        raw["Patient Age"],
    "Gender":     raw["Patient Sex"].str.strip().map({"Male": 1, "Female": 0}),
    "Myopia":     raw["M"],
})

# Build Right-eye frame
right = pd.DataFrame({
    "Patient_ID": raw["ID"],
    "Eye":        "Right",
    "Image_Name": raw["Right-Fundus"],
    "Age":        raw["Patient Age"],
    "Gender":     raw["Patient Sex"].str.strip().map({"Male": 1, "Female": 0}),
    "Myopia":     raw["M"],
})

df = pd.concat([left, right], ignore_index=True)
print(f"[INFO] After flattening: {len(df):,} eye rows.")

# ── Drop rows with missing Image_Name ─────────────────────────────────────────
before = len(df)
df.dropna(subset=["Image_Name"], inplace=True)
dropped = before - len(df)
if dropped:
    print(f"[WARN] Dropped {dropped} rows with NaN Image_Name.")
else:
    print("[INFO] No NaN Image_Name rows found — dataset is clean.")

# ── Drop rows with missing Gender (unmapped values) ───────────────────────────
before = len(df)
df.dropna(subset=["Gender"], inplace=True)
dropped = before - len(df)
if dropped:
    print(f"[WARN] Dropped {dropped} rows with unrecognised Gender value.")

# ── Type normalisation ────────────────────────────────────────────────────────
df["Patient_ID"] = df["Patient_ID"].astype(int)
df["Age"]        = df["Age"].astype(int)
df["Gender"]     = df["Gender"].astype(int)
df["Myopia"]     = df["Myopia"].astype(int)

# ── Sort for reproducibility ─────────────────────────────────────────────────
df.sort_values(["Patient_ID", "Eye"], inplace=True)
df.reset_index(drop=True, inplace=True)

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"[INFO] Saved -> {OUT_CSV}")
print(f"[INFO] Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns.")

# ── Quick class-balance report ────────────────────────────────────────────────
print()
print("=== Myopia label distribution ===")
counts = df["Myopia"].value_counts().rename({0: "No Myopia (0)", 1: "Myopia (1)"})
for label, n in counts.items():
    pct = n / len(df) * 100
    print(f"  {label}: {n:,}  ({pct:.1f}%)")
