import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/raw/myopia.csv", sep=";")

# Clean column names
df.columns = df.columns.str.replace('"', '').str.strip()

# Rename columns
df = df.rename(columns={
    "SPHEQ": "refraction_without",
    "AGE": "age",
    "AL": "axl_current",
    "SPORTHR": "outdoor_hours",
    "COMPHR": "computer_hours",
    "TVHR": "tv_hours",
    "READHR": "reading_hours",
    "STUDYHR": "study_hours",
    "MOMMY": "mom_myopia",
    "DADMY": "dad_myopia"
})

# Genetics (0,1,2)
df["genetics"] = df["mom_myopia"] + df["dad_myopia"]

# Screen time (USE NEW COLUMN NAMES ONLY)
df["screen_hours"] = (
    df["computer_hours"] +
    df["tv_hours"] +
    df["reading_hours"] +
    df["study_hours"]
)

# No cycloplegia data → copy
df["refraction_with"] = df["refraction_without"]

# axl_delta = 0 for all rows intentionally
# WHY: this dataset has no historical AXL measurements (only one timepoint)
# FUTURE: replace with real delta when a dataset with baseline + followup AXL is found
df["axl_delta"] = 0

# Create target label (Myopia)
# Use the actual ground truth label from the dataset
df["Myopia"] = df["MYOPIC"].astype(int)

# Final features
df = df[
    [
        "refraction_without",
        "refraction_with",
        "axl_current",
        "axl_delta",
        "age",
        "genetics",
        "screen_hours",
        "outdoor_hours",
        "Myopia",
    ]
]

# Clean
df = df.dropna()

# Save
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/odir_external_real.csv", index=False)

print("✅ Done. File saved in data/processed/")