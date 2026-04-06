"""
ODIR-5K Binary Myopia Dataset.

Adapts the MyopiaDataset contract for the ODIR schema:
  - Only two tabular features: Age (numerical) + Gender (already int 0/1)
  - Binary label: Myopia column (0 = no myopia, 1 = myopia)
  - Image_Path is a full absolute path built by the launcher

Column constants are intentionally kept separate from the original
dataset.py so both tasks can coexist without collision.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# UPDATED COLUMN DEFINITIONS
# ---------------------------------------------------------------------------

# CHANGED: removed Patient_ID (not in your dataset)
ODIR_PATIENT_ID_COL = None

# CHANGED: full tabular feature set (matches import_external.py output)
ODIR_TABULAR_COLS = [
    "refraction_without",
    "refraction_with",
    "axl_current",
    "axl_delta",
    "age",
    "genetics",
    "screen_hours",
    "outdoor_hours"
]

# CHANGED: label column
ODIR_LABEL_COL = "Myopia"

# CHANGED: no images used
ODIR_IMAGE_COL = None


class OdirDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        fit_imputers: bool = True,
        numerical_imputer: Optional[SimpleImputer] = None,
        categorical_imputer=None,
        label_encoders: Optional[Dict] = None,
    ) -> None:

        self.transform = transform

        # ADDED: dummy attributes to keep compatibility with existing training pipeline
        # Reason: train_odir.py expects these attributes even though we don't use them anymore
        self.categorical_imputer = None
        self.label_encoders = {}
        
        df = dataframe.copy()

        # ------------------------------------------------------------------
        # NUMERICAL IMPUTATION (for all features)
        # ------------------------------------------------------------------
        # CHANGED: use all tabular features instead of only "Age"
        if fit_imputers:
            self.numerical_imputer = SimpleImputer(strategy="median")
            df[ODIR_TABULAR_COLS] = self.numerical_imputer.fit_transform(
                df[ODIR_TABULAR_COLS]
            )
        else:
            if numerical_imputer is None:
                raise ValueError("Pass numerical_imputer when fit_imputers=False.")
            self.numerical_imputer = numerical_imputer
            df[ODIR_TABULAR_COLS] = self.numerical_imputer.transform(
                df[ODIR_TABULAR_COLS]
            )

        # ------------------------------------------------------------------
        # SCALE FEATURES
        # ------------------------------------------------------------------
        # CHANGED: scale all features properly
        self.scaler = StandardScaler()
        self.tabular_features: np.ndarray = self.scaler.fit_transform(
            df[ODIR_TABULAR_COLS]
        ).astype(np.float32)

        # ------------------------------------------------------------------
        # LABELS
        # ------------------------------------------------------------------
        self.labels: np.ndarray = df[ODIR_LABEL_COL].values.astype(np.int64)

        # CHANGED: no image paths needed
        self.image_paths = None

        self.tabular_input_dim: int = self.tabular_features.shape[1]  # now = 8

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ------------------------------------------------------------------
        # DUMMY IMAGE
        # ------------------------------------------------------------------
        # CHANGED: no real images → create blank image
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # ------------------------------------------------------------------
        # TABULAR + LABEL
        # ------------------------------------------------------------------
        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        label   = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, tabular, label