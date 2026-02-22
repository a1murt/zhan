"""
PyTorch Dataset for multimodal myopia progression data.

Expected DataFrame schema
-------------------------
Column          Type     Notes
-----------     ------   -----------------------------------------------
Patient_ID      str/int  Used for group-safe train/val splitting
Age             float    Numerical — imputed with median
Baseline_SE     float    Spherical equivalent (diopters) — imputed median
Axial_Length    float    mm — imputed median
Gender          str      Categorical ("M"/"F") — imputed most_frequent
Image_Path      str      Absolute or relative path to fundus photograph
Progression_Label int   0=Low, 1=Moderate, 2=High progression risk

Imputation is fit *only* on the training split and applied to val/test to
prevent data leakage.  Pass fit_imputers=False with pre-fitted objects for
downstream splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Column name constants (single source of truth for the entire project)
# ---------------------------------------------------------------------------
PATIENT_ID_COL    = "Patient_ID"
NUMERICAL_COLS    = ["Age", "Baseline_SE", "Axial_Length"]
CATEGORICAL_COLS  = ["Gender"]
LABEL_COL         = "Progression_Label"
IMAGE_COL         = "Image_Path"

# Feature columns in the order expected by the model's tabular branch
TABULAR_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS  # length 4


class MyopiaDataset(Dataset):
    """
    Multimodal dataset yielding (image_tensor, tabular_tensor, label) tuples.

    Parameters
    ----------
    dataframe         : pd.DataFrame matching the schema described above.
    transform         : Albumentations Compose pipeline (train or val).
    fit_imputers      : True  → fit new imputers/encoders on this DataFrame.
                        False → apply pre-fitted objects from the training set.
    numerical_imputer : Required when fit_imputers=False.
    categorical_imputer: Required when fit_imputers=False.
    label_encoders    : Required when fit_imputers=False.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[A.Compose] = None,
        fit_imputers: bool = True,
        numerical_imputer: Optional[SimpleImputer] = None,
        categorical_imputer: Optional[SimpleImputer] = None,
        label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    ) -> None:
        self.transform = transform
        df = dataframe.copy()

        # ------------------------------------------------------------------
        # 1. Impute numerical columns (median strategy)
        # ------------------------------------------------------------------
        if fit_imputers:
            self.numerical_imputer = SimpleImputer(strategy="median")
            df[NUMERICAL_COLS] = self.numerical_imputer.fit_transform(df[NUMERICAL_COLS])
        else:
            if numerical_imputer is None:
                raise ValueError(
                    "numerical_imputer must be provided when fit_imputers=False."
                )
            self.numerical_imputer = numerical_imputer
            df[NUMERICAL_COLS] = self.numerical_imputer.transform(df[NUMERICAL_COLS])

        # ------------------------------------------------------------------
        # 2. Impute categorical columns (most_frequent strategy)
        # ------------------------------------------------------------------
        if fit_imputers:
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            df[CATEGORICAL_COLS] = self.categorical_imputer.fit_transform(
                df[CATEGORICAL_COLS].astype(str)
            )
        else:
            if categorical_imputer is None:
                raise ValueError(
                    "categorical_imputer must be provided when fit_imputers=False."
                )
            self.categorical_imputer = categorical_imputer
            df[CATEGORICAL_COLS] = self.categorical_imputer.transform(
                df[CATEGORICAL_COLS].astype(str)
            )

        # ------------------------------------------------------------------
        # 3. Ordinal-encode categorical columns
        # ------------------------------------------------------------------
        if fit_imputers:
            self.label_encoders: Dict[str, LabelEncoder] = {}
            for col in CATEGORICAL_COLS:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        else:
            if label_encoders is None:
                raise ValueError(
                    "label_encoders must be provided when fit_imputers=False."
                )
            self.label_encoders = label_encoders
            for col in CATEGORICAL_COLS:
                le = self.label_encoders[col]
                df[col] = le.transform(df[col].astype(str))

        # ------------------------------------------------------------------
        # 4. Materialise arrays — avoids repeated DataFrame indexing in __getitem__
        # ------------------------------------------------------------------
        self.tabular_features: np.ndarray = (
            df[TABULAR_FEATURE_COLS].values.astype(np.float32)
        )
        self.labels: np.ndarray = df[LABEL_COL].values.astype(np.int64)
        self.image_paths: np.ndarray = df[IMAGE_COL].values

        # Expose to callers (e.g. model constructor, checkpoint saving)
        self.tabular_input_dim: int = self.tabular_features.shape[1]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---- Image -------------------------------------------------------
        image_path = str(self.image_paths[idx])
        raw = cv2.imread(image_path)
        if raw is None:
            raise FileNotFoundError(
                f"Could not read image at index {idx}: {image_path}"
            )
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]  # → CHW float32 tensor

        # ---- Tabular -----------------------------------------------------
        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)

        # ---- Label -------------------------------------------------------
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, tabular, label
