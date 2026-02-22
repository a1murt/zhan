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
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# ODIR-specific column constants
# ---------------------------------------------------------------------------
ODIR_PATIENT_ID_COL   = "Patient_ID"
ODIR_NUMERICAL_COLS   = ["Age"]
ODIR_CATEGORICAL_COLS = ["Gender"]      # stored as int 0/1 — LabelEncoder is a no-op
ODIR_LABEL_COL        = "Myopia"
ODIR_IMAGE_COL        = "Image_Path"   # full absolute path, built by launcher

ODIR_TABULAR_COLS     = ODIR_NUMERICAL_COLS + ODIR_CATEGORICAL_COLS  # dim = 2


class OdirDataset(Dataset):
    """
    Yields (image_tensor, tabular_tensor, label) tuples for the ODIR-5K task.

    Parameters
    ----------
    dataframe          : DataFrame with columns defined by ODIR_* constants above.
    transform          : Albumentations Compose pipeline.
    fit_imputers       : True on train split, False on val (pass fitted objects).
    numerical_imputer  : Pre-fitted SimpleImputer for Age.
    categorical_imputer: Pre-fitted SimpleImputer for Gender.
    label_encoders     : Dict of pre-fitted LabelEncoders keyed by column name.
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
        # Numerical imputation — Age (median)
        # ------------------------------------------------------------------
        if fit_imputers:
            self.numerical_imputer = SimpleImputer(strategy="median")
            df[ODIR_NUMERICAL_COLS] = self.numerical_imputer.fit_transform(
                df[ODIR_NUMERICAL_COLS]
            )
        else:
            if numerical_imputer is None:
                raise ValueError("Pass numerical_imputer when fit_imputers=False.")
            self.numerical_imputer = numerical_imputer
            df[ODIR_NUMERICAL_COLS] = self.numerical_imputer.transform(
                df[ODIR_NUMERICAL_COLS]
            )

        # ------------------------------------------------------------------
        # Categorical imputation — Gender (most_frequent)
        # Gender is already int 0/1; stringify so imputer is consistent.
        # ------------------------------------------------------------------
        if fit_imputers:
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            df[ODIR_CATEGORICAL_COLS] = self.categorical_imputer.fit_transform(
                df[ODIR_CATEGORICAL_COLS].astype(str)
            )
        else:
            if categorical_imputer is None:
                raise ValueError("Pass categorical_imputer when fit_imputers=False.")
            self.categorical_imputer = categorical_imputer
            df[ODIR_CATEGORICAL_COLS] = self.categorical_imputer.transform(
                df[ODIR_CATEGORICAL_COLS].astype(str)
            )

        # ------------------------------------------------------------------
        # Ordinal-encode Gender: "0"→0, "1"→1 (alphabetical → same order)
        # ------------------------------------------------------------------
        if fit_imputers:
            self.label_encoders: Dict[str, LabelEncoder] = {}
            for col in ODIR_CATEGORICAL_COLS:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        else:
            if label_encoders is None:
                raise ValueError("Pass label_encoders when fit_imputers=False.")
            self.label_encoders = label_encoders
            for col in ODIR_CATEGORICAL_COLS:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))

        # ------------------------------------------------------------------
        # Materialise as numpy arrays for zero-copy __getitem__
        # ------------------------------------------------------------------
        self.tabular_features: np.ndarray = (
            df[ODIR_TABULAR_COLS].values.astype(np.float32)
        )
        self.labels: np.ndarray = df[ODIR_LABEL_COL].values.astype(np.int64)
        self.image_paths: np.ndarray = df[ODIR_IMAGE_COL].values

        self.tabular_input_dim: int = self.tabular_features.shape[1]  # 2

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---- Image -------------------------------------------------------
        raw = cv2.imread(str(self.image_paths[idx]))
        if raw is None:
            raise FileNotFoundError(
                f"[OdirDataset] Cannot read image at index {idx}: "
                f"{self.image_paths[idx]}\n"
                "Ensure IMAGE_DIR in train_odir.py points to the folder "
                "containing the ODIR-5K fundus JPG files."
            )
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        tabular = torch.tensor(self.tabular_features[idx], dtype=torch.float32)
        label   = torch.tensor(self.labels[idx],           dtype=torch.long)
        return image, tabular, label
